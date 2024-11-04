import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from constants import *
import json


class CachedDataset(Dataset):
    def __init__(self, dataset_type: str, train: bool = True, transform = None):
        if dataset_type not in dataset_classes:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        transform_list = [transforms.ToTensor()]
        if transform:
            transform_list.append(transform)
        if dataset_type in normalization_values:
            mean, std = normalization_values[dataset_type]
            transform_list.append(transforms.Normalize(mean=mean, std=std))

        self.transform = transforms.Compose(transform_list)
        self.dataset = dataset_classes[dataset_type](
            root="./data",
            download=True,
            train=train,
            transform=self.transform
        )
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


class EarlyStoppingMechanism:
    def __init__(self, criterion: str = "loss", patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.criterion = criterion
        self.best_score = None

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        score_improved = (current_score < self.best_score) if self.criterion == 'loss' else (
                current_score > self.best_score)

        if score_improved:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Stopped early after {self.patience} epochs.")
                return True
            return False


def get_transforms(config_transform: dict):
    transform_list = []
    for transform_name, transform_params in config_transform.items():
        if transform_name in transform_methods:
            if transform_params:
                try:
                    transform_list.append(transform_methods[transform_name](**transform_params))
                except TypeError as e:
                    print(f"An error occurred while initializing {transform_name} with parameters {transform_params}: {e}")
            else:
                transform_list.append(transform_methods[transform_name]())
        else:
            print(f"Warning: {transform_name} is not a supported transformation! It will be ignored")

    return transforms.Compose(transform_list)


def get_optimizer(model_params, config_optimizer: dict):
    optim_type = config_optimizer.get("type", "SGD")
    optim_params = config_optimizer.get("params", {"lr": 0.1,
                                                   "momentum": 0.9,
                                                   "weight_decay": 5e-4})

    if optim_type not in optimizers:
        raise ValueError(f"Invalid optimizer type: {optim_type}")

    try:
        optimizer = optimizers[optim_type](model_params, **optim_params)
    except TypeError as e:
        raise ValueError(
            f"An error occurred while initializing optimizer: '{optim_type}' with parameters: "
            f"{optim_params}. Further details: {e}"
        )

    return optimizer


def get_scheduler(optimizer, config_scheduler: dict):
    scheduler_type = config_scheduler.get("type")
    if scheduler_type is None:
        return None

    if scheduler_type not in schedulers:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    scheduler_params = config_scheduler.get("params", {})

    try:
        scheduler = schedulers[scheduler_type](optimizer, **scheduler_params)
    except TypeError as e:
        raise ValueError(
            f"An error occurred while initializing scheduler: '{scheduler_type}' with parameters: "
            f"{scheduler_params}. Further details: {e}"
        )

    return scheduler


def get_loss_function(config_loss: str):
    if config_loss not in loss_functions:
        raise ValueError(f"Invalid loss function type: {config_loss}")

    return loss_functions[config_loss]()


def load_model(config_model: dict):
    if config_model["model_name"] not in available_models:
        raise ValueError(f"Invalid model: {config_model['model_name']}")

    return available_models[config_model["model_name"]]()


class TrainingPipeline:
    def __init__(self, configuration):
        if isinstance(configuration, str):
            with open(configuration) as f:
                self.config = json.load(f)
        else:
            self.config = configuration

        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        print(f"Running on: {self.device}")
        self.transform = get_transforms(self.config['transform'])
        self.train_dataset = CachedDataset(self.config['dataset'], train=True, transform=self.transform)
        self.test_dataset = CachedDataset(self.config['dataset'], train=False)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config["batch_size"]["train"] or 128,
            shuffle=self.config["shuffle"]["train"] or True,
            num_workers=self.config.get("num_workers", 8),
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config["batch_size"]["test"] or 128,
            shuffle=self.config["shuffle"]["test"] or False,
            num_workers=self.config.get("num_workers", 8),
            pin_memory=True
        )

        self.model = load_model(self.config["model"])
        self.optimizer = get_optimizer(self.model.parameters(), self.config["optimizer"])
        self.scheduler = get_scheduler(self.optimizer, self.config["scheduler"])
        self.loss_function = get_loss_function(self.config["loss"])
        self.epochs = self.config.get("epochs", 150)

        log_dir = self.config.get("logging", {}).get("log_dir", "./runs")
        transform_name = "RandomErasing" if "RandomErasing" in self.config['transform'] else "NoRandomErasing"
        run_name = f"{self.config['model']['model_name']}_{self.config['optimizer']['type']}_{transform_name}"
        self.writer = SummaryWriter(f"{log_dir}/{run_name}")

        if wandb.run is None:
            wandb.init(
                project="cifar100-sweep",
                entity="apetreirazvane-facultate",
                name=f"{self.config['model']['model_name']}_{self.config['optimizer']['type']}_{transform_name}",
                config=self.config
            )
            wandb.watch(self.model, log="all")

        early_stopping_config = self.config.get("early_stopping", {})
        self.early_stopping_mechanism = EarlyStoppingMechanism(
            patience=early_stopping_config.get("patience", 10),
            criterion=early_stopping_config.get("criterion", "loss"),
        )

        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self):
        try:
            self.model.to(self.device)
            best_train_acc = 0.0
            best_val_acc = 0.0

            use_amp = self.config.get("use_amp", False) and self.device.type == 'cuda'
            scaler = torch.amp.GradScaler("cuda") if use_amp else None

            transform_name = "RandomErasing" if "RandomErasing" in self.config['transform'] else "NoRandomErasing"
            print(f"Starting the training for: {self.config['model']['model_name']}_{self.config['optimizer']['type']}_{transform_name}")
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0.0
                correct = 0
                total = 0

                for i, (images, labels) in enumerate(self.train_dataloader):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    is_accumulation_step = (i + 1) % self.gradient_accumulation_steps != 0
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = self.model(images)
                        loss = self.loss_function(outputs, labels)
                        loss = loss / self.gradient_accumulation_steps

                    if use_amp:
                        scaler.scale(loss).backward()
                        if not is_accumulation_step:
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                    else:
                        loss.backward()
                        if not is_accumulation_step:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    total_loss += loss.item() * images.size(0)
                    predicted = outputs.argmax(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    del outputs
                    if not is_accumulation_step:
                        torch.cuda.empty_cache()

                avg_loss = total_loss / total
                train_acc = 100.0 * correct / total

                if train_acc > best_train_acc:
                    best_train_acc = train_acc

                val_loss, val_acc = self.validate(epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler is not None:
                    self.scheduler.step()

                if self.early_stopping_mechanism(val_loss if self.early_stopping_mechanism.criterion == 'loss' else val_acc):
                    print(f"Early Stopping Mechanism activated at {epoch + 1}")
                    break

                metrics = {
                    'epoch': epoch,
                    'train/loss': avg_loss,
                    'train/accuracy': train_acc,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'best_val_accuracy': best_val_acc
                }

                self.writer.add_scalar('Loss/train', avg_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

                wandb.log(metrics)

                # tbar.set_description(
                #     f"Epoch {epoch + 1}: "
                #     f"Train Loss: {avg_loss:.2f}, "
                #     f"Train Acc: {train_acc:.2f}%, "
                #     f"Val Loss: {val_loss:.2f}, "
                #     f"Val Acc: {val_acc:.2f}%, "
                #     f"Best Val acc: {best_val_acc:.2f}%")
                print(f"Epoch {epoch + 1}: "
                      f"Train Loss: {avg_loss:.2f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.2f}, "
                      f"Val Acc: {val_acc:.2f}%, "
                      f"Best Val acc: {best_val_acc:.2f}%")

            # dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            # self.writer.add_graph(self.model, dummy_input)

            self.writer.close()

            return best_train_acc, best_val_acc
        except Exception as e:
            print(f"Training error: {str(e)}")
            torch.cuda.empty_cache()
            raise e
        finally:
            torch.cuda.empty_cache()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        tta_enabled = self.config.get("use_tta", False)
        tta_repeats = self.config.get("tta_repeats", 5)

        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if tta_enabled:
                    batch_predictions = []

                    for _ in range(tta_repeats):
                        augmented_images = torch.stack([
                            self.transform(image.cpu()) for image in images
                        ]).to(self.device)

                        outputs = self.model(augmented_images)
                        batch_predictions.append(outputs)

                    outputs = torch.stack(batch_predictions).mean(dim=0)
                else:
                    outputs = self.model(images)

                loss = self.loss_function(outputs, labels)
                total_loss += loss.item() * images.size(0)
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()
        wandb.finish()
