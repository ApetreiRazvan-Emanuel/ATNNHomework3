import argparse

import wandb
import json
from pipeline import TrainingPipeline

WANDB_PROJECT = "cifar100-sweep"
WANDB_ENTITY = "apetreirazvane-facultate"

sweep_configuration = {
    'method': 'grid',
    'project': WANDB_PROJECT,
    'entity': WANDB_ENTITY,
    'name': 'cifar100-sweep',
    'metric': {
        'name': 'val/accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'model': {
            'values': [
                {'model_name': 'resnet18_cifar100'},
                {'model_name': 'PreActResNet18-CIFAR100'}
            ]
        },
        'transform': {
            'values': [
                {
                    'RandomCrop': {'size': 32, 'padding': 4},
                    'RandomHorizontalFlip': None
                },
                {
                    'RandomCrop': {'size': 32, 'padding': 4},
                    'RandomHorizontalFlip': None,
                    'RandomRotation': {'degrees': 15},
                    'RandomErasing': {'p': 0.5}
                }
            ]
        },
        'optimizer': {
            'values': [
                {
                    'type': 'SGD',
                    'params': {
                        'lr': 0.1,
                        'momentum': 0.9,
                        'weight_decay': 5e-4,
                        'nesterov': True
                    }
                },
                {
                    'type': 'AdamW',
                    'params': {
                        'lr': 0.001,
                        'weight_decay': 0.05
                    }
                }
            ]
        }
    }
}


def get_run_name(config):
    model_name = config['model']['model_name'].split('_')[0]
    optimizer_name = config['optimizer']['type']
    aug_type = "RandomErasing" if "RandomErasing" in config['transform'] else "NoRandomErasing"

    return f"{model_name}-{optimizer_name}-{aug_type}"


def train():
    with wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY) as run:
        with open('config.json', 'r') as f:
            config = json.load(f)

        config['model'] = run.config['model']
        config['transform'] = run.config['transform']
        config['optimizer'] = run.config['optimizer']

        wandb.run.name = get_run_name(config)
        wandb.run.save()

        pipeline = TrainingPipeline(config)
        train_acc, val_acc = pipeline.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, help='Existing sweep ID (if continuing)')
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep_configuration, project=WANDB_PROJECT, entity=WANDB_ENTITY)
        print(f"Created sweep with ID: {sweep_id}")

    wandb.agent(
        sweep_id,
        function=train,
        count=4,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )


if __name__ == "__main__":
    main()
