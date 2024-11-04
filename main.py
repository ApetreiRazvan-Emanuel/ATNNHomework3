import argparse
from pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description='Homework3 Apetrei Razvan-Emanuel Training Pipeline')
    parser.add_argument('--config', type=str, help='path to config file', default="config.json")
    args = parser.parse_args()

    pipeline = TrainingPipeline(args.config)
    best_train_acc, best_val_acc = pipeline.train()

    print(f"\nTraining completed!")
    print(f"Best training accuracy: {best_train_acc:.2f}%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
