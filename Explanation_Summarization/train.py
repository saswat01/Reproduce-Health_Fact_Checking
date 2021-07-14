from model import BartModel
import config
import sys
import argparse

def _parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='bart', choices=['bart', 'T5'],  help="The model to be trained")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--model_path", required="--save_model" in sys.argv)
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    Model = BartModel(config.train_path, config.eval_path, config.test_path, config.model_name[args.model])
    Model.train(args.epochs)
    if args.save_model:
        Model.save_model(args.model_path)

if __name__ == "__main__":
    main()
    