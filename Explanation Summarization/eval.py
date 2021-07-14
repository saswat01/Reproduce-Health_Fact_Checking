import argparse
from model import BartModel


def main():
    args = _parse_args()
    model = BartModel(eval=True)
    model.load_model(args.path)
    model.eval()

def _parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,  help="The path to model checkpoint")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()