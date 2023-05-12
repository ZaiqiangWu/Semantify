import argparse
from pathlib import Path


def main(args):
    gender = "male" if args.gender == "m" else "female"
    for file in Path(args.working_dir).rglob("*.png"):
        if gender not in file.stem:
            file.rename(file.as_posix().replace(".png", f"_{gender}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", type=str)
    parser.add_argument("-g", "--gender", type=str, choices=["m", "f"])
    args = parser.parse_args()
    main(args)
