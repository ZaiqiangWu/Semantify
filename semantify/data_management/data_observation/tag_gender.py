import cv2
import argparse
from pathlib import Path


def main(args):
    working_dir = Path(args.working_dir)
    for image_path in working_dir.rglob(f"*.{args.suffix}"):

        if "female" in image_path.stem or "male" in image_path.stem:
            continue

        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (512, 512))
        cv2.imshow("image", image)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

        if key == ord("m"):
            image_path.rename(image_path.as_posix().replace(f".{args.suffix}", f"_male.{args.suffix}"))

        if key == ord("f"):
            image_path.rename(image_path.as_posix().replace(f".{args.suffix}", f"_female.{args.suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", type=str)
    parser.add_argument("-s", "--suffix", type=str, default="png")
    args = parser.parse_args()
    main(args)
