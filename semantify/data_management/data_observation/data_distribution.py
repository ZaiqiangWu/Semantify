import json
import argparse
import logging
from pathlib import Path
from clip2mesh.utils import Utils


def main(args):
    utils = Utils()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
    logger = logging.getLogger(__name__)
    data_dir = args.data_dir
    files_generator = list(Path(data_dir).rglob("*_labels.json"))

    logger.info("starting to iterate over images")
    labels = utils.get_labels()
    min_max_dict = utils.get_min_max_values(data_dir)
    classes_dict = {label[0]: 0 for label in labels}

    for file in files_generator:
        with open(file.as_posix(), "r") as f:
            data = json.load(f)
        for key, value in data.items():
            data[key] = utils.normalize_data({key: value}, min_max_dict)[key]
        max_class = max(data, key=data.get)
        classes_dict[max_class] += 1

    logger.info(f"classes distribution: {classes_dict}")
    logger.info(f"total number of images: {len(files_generator)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args)
