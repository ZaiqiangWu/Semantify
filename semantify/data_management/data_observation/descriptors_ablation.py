import json
import hydra
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
from typing import List, Literal
from clip2mesh.optimizations.train_mapper import train
from clip2mesh.data_management.data_observation.evaluate_performance import EvaluatePerformance
from clip2mesh.data_management.data_observation.choosing_descriptors import ChoosingDescriptors


class DescriptorsAblation:
    def __init__(
        self,
        descriptors_options: List[int],
        model_type: Literal["smplx", "flame", "smal"],
        gender: Literal["male", "female", "neutral"],
        method: Literal["diff_coords", "L2"],
        mode: Literal["run", "eval"],
        data_path: str,
        output_path: str,
        descriptors_clusters_json: str,
        batch_size: int,
        renderer_kwargs: DictConfig,
        train_kwargs: DictConfig,
        optimize_features: Literal["betas", "flame_expression", "flame_shape", "beta"],
        min_slider_value: int = 15,
        max_slider_value: int = 30,
        effect_threshold: float = 0.5,
        models_dir: str = "/home/nadav2/dev/repos/Thesis/pre_production",
    ):
        self.descriptors_options = descriptors_options
        self.model_type = model_type
        self.gender = gender
        self.mode = mode
        self.method = method
        self.batch_size = batch_size
        self.train_kwargs = train_kwargs
        self.optimize_features = optimize_features
        self.min_slider_value = min_slider_value
        self.max_slider_value = max_slider_value
        self.effect_threshold = effect_threshold
        self.renderer_kwargs = renderer_kwargs
        self.models_dir: Path = Path(models_dir)
        self.output_path: Path = Path(output_path)
        self.data_path: Path = Path(data_path)
        self.descriptors_clusters_json: Path = Path(descriptors_clusters_json)

        self._get_logger()

    def _get_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    def get_descriptors(self, num_of_descriptors: int) -> List[str]:
        choosing_descriptors = ChoosingDescriptors(
            images_dir=self.data_path,
            max_num_of_descriptors=num_of_descriptors,
            min_num_of_descriptors=num_of_descriptors,
            descriptors_clusters_json=self.descriptors_clusters_json,
        )
        descriptors = choosing_descriptors.choose()
        descriptors = list(choosing_descriptors.flatten_dict_of_dicts(descriptors).keys())
        return descriptors

    def train_mapper(self, descriptors: List[str]) -> str:
        run_name = f"{self.model_type}_{self.gender}_{len(descriptors)}_descriptors"
        self.train_kwargs.tensorboard_logger.name = run_name
        self.train_kwargs.dataloader.batch_size = self.batch_size
        self.train_kwargs.dataset.data_dir = self.data_path.as_posix()
        self.train_kwargs.dataset.optimize_features = (
            [self.optimize_features] if isinstance(self.optimize_features, str) else self.optimize_features
        )
        self.train_kwargs.dataset.labels_to_get = descriptors
        if self.mode == "run":
            train(self.train_kwargs)
        return run_name

    def evaluate_performance(self, run_name: str):
        evaluator = EvaluatePerformance(
            model_type=self.model_type,
            min_value=self.min_slider_value,
            max_value=self.max_slider_value,
            effect_threshold=self.effect_threshold,
            renderer_kwargs=self.renderer_kwargs,
            method=self.method,
            model_path=(self.models_dir / f"{run_name}.ckpt").as_posix(),
            out_path=self.output_path.as_posix(),
            optimize_feature=self.optimize_features,
            gender=self.gender,
        )
        evaluator.evaluate()

    def visualize_performance(self, run_names: List[str]):
        results = {"vertices coverage": [], "num_of_descriptors": [], "overlap": []}

        for run_name in run_names:
            with open(self.output_path / run_name / "descriptors.json", "r") as f:
                res = json.load(f)
            results["vertices coverage"].append(res["iou"] * 100)
            results["num_of_descriptors"].append(res["labels"].__len__())
            results["overlap"].append(res["overlap"] * 100)

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax1.plot(results["num_of_descriptors"], results["vertices coverage"], label="vertices coverage", color="blue")
        ax2 = ax1.twinx()
        ax1.set_xticks(results["num_of_descriptors"])
        ax1.set_xlabel("# Descriptors")
        ax1.set_ylabel("vertices coverage")
        ax2.plot(results["num_of_descriptors"], results["overlap"], label="Overlap", color="green")
        ax2.set_ylabel("Overlap")
        fig.suptitle(f"Performance by # Descriptors - {self.model_type} {self.gender}")
        fig.tight_layout()
        fig.legend(loc="lower left", ncol=2)

        # save the plot
        fig.savefig(self.output_path / f"evaluation_{self.model_type}_{self.gender}.png")

    def run(self):

        self.logger.info(f"Starting descriptors ablation for {self.model_type} {self.gender} model")
        run_names = []
        for num_of_descriptors in self.descriptors_options:
            self.logger.info(f"Starting with {num_of_descriptors} descriptors")
            descriptors = self.get_descriptors(num_of_descriptors)
            self.logger.info(f"Chosen descriptors: {descriptors}")
            self.logger.info(f"Starting training")
            run_name = self.train_mapper(descriptors)
            run_names.append(run_name)
            self.logger.info(f"Starting evaluation")
            self.evaluate_performance(run_name)
            self.logger.info(f"Finished with {num_of_descriptors} descriptors")
            print()
        self.logger.info(f"Finished descriptors ablation for {self.model_type} {self.gender}")

        return run_names


@hydra.main(config_path="../../config", config_name="descriptors_ablation")
def main(cfg: DictConfig):
    descriptors_ablation = DescriptorsAblation(**cfg)
    descriptors_ablation.run()


if __name__ == "__main__":
    main()
