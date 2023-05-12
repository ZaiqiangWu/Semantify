import cv2
import json
import torch
import hydra
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from pytorch3d.structures import Meshes
from typing import Dict, Tuple, Literal
from clip2mesh.comparisons.comparison_utils import ComparisonUtils


class ZjuMocapComparison(ComparisonUtils):
    def __init__(self, args):
        super().__init__(**args)
        self._load_gt_jsons()

    def _load_gt_jsons(self, file_name: str = "smpl_params.npy"):
        self.gt_jsons = {}
        files_generator = list(self.gt_dir.rglob(file_name))
        for file in files_generator:
            subject_id = file.parents[0].name
            self.gt_jsons[subject_id] = np.load(file, allow_pickle=True).tolist()["shapes"]

    def get_gt_data(self, json_id: str, frame_idx: int) -> torch.Tensor:
        data = self.gt_jsons[json_id]
        return torch.tensor(data[frame_idx])[None]

    def get_body_shape(
        self, raw_img_path: Path, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """ """
        img_id = raw_img_path.stem
        camera_id = raw_img_path.parents[0].name
        subject_id = raw_img_path.parents[1].name

        # load raw image
        raw_img = cv2.imread(str(raw_img_path))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # ground truth shape
        gt_body_shape = self.get_gt_data(subject_id, int(img_id))

        # shapy prediction
        shpay_npz_path = self.comparison_dirs["shapy"] / subject_id / camera_id / f"{img_id}.npz"
        shapy_data = self.get_shapy_data(shpay_npz_path)
        shapy_body_shape = torch.tensor(shapy_data["betas"])[None]

        # spin prediction
        spin_npy_path = self.comparison_dirs["spin"] / subject_id / camera_id / f"{img_id}.npy"
        spin_body_shape = self.get_spin_data(spin_npy_path)

        # clip preprocess
        encoded_image = self.clip_preprocess(Image.fromarray(raw_img)).unsqueeze(0).to(self.device)

        # our prediction
        with torch.no_grad():
            clip_scores = self.clip_model(encoded_image, self.encoded_labels[gender])[0].float()
            # clip_scores = self.normalize_scores(clip_scores, gender)
            our_body_shape = self.model[gender](clip_scores).cpu()

        return {
            "shapy": shapy_body_shape,
            # "pixie": pixie_body_shape,
            "spin": spin_body_shape,
            "ours": our_body_shape,
            # "body_pose": shapy_body_pose,
            "gt": gt_body_shape,
        }, raw_img

    def __call__(self):
        max_num_imgs = 20
        for subject_id in self.raw_imgs_dir.iterdir():
            if not subject_id.name == "377":
                continue
            for camera_id in subject_id.iterdir():
                images_counter = 0
                if camera_id.name == "smpl_params.npy":
                    continue
                for raw_img_path in camera_id.iterdir():
                    if raw_img_path.suffix != ".jpg":
                        continue

                    self.logger.info(f"Processing {raw_img_path}...\n")

                    output_path = self.output_path / subject_id.name / camera_id.name / raw_img_path.stem
                    output_path.mkdir(parents=True, exist_ok=True)

                    gender = "male"

                    body_shapes, raw_img = self.get_body_shape(raw_img_path, gender)

                    l2_losses: Dict[str, torch.Tensor] = self.calc_l2_distances(body_shapes)

                    smplx_args: Dict[
                        str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                    ] = self.get_smplx_kwargs(body_shapes, gender)

                    meshes: Dict[str, Meshes] = self.get_meshes_from_shapes(smplx_args)

                    chamfer_distances: Dict[str, torch.Tensor] = self.calc_chamfer_distance(meshes)

                    frames_dir = output_path / "frames"
                    frames_dir.mkdir(exist_ok=True)

                    num_methods = len(meshes)
                    num_blocks = num_methods + 1  # +1 for the raw image
                    video_struct = self.get_video_structure(num_blocks)
                    video_shape = (self.renderer.height * video_struct[0], self.renderer.width * video_struct[1])

                    # create video from multiview data
                    if raw_img.shape[:2] != (self.renderer.height, self.renderer.width):
                        raw_img = cv2.resize(raw_img, (self.renderer.width, self.renderer.height))

                    smplx_kwargs: Dict[str, Dict[str, np.ndarray]] = self.mesh_attributes_to_kwargs(
                        smplx_args, to_tensor=True
                    )

                    self.multiview_data(frames_dir, smplx_kwargs, video_struct, raw_img)
                    self.create_video_from_dir(frames_dir, video_shape)

                    # columns are: ["image_name", "loss", "shapy", "pixie", "spin", "ours"]
                    single_img_results = pd.DataFrame.from_dict(
                        {
                            "image_name": [f"{subject_id.name}_{camera_id.name}_{raw_img_path.stem}"],
                            "loss": "chamfer distance",
                            **chamfer_distances,
                        }
                    )
                    self.results_df = pd.concat([self.results_df, single_img_results])

                    self.results_df.to_csv(
                        self.output_path / subject_id.name / camera_id.name / "results.csv", index=False
                    )
                    images_counter += 1
                    if images_counter >= max_num_imgs:
                        break


@hydra.main(config_path="../config", config_name="zjumocap_comparison")
def main(cfg: DictConfig):
    zju_comp = ZjuMocapComparison(cfg.comparison_kwargs)
    zju_comp()


if __name__ == "__main__":
    main()
