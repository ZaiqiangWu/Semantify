import cv2
import torch
import hydra
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from torch.nn import functional as F
from pytorch3d.structures import Meshes
from typing import Dict, Tuple, Literal
from clip2mesh.comparisons.comparison_utils import ComparisonUtils


class NeuralBodyComparison(ComparisonUtils):
    def __init__(self, args):
        super().__init__(**args)

    @staticmethod
    def get_gt_data(file_path: Path, betas: bool = False) -> torch.Tensor:
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        # data = h5py.File(h5_path, "r")
        if betas:
            return torch.tensor(data["betas"])[None].float()
        return torch.tensor(data["v_personal"], device="cuda")

    def get_body_shapes(
        self, raw_img_path: Path, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        img_id = raw_img_path.stem
        person_id = raw_img_path.parents[1].name

        # load raw image
        raw_img = cv2.imread(str(raw_img_path))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # ground truth shape
        nb_h5_path = self.gt_dir / person_id / "consensus.pkl"
        gt_body_shape = self.get_gt_data(nb_h5_path, betas=True)

        # # shapy prediction
        # shpay_npz_path = self.comparison_dirs["shapy"] / person_id / f"{img_id}.npz"
        # shapy_data = self.get_shapy_data(shpay_npz_path)
        # shapy_body_shape = torch.tensor(shapy_data["betas"])[None]  # TODO: maybe convert to tensor or add dimension
        # shapy_body_pose = torch.tensor(shapy_data["body_pose"])[None]

        # # pixie prediction
        # pixie_pkl_path = self.comparison_dirs["pixie"] / person_id / f"{img_id}_param.pkl"
        # pixie_body_shape = self.get_pixie_data(pixie_pkl_path)

        # spin prediction
        spin_npy_path = self.comparison_dirs["spin"] / person_id / f"{img_id}.npy"
        spin_body_shape = self.get_spin_data(spin_npy_path)

        # clip preprocess
        encoded_image = self.clip_preprocess(Image.fromarray(raw_img)).unsqueeze(0).to(self.device)

        # our prediction
        with torch.no_grad():
            clip_scores = self.clip_model(encoded_image, self.encoded_labels[gender])[0].float()
            our_body_shape = self.model[gender](clip_scores).cpu()

        return {
            # "shapy": shapy_body_shape,
            # "pixie": pixie_body_shape,
            "spin": spin_body_shape,
            "ours": our_body_shape,
            # "body_pose": shapy_body_pose,
            "gt": gt_body_shape,
        }, raw_img

    def calc_verts_distances(self, meshes: Dict[str, Meshes]) -> Dict[str, torch.Tensor]:
        l2_vertices = {}
        for method, mesh in meshes.items():
            if method == "gt":
                continue
            l2_vertices[method] = F.mse_loss(mesh.verts_packed(), meshes["gt"].verts_packed()).sqrt().item()
        return l2_vertices

    def __call__(self):

        for person_id in self.raw_imgs_dir.iterdir():
            frames_dir = self.raw_imgs_dir / person_id.name / f"{person_id.name}_frames"
            images_counter = 0
            images_generator = sorted(list(frames_dir.iterdir()), key=lambda x: int(x.stem))[::50]
            for raw_img_path in images_generator:

                self.logger.info(f"Processing {raw_img_path.name}...\n")

                output_path = self.output_path / person_id.name / raw_img_path.stem
                output_path.mkdir(exist_ok=True, parents=True)

                gender = raw_img_path.parent.name.split("-")[0]
                # gender = "neutral"

                body_shapes, raw_img = self.get_body_shapes(raw_img_path, "neutral")
                # body_pose: torch.Tensor = body_shapes.pop("body_pose")

                l2_shape: Dict[str, torch.Tensor] = self.calc_l2_distances(body_shapes)

                # if not (output_path / "out_vid.mp4").exists():
                #     self.logger.info(f"Video for {raw_img_path.name} already exists. Skipping...\n")

                smplx_args: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = self.get_smplx_kwargs(
                    body_shapes, gender, get_smpl=True
                )

                meshes: Dict[str, Meshes] = self.get_meshes_from_shapes(smplx_args)
                # TODO: add chamfer loss with and without body pose

                gt_verts = self.get_gt_data(self.gt_dir / person_id.name / "consensus.pkl")

                l2_vertices: Dict[str, torch.Tensor] = self.calc_verts_distances(meshes)

                frames_dir = output_path / "frames"
                frames_dir.mkdir(exist_ok=True)

                num_methods = len(meshes)
                num_blocks = num_methods + 1  # +1 because we have also the raw image
                video_struct = self.get_video_structure(num_blocks)
                video_shape = (self.renderer.height * video_struct[0], self.renderer.width * video_struct[1])
                #
                # create video from multiview data
                if raw_img.shape[:2] != (self.renderer.height, self.renderer.width):
                    raw_img = cv2.resize(raw_img, (self.renderer.width, self.renderer.height))

                smplx_kwargs: Dict[str, Dict[str, np.ndarray]] = self.mesh_attributes_to_kwargs(
                    smplx_args, to_tensor=True
                )
                self.multiview_data(frames_dir, smplx_kwargs, video_struct, raw_img)
                self.create_video_from_dir(frames_dir, video_shape)

                # columns are: ["image_name", "loss", "shapy", "pixie", "spin", "ours"]
                single_img_l2_results = pd.DataFrame.from_dict(
                    {
                        "image_name": [f"{raw_img_path.parent.name.split('_')[0]}_{raw_img_path.stem}"],
                        "loss": "l2_shape",
                        **l2_shape,
                    }
                )
                single_img_chamfer_results = pd.DataFrame.from_dict(
                    {
                        "image_name": [f"{raw_img_path.parent.name.split('_')[0]}_{raw_img_path.stem}"],
                        "loss": "l2_vertices",
                        **l2_vertices,
                    }
                )

                self.results_df = pd.concat([self.results_df, single_img_l2_results, single_img_chamfer_results])

                # images_counter += 1
                # if images_counter == max_num_imgs:
                #     break

                self.results_df.to_csv(output_path.parent / "results.csv", index=False)


@hydra.main(config_path="../config", config_name="neuralbody_comparison")
def main(cfg: DictConfig) -> None:
    neuralbody_comparison = NeuralBodyComparison(cfg.comparison_kwargs)
    neuralbody_comparison()


if __name__ == "__main__":
    main()
