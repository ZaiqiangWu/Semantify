import cv2
import clip
import torch
import hydra
import logging
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from torch.nn import functional as F
from pytorch3d.structures import Meshes
from typing import Dict, Tuple, Literal
from pytorch3d.loss import chamfer_distance
from pytorch3d.io import load_objs_as_meshes
from clip2mesh.utils import Image2ShapeUtils, Utils


class CelebComparison(Image2ShapeUtils):
    def __init__(
        self,
        smplx_models: Dict[str, str],
        raw_imgs_dir: str,
        output_path: str,
        comparison_dirs: Dict[str, str],
        renderer_kwargs: Dict[str, Dict[str, float]],
        num_coeffs: int = 10,
    ):
        super().__init__()
        self.raw_imgs_dir: Path = Path(raw_imgs_dir)
        self.comparison_dirs: Dict[str, str] = comparison_dirs
        self.output_path: Path = Path(output_path)
        self.num_coeffs = num_coeffs
        self.utils: Utils = Utils(comparison_mode=True)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_renderer(renderer_kwargs)
        self.load_smplx_model(smplx_models)
        self._load_clip_model()
        self._perpare_comparisons_dir()
        self._load_logger()

    def _get_gt_data(self, npy_path: Path) -> Tuple[np.ndarray, Meshes]:
        verts = torch.tensor(np.load(npy_path))
        mesh = load_objs_as_meshes([npy_path.with_suffix(".obj")], device=self.device)
        return verts, mesh

    @staticmethod
    def get_shapy_data(npz_path: Path) -> Dict[str, torch.Tensor]:
        """Load shapy predictions from pre-processed npz file"""
        relevant_keys = ["body_pose", "betas"]
        data = np.load(npz_path, allow_pickle=True)
        return {k: torch.from_numpy(v) for k, v in data.items() if k in relevant_keys}

    def _perpare_comparisons_dir(self):
        """Create a directory for the comparison results"""
        self.comparison_dirs = {k: Path(v) for k, v in self.comparison_dirs.items()}

    def get_smplx_kwargs(
        self, body_shapes: Dict[str, torch.Tensor], gender: Literal["male", "female", "neutral"]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the smplx kwargs for the different methods -> (vertices, faces, vt, ft)"""
        smplx_kwargs = {}
        for method, body_shape in body_shapes.items():
            real_gender = gender if method == "ours_gender" else "neutral"
            verts, faces, vt, ft = self._get_smplx_attributes(body_shape, real_gender)
            smplx_kwargs[method] = (verts, faces, vt, ft)
        return smplx_kwargs

    def _load_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    def load_smplx_model(self, models_paths: Dict[str, str]):
        smplx_models = {}
        for gender, model_path in models_paths.items():
            smplx_models[gender] = {"model": None, "labels": None}
            model, labels = self.utils.get_model_to_eval(model_path)
            labels = self._flatten_list_of_lists(labels)
            smplx_models[gender]["model"] = model
            smplx_models[gender]["labels"] = clip.tokenize(labels).to(self.device)
        self.smplx_models = smplx_models

    def get_meshes_from_shapes(
        self, smplx_kwargs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> Dict[str, Meshes]:
        """Get the meshes from the smplx kwargs"""
        meshes = {}
        for method, args in smplx_kwargs.items():
            meshes[method] = self.renderer.get_mesh(*args)
        return meshes

    @staticmethod
    def get_pixie_data(pkl_path: Path) -> torch.Tensor:
        """Load pixie predictions from pre-processed pkl file"""
        data = np.load(pkl_path, allow_pickle=True)
        return torch.tensor(data["shape"][:10])[None]

    def get_body_shapes(
        self, raw_img_path: Path, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        img_id = raw_img_path.stem
        person_id = raw_img_path.parents[1].name

        # load raw image
        raw_img = cv2.imread(str(raw_img_path))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # pixie prediction
        pixie_pkl_path = self.comparison_dirs["pixie"] / img_id / f"{img_id}_param.pkl"
        pixie_body_shape = self.get_pixie_data(pixie_pkl_path)

        # shapy prediction
        shpay_npz_path = self.comparison_dirs["shapy"] / f"{img_id}.npz"
        shapy_data = self.get_shapy_data(shpay_npz_path)
        shapy_body_shape = torch.tensor(shapy_data["betas"])[None]  # TODO: maybe convert to tensor or add dimension

        # clip preprocess
        encoded_image = self.clip_preprocess(Image.fromarray(raw_img)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # our prediction - neutral
            clip_scores = self.clip_model(encoded_image, self.smplx_models["neutral"]["labels"])[0].float()
            our_neutral_body_shape = self.smplx_models["neutral"]["model"](clip_scores).cpu()
            if our_neutral_body_shape.shape != (1, self.num_coeffs):
                our_neutral_body_shape = torch.cat(
                    [our_neutral_body_shape, torch.zeros(1, self.num_coeffs - our_neutral_body_shape.shape[1])], dim=1
                )

            # out prediction - real gender
            clip_scores = self.clip_model(encoded_image, self.smplx_models[gender]["labels"])[0].float()
            our_real_gender_body_shape = self.smplx_models[gender]["model"](clip_scores).cpu()
            if our_real_gender_body_shape.shape != (1, self.num_coeffs):
                our_real_gender_body_shape = torch.cat(
                    [our_real_gender_body_shape, torch.zeros(1, self.num_coeffs - our_real_gender_body_shape.shape[1])],
                    dim=1,
                )

        return {
            "shapy": shapy_body_shape,
            "ours": our_neutral_body_shape,
            "pixie": pixie_body_shape,
            "ours_gender": our_real_gender_body_shape,
        }, raw_img

    def get_rendered_images(
        self, smplx_kwargs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], angle: float
    ) -> Dict[str, np.ndarray]:
        """Render the meshes for the different methods"""
        rendered_imgs = {}
        for method, kwargs in smplx_kwargs.items():
            kwargs.update({"rotate_mesh": {"degrees": float(angle), "axis": "y"}})
            rendered_img = self.renderer.render_mesh(**kwargs)
            rendered_imgs[method] = self.adjust_rendered_img(rendered_img)
        return rendered_imgs

    def multiview_data(
        self,
        frames_dir: Path,
        smplx_kwargs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        video_struct: Tuple[int, int],
        raw_img: np.ndarray,
    ):
        """Create the multiview frames for the different methods"""
        for frame_idx, angle in enumerate(range(0, 365, 45)):

            rendered_imgs: Dict[str, np.ndarray] = self.get_rendered_images(smplx_kwargs, angle)

            # resize images to the same size  as the raw image
            for method, img in rendered_imgs.items():
                rendered_imgs[method] = cv2.resize(img, (raw_img.shape[1], raw_img.shape[0]))

            # add description to the image of its type (gt, shapy, pixie, spin, our)
            for method, img in rendered_imgs.items():
                cv2.putText(
                    img,
                    method,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            num_rows, num_cols = video_struct

            # if there are less methods than the video structure allows, add empty image
            # +1 because we have also the raw image
            if len(rendered_imgs) + 1 < num_rows * num_cols:
                for _ in range(num_rows * num_cols - len(rendered_imgs)):
                    rendered_imgs["empty"] = np.zeros_like(rendered_imgs["ours"])

            row_imgs = []
            root_imgs = [raw_img]
            cols_counter = 0
            offset = len(root_imgs)
            for row_idx in range(num_rows):
                if row_idx == 0:
                    row_imgs.append(cv2.hconcat(root_imgs + list(rendered_imgs.values())[: num_cols - offset]))
                else:
                    row_imgs.append(
                        cv2.hconcat(list(rendered_imgs.values())[num_cols - offset : num_cols + cols_counter])
                    )
                cols_counter += num_cols
            final_img = cv2.vconcat(row_imgs)

            final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frames_dir / f"{frame_idx}.png"), final_img)

    def get_video_structure(self, num_methods: int) -> Tuple[int, int]:
        """Get the video structure for the multiview video, based on the number of methods to compare"""
        suggested_video_struct, num_imgs = self.utils.get_plot_shape(num_methods)
        if num_imgs < num_methods:
            suggested_video_struct = list(suggested_video_struct)
            while num_imgs < num_methods:
                suggested_video_struct[0] += 1
                num_imgs += 1
        video_struct = tuple(suggested_video_struct)
        if video_struct[0] > video_struct[1]:
            video_struct = (video_struct[1], video_struct[0])
        return video_struct

    def __call__(self):

        for raw_img_path in self.raw_imgs_dir.iterdir():

            person_id_name = raw_img_path.stem

            self.logger.info(f"Processing {raw_img_path.name}...\n")

            output_path = self.output_path / person_id_name
            output_path.mkdir(exist_ok=True, parents=True)

            if (output_path / "out_vid.mp4").exists():
                self.logger.info(f"Video already exists, skipping...")
                continue

            gender = raw_img_path.stem.split("_")[-1]
            if not gender in ["neutral", "female", "male"]:
                gender = "neutral"

            body_shapes, raw_img = self.get_body_shapes(raw_img_path, gender)

            smplx_args: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = self.get_smplx_kwargs(
                body_shapes, gender
            )

            meshes: Dict[str, Meshes] = self.get_meshes_from_shapes(smplx_args)

            smplx_kwargs: Dict[str, Dict[str, np.ndarray]] = self.mesh_attributes_to_kwargs(smplx_args, to_tensor=True)

            frames_dir = output_path / "frames"
            frames_dir.mkdir(exist_ok=True)

            num_methods = len(meshes)
            num_blocks = num_methods + 1  # +1 because we have also the raw image
            video_struct = self.get_video_structure(num_blocks)
            video_shape = (self.renderer.height * video_struct[0], self.renderer.width * video_struct[1])

            # create video from multiview data
            if raw_img.shape[:2] != (self.renderer.height, self.renderer.width):
                raw_img = cv2.resize(raw_img, (self.renderer.width, self.renderer.height))

            self.multiview_data(frames_dir, smplx_kwargs, video_struct, raw_img)
            self.create_video_from_dir(frames_dir, video_shape)


@hydra.main(config_path="../config", config_name="celebrities_comparison")
def main(cfg: DictConfig) -> None:
    hbw_comparison = CelebComparison(**cfg.comparison_kwargs)
    hbw_comparison()


if __name__ == "__main__":
    main()
