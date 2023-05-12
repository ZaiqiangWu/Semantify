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
from semantify.utils import Image2ShapeUtils, Utils


class HBWComparison(Image2ShapeUtils):
    def __init__(
        self,
        smplx_models: Dict[str, str],
        raw_imgs_dir: str,
        gt_dir: str,
        output_path: str,
        comparison_dirs: Dict[str, str],
        renderer_kwargs: Dict[str, Dict[str, float]],
    ):
        super().__init__()
        self.raw_imgs_dir: Path = Path(raw_imgs_dir)
        self.comparison_dirs: Dict[str, str] = comparison_dirs
        self.gt_dir: Path = Path(gt_dir)
        self.output_path: Path = Path(output_path)
        self.utils: Utils = Utils(comparison_mode=True)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_renderer(renderer_kwargs)
        self.load_smplx_model(smplx_models)
        self._load_clip_model()
        self._perpare_comparisons_dir()
        self._load_results_df()
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

    def _load_results_df(self):
        self.results_df = pd.DataFrame(columns=["image_name", "loss", "shapy", "ours", "ours_gender"])

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

    def calc_l2_distances(
        self, methods_features: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], gt: torch.Tensor
    ) -> Dict[str, float]:
        distances = {}
        for method, mesh_features in methods_features.items():
            if method == "gt":
                continue
            distances[method] = F.mse_loss(torch.tensor(mesh_features[0]), gt).item()
        return distances

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
        person_id = raw_img_path.parent.name

        # load raw image
        raw_img = cv2.imread(str(raw_img_path))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # pixie prediction
        pixie_pkl_path = self.comparison_dirs["pixie"] / person_id / img_id / f"{img_id}_param.pkl"
        pixie_body_shape = self.get_pixie_data(pixie_pkl_path)

        # shapy prediction
        shpay_npz_path = self.comparison_dirs["shapy"] / person_id / f"{img_id}.npz"
        shapy_data = self.get_shapy_data(shpay_npz_path)
        shapy_body_shape = torch.tensor(shapy_data["betas"])[None]  # TODO: maybe convert to tensor or add dimension

        # clip preprocess
        encoded_image = self.clip_preprocess(Image.fromarray(raw_img)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # our prediction - neutral
            clip_scores = self.clip_model(encoded_image, self.smplx_models["neutral"]["labels"])[0].float()
            our_neutral_body_shape = self.smplx_models["neutral"]["model"](clip_scores).cpu()

            # out prediction - real gender
            clip_scores = self.clip_model(encoded_image, self.smplx_models[gender]["labels"])[0].float()
            our_real_gender_body_shape = self.smplx_models[gender]["model"](clip_scores).cpu()

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

    def calc_chamfer_distance(self, meshes: Dict[str, Meshes]) -> Dict[str, float]:
        """Calculate the chamfer distance between the gt and the other methods"""
        losses = {}
        for method, mesh in meshes.items():
            if method == "gt":
                continue
            losses[method] = chamfer_distance(meshes["gt"].verts_packed()[None], mesh.verts_packed()[None])[0].item()
        return losses

    def multiview_data(
        self,
        frames_dir: Path,
        smplx_kwargs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        video_struct: Tuple[int, int],
        raw_img: np.ndarray,
    ):
        """Create the multiview frames for the different methods"""
        for frame_idx, angle in enumerate(range(0, 1, 1)):

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
                empty_img = np.ones_like(rendered_imgs["gt"])
                for _ in range(num_rows * num_cols - len(rendered_imgs)):
                    rendered_imgs["empty"] = empty_img

            gt_img = rendered_imgs.pop("gt")

            row_imgs = []
            root_imgs = [raw_img, gt_img]
            cols_counter = 0
            offset = 2
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

        for person_id in self.raw_imgs_dir.iterdir():
            image_counter = 0

            # ------ DEBUG ------
            if person_id.name not in ["017_111_48", "029_66_21", "020_12_25", "022_12_34", "033_85_38", "012_13_20"]:
                continue
            # -------------------
            for raw_img_path in person_id.iterdir():

                # ------ DEBUG ------
                if raw_img_path.stem not in [
                    # "00283_male",
                    # "01777_female",
                    # "00472_male",
                    # "02455_male",
                    # "01104_female",
                    # "01767_female",
                    "00010_female"
                ]:
                    continue
                # -------------------

                person_id_name = person_id.name.split("_")[0]

                self.logger.info(f"Processing {person_id.name} | {raw_img_path.name}...\n")

                output_path = self.output_path / person_id.name / raw_img_path.stem
                output_path.mkdir(exist_ok=True, parents=True)

                # ------ DEBUG ------
                # if (output_path / "out_vid.mp4").exists():
                #     self.logger.info(f"Video already exists, skipping...")
                #     continue
                # -------------------

                gender = raw_img_path.stem.split("_")[-1]

                gt_verts, gt_mesh = self._get_gt_data(self.gt_dir / f"{person_id_name}.npy")
                gt_verts += self.utils.smplx_offset_numpy.astype(np.float32)

                body_shapes, raw_img = self.get_body_shapes(raw_img_path, gender)

                smplx_args: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = self.get_smplx_kwargs(
                    body_shapes, gender
                )

                smplx_args["gt"] = (
                    gt_verts,
                    gt_mesh.faces_packed(),
                    smplx_args["shapy"][2],
                    smplx_args["shapy"][3],
                )

                meshes: Dict[str, Meshes] = self.get_meshes_from_shapes(smplx_args)

                meshes["gt"] = gt_mesh

                chamfer_distances: Dict[str, torch.Tensor] = self.calc_chamfer_distance(meshes)
                l2_distances: Dict[str, np.ndarray] = self.calc_l2_distances(smplx_args, gt_verts)

                smplx_kwargs: Dict[str, Dict[str, np.ndarray]] = self.mesh_attributes_to_kwargs(
                    smplx_args, to_tensor=True
                )

                # ------ DEBUG ------
                import matplotlib.pyplot as plt

                shapy_diff = np.linalg.norm(smplx_args["gt"][0] - smplx_args["shapy"][0], axis=-1)
                our_diff = np.linalg.norm(smplx_args["gt"][0] - smplx_args["ours"][0], axis=-1)
                self.logger.info(f"{person_id.name} | {raw_img_path.name}")
                self.logger.info("shapy diff", np.sqrt(shapy_diff).mean())

                shapy_diff_normed = shapy_diff / 0.1094  # np.max(shapy_diff)
                our_diff_normed = our_diff / 0.1094  # np.max(shapy_diff)
                color_map = plt.get_cmap("coolwarm")
                shapy_vertex_colors = torch.tensor(color_map(shapy_diff_normed)[:, :3]).float().to(self.device)
                our_vertex_colors = torch.tensor(color_map(our_diff_normed)[:, :3]).float().to(self.device)
                shapy_diff = self.adjust_rendered_img(
                    self.renderer.render_mesh(**smplx_kwargs["shapy"], texture_color_values=shapy_vertex_colors[None])
                )
                pixie_diff = self.adjust_rendered_img(
                    self.renderer.render_mesh(**smplx_kwargs["pixie"], texture_color_values=shapy_vertex_colors[None])
                )
                our_diff = self.adjust_rendered_img(
                    self.renderer.render_mesh(**smplx_kwargs["ours"], texture_color_values=our_vertex_colors[None])
                )
                gt_img = self.adjust_rendered_img(self.renderer.render_mesh(**smplx_kwargs["gt"]))
                colored = np.concatenate([shapy_diff, pixie_diff, our_diff, gt_img], axis=1)
                colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
                outpath = "/home/nadav2/dev/data/CLIP2Shape/outs/images_to_shape/HBW_DATA/comparison_2k/test"
                cv2.imwrite(f"{outpath}/{person_id.name}_{raw_img_path.stem}.png", colored)
                # -------------------

                # frames_dir = output_path / "frames"
                # frames_dir.mkdir(exist_ok=True)

                # num_methods = len(meshes)
                # num_blocks = num_methods + 1  # +1 because we have also the raw image
                # video_struct = self.get_video_structure(num_blocks)
                # video_shape = (self.renderer.height * video_struct[0], self.renderer.width * video_struct[1])

                # # create video from multiview data
                # if raw_img.shape[:2] != (self.renderer.height, self.renderer.width):
                #     raw_img = cv2.resize(raw_img, (self.renderer.width, self.renderer.height))

                # self.multiview_data(frames_dir, smplx_kwargs, video_struct, raw_img)
                # # self.create_video_from_dir(frames_dir, video_shape)

                # # columns are: ["image_name", "loss", "shapy", "pixie", "spin", "ours"]
                # single_img_results_l2 = pd.DataFrame.from_dict(
                #     {"image_name": [raw_img_path.stem], "loss": "l2", **l2_distances}
                # )
                # # single_img_results_chamfer = pd.DataFrame.from_dict(
                # #     {"image_name": [raw_img_path.stem], "loss": "chamfer", **chamfer_distances}
                # # )
                # self.results_df = pd.concat([self.results_df, single_img_results_l2])  # , single_img_results_chamfer])

                # image_counter += 1

                # self.results_df.to_csv(self.output_path / "results.csv", index=False)


@hydra.main(config_path="../config", config_name="hbw_comparison")
def main(cfg: DictConfig) -> None:
    hbw_comparison = HBWComparison(**cfg.comparison_kwargs)
    hbw_comparison()


if __name__ == "__main__":
    main()
