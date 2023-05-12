import cv2
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from typing import Dict, Tuple, Literal, List
from clip2mesh.utils import Image2ShapeUtils, Utils


class ComparisonUtils(Image2ShapeUtils):
    def __init__(
        self,
        raw_imgs_dir: str,
        comparison_dirs: Dict[str, str],
        gt_dir: str,
        output_path: str,
        renderer_kwargs: Dict[str, Dict[str, float]],
        smplx_models_paths: Dict[str, str],
    ):
        super().__init__()
        self.raw_imgs_dir: Path = Path(raw_imgs_dir)
        self.comparison_dirs: Dict[str, str] = comparison_dirs
        self.gt_dir: Path = Path(gt_dir)
        self.output_path: Path = Path(output_path)
        self.utils: Utils = Utils(comparison_mode=True)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_renderer(renderer_kwargs)
        self._load_smplx_models(smplx_models_paths)
        self._load_clip_model()
        self._encode_labels()
        self._perpare_comparisons_dir()
        self._load_results_df()
        self._load_logger()

    def _load_results_df(self):
        self.results_df = pd.DataFrame(columns=["image_name", "loss", "shapy", "pixie", "spin", "ours"])

    def _load_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_shapy_data(npz_path: Path) -> Dict[str, torch.Tensor]:
        """Load shapy predictions from pre-processed npz file"""
        relevant_keys = ["body_pose", "betas"]
        data = np.load(npz_path, allow_pickle=True)
        return {k: torch.from_numpy(v) for k, v in data.items() if k in relevant_keys}

    @staticmethod
    def get_pixie_data(pkl_path: Path) -> torch.Tensor:
        """Load pixie predictions from pre-processed pkl file"""
        data = np.load(pkl_path, allow_pickle=True)
        return torch.tensor(data["shape"][:10])[None]

    @staticmethod
    def get_spin_data(npy_path: Path) -> Dict[str, torch.Tensor]:
        """Load spin predictions from pre-processed npy file"""
        data = np.load(npy_path, allow_pickle=True)
        return torch.tensor(data)[None]

    def _perpare_comparisons_dir(self):
        """Create a directory for the comparison results"""
        self.comparison_dirs = {k: Path(v) for k, v in self.comparison_dirs.items()}

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

    def calc_l2_distances(self, body_shapes: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate the distance between the gt and the other methods"""
        losses = {}
        for k, v in body_shapes.items():
            if k == "gt":
                continue
            losses[k] = torch.linalg.norm(body_shapes["gt"] - v, dim=1).item()
        return losses

    def calc_chamfer_distance(self, meshes: Dict[str, Meshes]) -> Dict[str, float]:
        """Calculate the chamfer distance between the gt and the other methods"""
        losses = {}
        for method, mesh in meshes.items():
            if method == "gt":
                continue
            losses[method] = chamfer_distance(meshes["gt"].verts_packed()[None], mesh.verts_packed()[None])[0].item()
        return losses

    def get_smplx_kwargs(
        self, body_shapes: Dict[str, torch.Tensor], gender: Literal["male", "female", "neutral"], get_smpl: bool = False
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the smplx kwargs for the different methods -> (vertices, faces, vt, ft)"""
        smplx_kwargs = {}
        for method, body_shape in body_shapes.items():
            # get_smpl = True if method in ["spin"] else False
            fixed_gender = gender if method in "gt" else "neutral"
            smplx_kwargs[method] = self._get_smplx_attributes(body_shape, fixed_gender, get_smpl=get_smpl)
        return smplx_kwargs

    def get_meshes_from_shapes(
        self, smplx_kwargs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> Dict[str, Meshes]:
        """Get the meshes from the smplx kwargs"""
        meshes = {}
        for method, args in smplx_kwargs.items():
            meshes[method] = self.renderer.get_mesh(*args)
        return meshes

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
        for frame_idx, angle in enumerate(range(0, 365, 5)):

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

    @staticmethod
    def create_vid_with_history_plot(
        distances_dict: Dict[str, torch.Tensor], data_dir_path: Path, vid_shape: Tuple[int, int]
    ):
        """
        Create a video with a plot of the history of the loss

        data_dir_path: Path to the directory containing the frames directories, e.g.:
            |
            frames_dir
                |
                frames
                    |
                    0.png
        """
        sorted_dir = sorted(data_dir_path.iterdir(), key=lambda x: int(x.stem))
        out_vid = cv2.VideoWriter(
            str(data_dir_path.parent / f"{data_dir_path.name}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            15,
            (vid_shape[0] * 2, vid_shape[1]),
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(history_len):
            img_path = sorted_dir[i] / "frames" / "0.png"
            temp_plot = sorted_dir[i].parent / "temp_plot.png"
            img = cv2.imread(str(img_path))
            if i == 0:
                reference = {method: body_shape[i] for method, body_shape in history.items()}
            else:
                # plot history is the norm of the difference between the current and the reference
                if history_plot is None:
                    history_plot = {k: [torch.norm(v[i] - reference[k], dim=1).item()] for k, v in history.items()}
                else:
                    # extend the history plot
                    for k, v in history.items():
                        history_plot[k].extend([torch.norm(v[i - 1] - reference[k], dim=1).item()])
                # plot history
                for key in history_plot:
                    ax.plot(history_plot[key], label=key)
                ax.legend(loc="upper right")
                ax.set_yticks([])
                fig.savefig(temp_plot)
                ax.clear()
                history_plot_img = cv2.imread(str(temp_plot))
                history_plot_img = cv2.resize(history_plot_img, img.shape[:2][::-1])

                # write the plot on the image to the video
                img = np.concatenate([img, history_plot_img], axis=1)
                out_vid.write(img)
                temp_plot.unlink()
        out_vid.release()
