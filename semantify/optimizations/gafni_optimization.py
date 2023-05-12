import cv2
import clip
import hydra
import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from typing import Tuple, Literal, List
from torchvision.transforms import Resize
from clip2mesh.utils import ModelsFactory, Pytorch3dRenderer, Utils


class Model(nn.Module):
    def __init__(self, params_size: Tuple[int, int] = (1, 10)):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(params_size))

    def forward(self):
        return self.weights


class CLIPLoss(nn.Module):
    def __init__(self, inverse: bool = False):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.inverse = inverse

    def forward(self, image, text):
        similarity = -self.model(image, text)[0]
        return similarity


class GafniOptimization:
    def __init__(
        self,
        model_type: str,
        optimize_features: str,
        text: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        total_steps: int = 1000,
        lr: float = 0.1,
        output_dir: str = "./",
        fps: int = 10,
        gender: Literal["male", "female", "neutral"] = "neutral",
        img_out_size: Tuple[int, int] = (512, 512),
        display: bool = False,
        num_coeffs: int = 10,
        renderer_kwargs: DictConfig = None,
        loss_type: Literal["clip", "mse"] = "clip",
    ):
        super().__init__()
        self.total_steps = total_steps
        self.device = device
        self.gender = gender
        self.model_type = model_type
        self.optimize_features = optimize_features
        self.models_factory = ModelsFactory(model_type)
        self.clip_model, self.image_encoder = clip.load("ViT-B/32", device=device)
        self.model = Model()
        self.lr = lr
        self.utils = Utils()
        self.clip_renderer = Pytorch3dRenderer(**renderer_kwargs)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.display = display
        self.num_coeffs = num_coeffs
        self.img_out_size = img_out_size
        self.view_angles = range(0, 360, 45)
        self.resize = Resize((224, 224))
        self.loss_type = loss_type
        self.num_rows, self.num_cols = self.get_collage_shape()

        self._load_logger()
        self._get_default_model_img()
        self._encode_labels(text)

    def record_video(self, fps, output_dir, text) -> cv2.VideoWriter:
        collage_size = (self.img_out_size[0] * self.num_cols, self.img_out_size[1] * self.num_rows)
        video_recorder = cv2.VideoWriter(
            f"{output_dir}/{text.replace(' ', '_')}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, collage_size
        )
        return video_recorder

    def _encode_labels(self, text: List[str]):
        self.encoded_labels = clip.tokenize(text).to(self.device)

    def get_first_scores(self, image: np.ndarray):
        with torch.no_grad():
            scores = self.clip_model(image, self.encoded_labels)[0]
        return scores

    def _load_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _get_default_model_img(self):
        verts, faces, vt, ft = self.models_factory.get_model(device=self.device, gender=self.gender)
        if self.model_type == "smplx":
            verts += self.utils.smplx_offset_tensor.to(self.device)
        self.default_model_img = self.clip_renderer.render_mesh(verts=verts, faces=faces[None], vt=vt, ft=ft)

    def render_image(self, parameters: torch.Tensor, angle: float) -> torch.Tensor:
        model_kwargs = {
            self.optimize_features: parameters,
            "device": self.device,
            "num_coeffs": self.num_coeffs,
            "gender": self.gender,
        }
        verts, faces, vt, ft = self.models_factory.get_model(**model_kwargs)
        if self.model_type == "smplx":
            verts += self.utils.smplx_offset_tensor.to(verts.device)
        rotate_meah_kwargs = {"degrees": float(angle), "axis": "y"}
        image_for_clip = self.clip_renderer.render_mesh(
            verts=verts, faces=faces[None], vt=vt, ft=ft, rotate_mesh=rotate_meah_kwargs
        )

        return torch.cat([image_for_clip[0], self.default_model_img[0]], 1).unsqueeze(0)

    def get_collage_shape(self):
        num_rows, num_cols = self.utils.get_plot_shape(len(self.view_angles))[0]
        if num_rows > num_cols:
            return num_cols, num_rows
        return num_rows, num_cols

    def get_collage(self, images_list: List[np.ndarray]) -> np.ndarray:
        imgs_collage = [
            cv2.cvtColor(
                cv2.resize(rend_img.detach().cpu().numpy()[0], self.img_out_size),
                cv2.COLOR_RGB2BGR,
            )
            for rend_img in images_list
        ]
        collage = np.concatenate(
            [
                np.concatenate(imgs_collage[i * self.num_cols : (i + 1) * self.num_cols], axis=1)
                for i in range(self.num_rows)
            ],
            axis=0,
        )
        return collage

    def optimize(self, idx_to_modify: int):

        rendered_img = self.clip_renderer.render_mesh(
            *self.models_factory.get_model(gender=self.gender, device=self.device)
        )
        clip_rendered_image = self.resize(rendered_img[..., :3].permute(0, 3, 1, 2))

        gt_scores = self.get_first_scores(clip_rendered_image)
        gt_scores[0, idx_to_modify] = 100.0

        loss_fn = nn.MSELoss() if self.loss_type == "mse" else CLIPLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        progress_bar = tqdm(range(self.total_steps), desc="loss: 0.0", total=self.total_steps)
        for _ in progress_bar:

            optimizer.zero_grad()
            parameters = self.model()
            img = self.render_image(parameters, angle=0.0)
            img_to_clip = self.resize(img[..., :3].permute(0, 3, 1, 2))
            img_to_video = img[0, ..., :3].detach().cpu().numpy()
            if self.loss_type == "mse":
                scores = self.clip_model(img_to_clip, self.encoded_labels)[0]
                loss = loss_fn(scores, gt_scores)
                loss += loss_fn(scores[0, idx_to_modify], gt_scores[0, idx_to_modify]) * 5
            else:
                loss = loss_fn(img_to_clip, self.encoded_labels)
            loss.backward()
            optimizer.step()

            self.logger.info(f"parameters: {parameters}")
            progress_bar.set_description(f"loss: {loss.item():.4f}")

            img_to_video = cv2.resize(img_to_video, (896, 896))
            cv2.putText(
                img_to_video,
                f"loss: {loss.item():.4f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            img_to_video = cv2.cvtColor(img_to_video, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", img_to_video)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        return parameters


@hydra.main(config_path="../config", config_name="gafni_optimization")
def main(cfg: DictConfig):
    optimization = GafniOptimization(**cfg)
    optimization.optimize(idx_to_modify=0)


if __name__ == "__main__":
    main()
