import cv2
import clip
import hydra
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from typing import Tuple, Literal, List
from pytorch_lightning import seed_everything
from torchvision.transforms import Resize, Compose, RandomResizedCrop, Normalize
from clip2mesh.utils import ModelsFactory, Pytorch3dRenderer, Utils

seed_everything(42)


class Model(nn.Module):
    def __init__(self, params_size: Tuple[int, int] = (1, 10)):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(params_size))

    def forward(self):
        return self.weights


class CLIPLoss(nn.Module):
    def __init__(self, inverse: bool = False):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.inverse = inverse

    def forward(self, image, text):
        if self.inverse:
            similarity = self.model(image, text)[0] / 100
        else:
            similarity = 1 - self.model(image, text)[0] / 100
        return similarity

    # def forward(self, image, text):
    #     gt_similarity = torch.tensor([0.15]).to(image.device) if self.inverse else torch.tensor([0.3]).to(image.device)
    #     encoded_image = self.model.encode_image(image)
    #     encoded_text = self.model.encode_text(text)
    #     if self.inverse:
    #         similarity = torch.cosine_similarity(encoded_image, encoded_text, axis=-1)
    #     else:
    #         similarity = -torch.cosine_similarity(encoded_image, encoded_text, axis=-1)
    #     return self.mse(similarity.float(), gt_similarity)


class Optimization:
    def __init__(
        self,
        model_type: str,
        optimize_features: str,
        descriptors: List[str],
        renderer_kwargs: DictConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        total_steps: int = 1000,
        lr: float = 0.001,
        output_dir: str = "./",
        fps: int = 10,
        gender: Literal["male", "female", "neutral"] = "neutral",
        display: bool = False,
        num_coeffs: int = 10,
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
        self.descriptors = descriptors
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.display = display
        self.num_coeffs = num_coeffs
        self.img_out_size = renderer_kwargs["img_size"]
        self.resize = Resize((224, 224))
        self.view_angles = range(45, 46)
        try:
            self.num_rows, self.num_cols = self.get_collage_shape()
        except TypeError:
            self.num_rows, self.num_cols = 1, 1

        self._load_logger()

    def record_video(self, fps, output_dir, text) -> cv2.VideoWriter:
        video_recorder = cv2.VideoWriter(
            f"{output_dir}/{text.replace(' ', '_')}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(self.img_out_size)
        )
        return video_recorder

    def _load_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
        self.logger = logging.getLogger(__name__)

    def render_image(self, parameters: torch.Tensor, angle: float) -> torch.Tensor:
        model_kwargs = {self.optimize_features: parameters, "device": self.device, "num_coeffs": self.num_coeffs}
        verts, faces, vt, ft = self.models_factory.get_model(gender=self.gender, **model_kwargs)
        if self.model_type == "smplx":
            verts += self.utils.smplx_offset_tensor.to(verts.device)
        rotate_meah_kwargs = {"degrees": float(angle), "axis": "y"}
        image_for_clip = self.clip_renderer.render_mesh(
            verts=verts, faces=faces[None], vt=vt, ft=ft, rotate_mesh=rotate_meah_kwargs
        )
        return image_for_clip

    def loss(
        self, parameters, loss_fn: CLIPLoss, text: torch.Tensor, angle: float = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rendered_img = self.render_image(parameters, angle=angle)
        clip_rendered_image = self.resize(rendered_img[..., :3].permute(0, 3, 1, 2))
        loss = loss_fn(clip_rendered_image, text)
        return loss, rendered_img

    def get_collage_shape(self):
        num_rows, num_cols = self.utils.get_plot_shape(len(self.view_angles))[0]
        if num_rows > num_cols:
            return num_cols, num_rows
        return num_rows, num_cols

    def get_collage(self, images_list: List[np.ndarray], losses: List[float]) -> np.ndarray:
        imgs_collage = [
            cv2.cvtColor(
                rend_img.detach().cpu().numpy()[0],
                cv2.COLOR_RGB2BGR,
            )
            for rend_img in images_list
        ]
        imgs_collage = [
            cv2.putText(img, f"loss: {loss:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            for img, loss in zip(imgs_collage, losses)
        ]
        collage = np.concatenate(
            [
                np.concatenate(imgs_collage[i * self.num_cols : (i + 1) * self.num_cols], axis=1)
                for i in range(self.num_rows)
            ],
            axis=0,
        )
        return collage

    def optimize(self):

        for word_desciptor in self.descriptors:

            self.logger.info(f"Optimizing for {word_desciptor}...")
            output_dir = self.output_dir / word_desciptor
            output_dir.mkdir(parents=True, exist_ok=True)

            encoded_text = clip.tokenize([word_desciptor]).to(self.device)

            for phase in ["regular", "inverse"]:

                file_name = f"{word_desciptor}_{phase}.npy" if phase == "inverse" else f"{word_desciptor}.npy"
                out_file_path = output_dir / file_name
                if out_file_path.exists():
                    self.logger.info(f"File {out_file_path} already exists. Skipping...")
                    continue

                self.logger.info(f"Phase: {phase}")
                inverse = True if phase == "inverse" else False
                loss_fn = CLIPLoss(inverse=inverse)
                video_recorder = self.record_video(self.fps, output_dir, f"{word_desciptor}_{phase}")
                model = Model().to(self.device)
                self.logger.info(f"learning rate: {self.lr}")
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                total_losses = []
                pbar = tqdm(range(self.total_steps))
                for _ in pbar:
                    optimizer.zero_grad()
                    parameters = model()
                    loss = 0
                    rend_imgs = []
                    losses = []
                    for angle in self.view_angles:
                        temp_loss, temp_rendered_img = self.loss(
                            parameters, loss_fn=loss_fn, text=encoded_text, angle=angle
                        )
                        loss += temp_loss
                        losses.append(temp_loss.item())
                        rend_imgs.append(temp_rendered_img)
                    total_losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(f"Loss: {loss.item():.4f}")
                    collage = self.get_collage(rend_imgs, losses)
                    collage = cv2.resize(collage.copy(), (896, 896))
                    if self.display:
                        cv2.imshow("image", collage)
                        cv2.waitKey(1)
                    img_for_vid = np.clip((collage * 255), 0, 255).astype(np.uint8)
                    video_recorder.write(img_for_vid)
                plt.plot(total_losses)
                plt.savefig(output_dir / f"{word_desciptor}_{phase}.png")
                plt.close()
                video_recorder.release()
                cv2.destroyAllWindows()
                model_weights = model().detach().cpu().numpy()
                np.save(out_file_path, model_weights)


@hydra.main(config_path="../config", config_name="clip_optimization")
def main(cfg: DictConfig):
    optimization = Optimization(**cfg)
    optimization.optimize()


if __name__ == "__main__":
    main()
