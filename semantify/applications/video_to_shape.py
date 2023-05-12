import cv2
import clip
import h5py
import torch
import hydra
import numpy as np
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from copy import deepcopy
from omegaconf import DictConfig
from pytorch3d.io import save_obj
from typing import Union, Literal, Dict, Tuple, Any
from clip2mesh.utils import Utils, Pytorch3dRenderer
from clip2mesh.applications.image_to_shape import Image2Shape


class Video2Shape(Image2Shape):
    def __init__(self, args):
        super().__init__(args)
        self.data_dir = Path(args.data_dir)
        self.utils = Utils(comparison_mode=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = args.model_type
        self.display = args.display
        self.create_vid = args.create_vid
        self.with_face = args.with_face
        self.renderer_kwargs = args.renderer_kwargs
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self._load_smplx_models(**args.smplx_models_paths)
        self._load_weights(args.labels_weights)
        self._encode_labels()

    def _from_h5_to_img(
        self, h5_file_path: Union[str, Path], gender: Literal["male", "female", "neutral"], renderer: Pytorch3dRenderer
    ) -> np.ndarray:
        data = h5py.File(h5_file_path, "r")
        shape_vector = torch.tensor(data["betas"])[None].float()
        render_mesh_kwargs = self.get_render_mesh_kwargs(shape_vector, gender=gender)
        rendered_img = renderer.render_mesh(**render_mesh_kwargs)
        rendered_img = self.adjust_rendered_img(rendered_img)
        return rendered_img, shape_vector

    # def _load_renderer(self, kwargs: Union[DictConfig, Dict[str, Any]]):
    #     return Pytorch3dRenderer(**kwargs)

    # def _load_weights(self, labels_weights: Dict[str, float]):
    #     self.labels_weights = {}
    #     for gender, weights in labels_weights.items():
    #         self.labels_weights[gender] = (
    #             torch.tensor(weights).to(self.device)
    #             if weights is not None
    #             else torch.ones(len(self.labels[gender])).to(self.device)
    #         )

    # def _get_smplx_attributes(
    #     self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"]
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     betas = pred_vec.cpu()
    #     if hasattr(self, "rest_pose"):
    #         body_pose = self.rest_pose
    #     else:
    #         body_pose = None
    #     smplx_out = self.utils.get_smplx_model(betas=betas, gender=gender, body_pose=body_pose)
    #     return smplx_out

    # @staticmethod
    # def adjust_rendered_img(img: torch.Tensor):
    #     img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
    #     return img

    # def get_render_mesh_kwargs(
    #     self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"]
    # ) -> Dict[str, np.ndarray]:
    #     out = self._get_smplx_attributes(pred_vec=pred_vec, gender=gender)

    #     kwargs = {"verts": out[0], "faces": out[1], "vt": out[2], "ft": out[3]}

    #     return kwargs

    # @staticmethod
    # def _flatten_list_of_lists(list_of_lists):
    #     return [item for sublist in list_of_lists for item in sublist]

    # def _load_smplx_models(self, smplx_male: str, smplx_female: str) -> Tuple[nn.Module, nn.Module]:
    #     smplx_female, labels_female = self.utils.get_model_to_eval(smplx_female)
    #     smplx_male, labels_male = self.utils.get_model_to_eval(smplx_male)
    #     labels_female = self._flatten_list_of_lists(labels_female)
    #     labels_male = self._flatten_list_of_lists(labels_male)
    #     self.model = {"male": smplx_male, "female": smplx_female}
    #     self.labels = {"male": labels_male, "female": labels_female}

    # def _encode_labels(self):
    #     self.encoded_labels = {
    #         gender: clip.tokenize(self.labels[gender]).to(self.device) for gender in self.labels.keys()
    #     }

    # def normalize_scores(self, scores: torch.Tensor, gender: Literal["male", "female", "neutral"]) -> torch.Tensor:
    #     normalized_score = scores * self.labels_weights[gender]
    #     return normalized_score.float()

    def create_video_from_dir(self, dir_path: Union[str, Path], image_shape: Tuple[int, int]):
        dir_path = Path(dir_path)
        out_vid_path = dir_path.parent / "out_vid.mp4"
        out_vid = cv2.VideoWriter(out_vid_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), 30, image_shape)
        sorted_frames = sorted(dir_path.iterdir(), key=lambda x: int(x.stem))
        for frame in tqdm(sorted_frames, desc="Creating video", total=len(sorted_frames)):
            out_vid.write(cv2.imread(frame.as_posix()))
        out_vid.release()

    def __call__(self):
        for folder in self.data_dir.iterdir():

            if folder.is_dir():

                print(f"Processing {folder.name}...")

                dir_output_path = self.output_path / folder.name
                dir_output_path.mkdir(parents=True, exist_ok=True)

                images_output_path = dir_output_path / "images"
                images_output_path.mkdir(parents=True, exist_ok=True)

                if (images_output_path.parent / "out_vid.mp4").exists():
                    print(f"Video already exists for {folder.name}...")
                    continue
                gender = folder.name.split("-")[0]
                print(f"The gender is {gender}")

                renderer_kwargs = deepcopy(self.renderer_kwargs)
                renderer_kwargs.update({"tex_path": (folder / f"tex-{folder.name}.jpg").as_posix()})
                renderer = self._load_renderer(renderer_kwargs)

                gt_mesh_image, gt_shape_tensor = self._from_h5_to_img(
                    folder / "reconstructed_poses.hdf5", gender=gender, renderer=renderer
                )
                video_path = folder / (folder.name + ".mp4")
                video = cv2.VideoCapture(str(video_path))
                frame_counter = 0

                while video.isOpened():
                    ret, frame = video.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        encoded_image = self.clip_preprocess(frame).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            clip_scores = self.clip_model(encoded_image, self.encoded_labels[gender])[0]
                            clip_scores = self.normalize_scores(clip_scores, gender)
                            pred_shape_tensor = self.model[gender](clip_scores)

                        # calculate the l2 loss between the predicted and the ground truth shape
                        l2_loss = F.mse_loss(pred_shape_tensor.cpu(), gt_shape_tensor)

                        pred_mesh_kwargs = self.get_render_mesh_kwargs(pred_shape_tensor, gender=gender)
                        pred_mesh_img = renderer.render_mesh(**pred_mesh_kwargs)
                        pred_mesh_img = self.adjust_rendered_img(pred_mesh_img)

                        resized_frame = cv2.resize(np.array(frame), pred_mesh_img.shape[:2][::-1])

                        cv2.putText(resized_frame, "orig", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(
                            resized_frame,
                            f"loss: {l2_loss.item():.4f}",
                            (0, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
                        cv2.putText(pred_mesh_img, "pred", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(gt_mesh_image, "gt", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        concatenated_img = np.concatenate((resized_frame, pred_mesh_img, gt_mesh_image), axis=1)
                        concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_RGB2BGR)

                        cv2.imwrite(str(images_output_path / f"{frame_counter}.png"), concatenated_img)

                        if self.display:
                            cv2.putText(
                                concatenated_img,
                                f"frame: {frame_counter}",
                                (40, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )
                            cv2.imshow("frame", concatenated_img)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

                        frame_counter += 1

                    else:
                        break

                video.release()
                cv2.destroyAllWindows()

                if self.create_vid:
                    self.create_video_from_dir(images_output_path, concatenated_img.shape[:2][::-1])


@hydra.main(config_path="../config", config_name="video_to_shape")
def main(args: DictConfig):
    video2shape = Video2Shape(args)
    video2shape()


if __name__ == "__main__":
    main()
