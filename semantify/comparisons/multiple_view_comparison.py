import cv2
import hydra
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, Union, Any, Tuple
from clip2mesh.utils import Pytorch3dRenderer, Utils


class MultiViewCompare:
    def __init__(self, args):
        self.results_dir = Path(args.results_dir)
        self.utils = Utils(comparison_mode=True)

        self._load_renderer(args.renderer_kwargs)
        self._load_gender_dict()

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def get_kwargs_to_renderer(
        self, verts: np.ndarray, faces: np.ndarray, vt: np.ndarray = None, ft: np.ndarray = None
    ) -> Dict[str, torch.Tensor]:
        return {
            "verts": torch.tensor(verts + self.utils.smplx_offset_numpy.astype(np.float32)).unsqueeze(0),
            "faces": torch.tensor(faces).unsqueeze(0),
            "vt": torch.tensor(vt).unsqueeze(0) if vt is not None else None,
            "ft": torch.tensor(ft).unsqueeze(0) if ft is not None else None,
        }

    def _load_renderer(self, kwargs: Union[DictConfig, Dict[str, Any]]):
        self.renderer = Pytorch3dRenderer(**kwargs)

    def _load_gender_dict(self):
        self.gender_dict = {
            "img_00": "female",
            "img_01": "female",
            "img_02": "female",
            "img_03": "female",
            "img_04": "male",
            "img_05": "female",
            "img_06": "male",
            "img_07": "female",
            "img_08": "male",
            "img_09": "female",
            "img_10": "male",
            "img_11": "male",
            "img_12": "female",
            "img_13": "female",
            "img_14": "male",
            "img_15": "female",
            "img_16": "male",
            "img_17": "female",
            "img_18": "female",
            "img_19": "male",
            "img_20": "male",
            "img_21": "male",
            "img_22": "male",
        }

    def create_video_from_frames(self, person_dir: Path, frame_shape: Tuple[int, int]):
        frames_paths = sorted(list(person_dir.rglob("*.png")), key=lambda x: int(x.stem))
        video_path = person_dir.parent / f"out_vid.mp4"
        video = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            frame_shape,
        )
        for frame_path in tqdm(frames_paths, desc="Creating video", total=len(frames_paths)):
            frame = cv2.imread(str(frame_path))
            video.write(frame)
        video.release()

    def __call__(self):
        for person_dir in self.results_dir.iterdir():

            print(f"Processing {person_dir.name}...")

            orig_image = cv2.imread(str(person_dir / "orig.png"))
            pred_betas = torch.tensor(np.load(person_dir / f"{person_dir.name}_pred_shape_tensor.npy"))
            shapy_betas = torch.tensor(np.load(person_dir / f"{person_dir.name}_shapy_shape_tensor.npy"))

            gender = self.gender_dict[person_dir.name]

            pred_features = self.utils.get_smplx_model(betas=pred_betas, gender=gender)
            shapy_kwargs = self.utils.get_smplx_model(betas=shapy_betas, gender=gender)

            pred_kwargs = self.get_kwargs_to_renderer(*pred_features)
            shapy_kwargs = self.get_kwargs_to_renderer(*shapy_kwargs)

            frames_dir = person_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

            for frame_idx, angle in enumerate(range(0, 360, 5)):

                pred_kwargs.update({"rotate_mesh": {"degrees": float(angle), "axis": "y"}})
                shapy_kwargs.update({"rotate_mesh": {"degrees": float(angle), "axis": "y"}})

                pred_img = self.renderer.render_mesh(**pred_kwargs)
                shapy_img = self.renderer.render_mesh(**shapy_kwargs)

                pred_img = self.adjust_rendered_img(pred_img)
                shapy_img = self.adjust_rendered_img(shapy_img)

                pred_img = cv2.resize(pred_img, (orig_image.shape[1], orig_image.shape[0]))
                shapy_img = cv2.resize(shapy_img, (orig_image.shape[1], orig_image.shape[0]))

                cv2.putText(
                    orig_image,
                    f"orig",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    pred_img,
                    f"ours",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    shapy_img,
                    f"shapy",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                concatenated_img = np.concatenate([orig_image, pred_img, shapy_img], axis=1)
                cv2.imshow("comparison", concatenated_img)
                cv2.waitKey(1)
                cv2.imwrite(str(frames_dir / f"{frame_idx}.png"), concatenated_img)

            cv2.destroyAllWindows()
            self.create_video_from_frames(frames_dir, concatenated_img.shape[:2][::-1])


@hydra.main(config_path="../config", config_name="multi_view_comparison")
def main(args: DictConfig):
    MultiViewCompare(args)()


if __name__ == "__main__":
    main()
