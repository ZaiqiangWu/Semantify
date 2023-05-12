import cv2
import clip
import torch
import hydra
import numpy as np
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from pytorch3d.io import save_obj
from typing import Dict, Any, Tuple, List, Literal, Union
from clip2mesh.utils import Utils, Pytorch3dRenderer


class CompareToShapy:
    def __init__(
        self,
        shapy_dir: str,
        output_path: str,
        display: bool,
        smplx_models_paths: Dict[str, str],
        renderer_kwargs: Dict[str, Dict[str, float]],
    ):
        self.shapy_dir = Path(shapy_dir)
        self.output_path = Path(output_path)
        self.utils = Utils(comparison_mode=True)
        self.display = display
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self._load_gender_dict()
        self._load_smplx_models(**smplx_models_paths)
        self._encode_labels()
        self._load_renderer(renderer_kwargs)

    def _load_renderer(self, kwargs: Union[DictConfig, Dict[str, Any]]):
        self.renderer = Pytorch3dRenderer(**kwargs)

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _flatten_list_of_lists(list_of_lists: List[List[str]]) -> List[str]:
        return [item for sublist in list_of_lists for item in sublist]

    def _load_smplx_models(self, smplx_male: str, smplx_female: str) -> Tuple[nn.Module, nn.Module]:
        smplx_female, labels_female = self.utils.get_model_to_eval(smplx_female)
        smplx_male, labels_male = self.utils.get_model_to_eval(smplx_male)
        labels_female = self._flatten_list_of_lists(labels_female)
        labels_male = self._flatten_list_of_lists(labels_male)
        self.model = {"male": smplx_male, "female": smplx_female}
        self.labels = {"male": labels_male, "female": labels_female}

    @staticmethod
    def get_npz_data(npz_path: Path) -> Dict[str, torch.Tensor]:
        relevant_keys = ["body_pose", "betas", "transl", "global_rot"]
        data = np.load(npz_path, allow_pickle=True)
        return {k: torch.from_numpy(v) for k, v in data.items() if k in relevant_keys}

    @staticmethod
    def save_obj(
        obj_out_path: Union[Path, str],
        verts: np.ndarray,
        faces: np.ndarray,
        vt: np.ndarray = None,
        ft: np.ndarray = None,
    ):
        verts = torch.tensor(verts).squeeze()
        faces = torch.tensor(faces).squeeze()
        save_obj(f=obj_out_path, verts=verts, faces=faces)

    def save_shape_tensor(self, shape_tesnor: torch.Tensor, file_path: Union[str, Path]):
        file_path = Path(file_path)
        shape_tesnor = shape_tesnor.cpu().numpy()
        np.save(file_path, shape_tesnor)

    def save_images(
        self,
        orig_image: np.ndarray,
        pred_image: np.ndarray,
        shapy_image: np.ndarray,
        out_dir_path: Union[str, Path],
        loss: float,
    ):
        # save images
        out_dir_path = Path(out_dir_path)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR, orig_image)
        cv2.imwrite((out_dir_path / "orig.png").as_posix(), orig_image)
        cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR, pred_image)
        cv2.imwrite((out_dir_path / "pred.png").as_posix(), pred_image)
        cv2.cvtColor(shapy_image, cv2.COLOR_RGB2BGR, shapy_image)
        cv2.imwrite((out_dir_path / "shapy.png").as_posix(), shapy_image)

        # concat images
        orig_image = cv2.resize(orig_image, (pred_image.shape[1], pred_image.shape[0]))
        cv2.putText(orig_image, "orig", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(pred_image, "pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(shapy_image, "shapy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        concat_img = np.concatenate([orig_image, pred_image, shapy_image], axis=1)
        cv2.putText(concat_img, f"loss: {loss:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite((out_dir_path / "concat.png").as_posix(), concat_img)

        # show images
        if self.display:
            cv2.imshow("concat", concat_img)
            key = cv2.waitKey(0)
            if key == ord("q"):
                cv2.destroyAllWindows()
                exit()

    def _encode_labels(self):
        self.encoded_labels = {
            gender: clip.tokenize(self.labels[gender]).to(self.device) for gender in self.labels.keys()
        }

    def _get_smplx_kwargs(
        self,
        pred_vec: torch.Tensor,
        gender: Literal["male", "female", "neutral"],
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        betas = pred_vec.cpu()
        smplx_out = self.utils.get_smplx_model(
            betas=betas,
            gender=gender,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )
        kwargs = {
            "verts": smplx_out[0] + self.utils.smplx_offset_numpy,
            "faces": smplx_out[1],
            "vt": smplx_out[2],
            "ft": smplx_out[3],
        }
        return kwargs

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

    def process_data_unit(self, img_path: Path):
        # create output dir
        img_id = img_path.stem.split("_hd_imgs")[0]
        output_path = self.output_path / img_id
        output_path.mkdir(parents=True, exist_ok=True)

        # load image
        img = cv2.imread(img_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # prepare image for CLIP
        img = Image.fromarray(img)
        encoded_image = self.clip_preprocess(img).unsqueeze(0).to(self.device)

        # get gender
        if hasattr(self, "gender_dict"):
            gender = self.gender_dict[img_id]
        else:
            gender = img_path.stem.split("-")[0]

        # get shapy data
        shapy_data = self.get_npz_data(self.shapy_dir / f"{img_id}.npz")
        body_pose = shapy_data["body_pose"][None]
        global_orient = shapy_data["global_rot"]
        global_orient = None
        # transl = shapy_data["transl"][None]
        transl = None
        shapy_shape_tensor = shapy_data["betas"][None]

        # get pred data
        with torch.no_grad():
            clip_scores = self.clip_model(encoded_image, self.encoded_labels[gender])[0]
            pred_shape_tensor = self.model[gender](clip_scores.float())

        # calculate distance between shapy and pred
        l2_loss = F.mse_loss(pred_shape_tensor.cpu(), shapy_shape_tensor)

        # get shapy rendered image
        shapy_mesh_kwargs = self._get_smplx_kwargs(
            shapy_shape_tensor,
            gender="neutral",
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )
        shapy_mesh_img = self.renderer.render_mesh(**shapy_mesh_kwargs)
        shapy_mesh_img = self.adjust_rendered_img(shapy_mesh_img)

        # get pred rendered image
        pred_mesh_kwargs = self._get_smplx_kwargs(
            pred_shape_tensor,
            gender=gender,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )
        pred_mesh_img = self.renderer.render_mesh(**pred_mesh_kwargs)
        pred_mesh_img = self.adjust_rendered_img(pred_mesh_img)

        self.save_images(np.array(img), pred_mesh_img, shapy_mesh_img, output_path, l2_loss)
        self.save_obj(output_path / f"{img_id}_pred.obj", **pred_mesh_kwargs)
        self.save_obj(output_path / f"{img_id}_shapy.obj", **shapy_mesh_kwargs)
        self.save_shape_tensor(pred_shape_tensor, output_path / f"{img_id}_pred_shape_tensor.npy")
        self.save_shape_tensor(shapy_shape_tensor, output_path / f"{img_id}_shapy_shape_tensor.npy")

    def __call__(self):
        files_generator = list(self.shapy_dir.rglob("*hd_imgs.png"))
        for file in tqdm(files_generator, desc="Processing Image", total=len(files_generator)):
            self.process_data_unit(file)


@hydra.main(config_path="../config", config_name="compare_to_shapy")
def main(cfg: DictConfig):
    compare = CompareToShapy(**cfg)
    compare()


if __name__ == "__main__":
    main()
