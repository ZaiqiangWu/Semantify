import hydra
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from typing import Union, Literal
from clip2mesh.utils import Utils, ModelsFactory, Pytorch3dRenderer, Open3dRenderer, VertsIdx


class DataCreator:
    def __init__(
        self,
        output_path: str,
        model_type: Literal["smpl", "smal", "smplx", "flame"],
        gender: Literal["male", "female", "neutral"],
        multiview: bool = False,
        img_tag: str = None,
        with_face: bool = False,
        num_coeffs: int = 10,
        get_tall_data: bool = False,
        num_of_imgs: int = 1000,
        renderer_type: Literal["pytorch3d", "open3d"] = "pytorch3d",
        renderer_kwargs: DictConfig = None,
    ):
        # parameters from config
        self.multiview = multiview
        self.img_tag = img_tag
        self.with_face = with_face
        self.num_coeffs = num_coeffs
        self.get_tall_data = get_tall_data
        self.num_of_imgs = num_of_imgs
        self.output_path: Path = Path(output_path)
        self.model_type = model_type
        self.gender = gender
        self.renderer_type: Literal["pytorch3d", "open3d"] = renderer_type
        self.get_smpl = True if self.model_type == "smpl" else False

        # utils
        self.utils: Utils = Utils()
        self.models_factory: ModelsFactory = ModelsFactory(self.model_type)
        self._load_renderer(renderer_kwargs)

    def _load_renderer(self, kwargs):
        self.renderer: Union[Pytorch3dRenderer, Open3dRenderer] = self.models_factory.get_renderer(**kwargs)

    def __call__(self):
        # start creating data
        for _ in tqdm(range(self.num_of_imgs), total=self.num_of_imgs, desc="creating data"):

            # get image id
            try:
                img_id = (
                    int(
                        sorted(list(Path(self.output_path).glob("*.png")), key=lambda x: int(x.stem.split("_")[0]))[
                            -1
                        ].stem.split("_")[0]
                    )
                    + 1
                )
            except IndexError:
                img_id = 0

            # set image name
            img_name = self.img_tag if self.img_tag is not None else str(img_id)

            # get random 3DMM parameters
            model_kwargs = self.models_factory.get_random_params(
                with_face=self.with_face, num_coeffs=self.num_coeffs, tall_data=self.get_tall_data
            )

            if self.get_smpl:
                model_kwargs["get_smpl"] = True

            # extract verts, faces, vt, ft
            verts, faces, vt, ft = self.models_factory.get_model(
                **model_kwargs, gender=self.gender, num_coeffs=self.num_coeffs
            )
            if self.model_type == "flame":
                y_top_lip = verts[0, VertsIdx.TOP_LIP_MIN.value : VertsIdx.TOP_LIP_MAX.value, 1]
                y_bottom_lip = verts[0, VertsIdx.BOTTOM_LIP_MIN.value : VertsIdx.BOTTOM_LIP_MAX.value, 1]
                if y_top_lip - y_bottom_lip < 1e-3:
                    continue

            if self.model_type in ["smplx", "smpl"]:
                verts += self.utils.smplx_offset_numpy

            # render mesh and save image
            if self.renderer_type == "open3d":
                self.renderer.render_mesh()
                self.renderer.visualizer.capture_screen_image(f"{self.output_path}/{img_name}.png")
                self.renderer.visualizer.destroy_window()
            else:
                if self.multiview:
                    for azim in [0.0, 90.0]:
                        img_suffix = "front" if azim == 0.0 else "side"
                        img = self.renderer.render_mesh(
                            verts=verts, faces=faces[None], vt=vt, ft=ft, rotate_mesh={"degrees": azim, "axis": "y"}
                        )
                        self.renderer.save_rendered_image(img, f"{self.output_path}/{img_name}_{img_suffix}.png")
                else:
                    img = self.renderer.render_mesh(verts=verts, faces=faces[None], vt=vt, ft=ft)
                    self.renderer.save_rendered_image(img, f"{self.output_path}/{img_name}.png")

            self.utils.create_metadata(metadata=model_kwargs, file_path=f"{self.output_path}/{img_name}.json")


@hydra.main(config_path="../../config", config_name="create_data")
def main(cfg):
    data_creator = DataCreator(**cfg)
    data_creator()


if __name__ == "__main__":
    main()
