import cv2
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from typing import Tuple, Union, Literal
from pytorch3d.structures import Meshes
from clip2mesh.utils import Utils, ModelsFactory, Pytorch3dRenderer, Open3dRenderer


class DepthMaps:
    def __init__(self, args: DictConfig):
        self.utils = Utils()
        self.models_factory = ModelsFactory(model_type=args.model_type)
        self.out_dir = Path(args.out_dir)
        self.get_rgb = args.get_rgb
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.azims = [-90, -45, 0, 45, 90]

        self.verts, self.faces, self.vt, self.ft = self.get_verts_faces(args.gender)

        self._load_renderers(args.renderer_kwargs)
        self._load_default_mesh()

    def _load_renderers(self, kwargs):
        self.renderers = []
        if self.models_factory.model_type == "flame":
            for azim in self.azims:
                kwargs.update({"azim": azim})
                self.renderers.append(self.models_factory.get_renderer(py3d=True, **kwargs))
        elif self.models_factory.model_type == "smplx":
            self.renderers.append(self.models_factory.get_renderer(py3d=True, **kwargs))
        else:
            raise NotImplementedError

    def get_verts_faces(
        self, gender: Literal["male", "female", "neutral"] = "neutral"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        verts, faces, vt, ft = self.models_factory.get_model(gender=gender)
        return verts, faces, vt, ft

    def _load_default_mesh(self):
        self.mesh = self.renderers[0].get_mesh(verts=self.verts, faces=self.faces, vt=self.vt, ft=self.ft)

    def get_single_depth_map(self, mesh: Meshes, renderer: Union[Pytorch3dRenderer, Open3dRenderer]) -> np.ndarray:
        if isinstance(renderer, Open3dRenderer):
            raise NotImplementedError
        depth_map = renderer.depth_rasterizer(mesh).zbuf.cpu().numpy()
        return depth_map

    def get_rendered_image(self, renderer: Union[Pytorch3dRenderer, Open3dRenderer]) -> np.ndarray:
        if isinstance(renderer, Open3dRenderer):
            raise NotImplementedError
        rendered_image = renderer.render_mesh(verts=self.verts, faces=self.faces, vt=self.vt, ft=self.ft)
        rendered_image = rendered_image.detach().cpu().numpy().squeeze()
        rendered_image = np.clip(rendered_image[..., :3] * 255, 0, 255).astype(np.uint8)
        return rendered_image

    def __call__(self):
        for azim, renderer in zip(self.azims, self.renderers):
            depth_map = self.get_single_depth_map(self.mesh, renderer)
            depth_map = ~depth_map[0].astype(np.uint8)
            cv2.imwrite(str(self.out_dir / f"depth_map_{azim}.png"), depth_map)
            if self.get_rgb:
                rendered_image = self.get_rendered_image(renderer)
                rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(self.out_dir / f"rendered_image_{azim}.png"), rendered_image)


@hydra.main(config_path="../../config", config_name="depth_maps")
def main(args: DictConfig):
    depth_maps = DepthMaps(args)
    depth_maps()


if __name__ == "__main__":
    main()
