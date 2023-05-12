import os
import cv2
import json
import clip
import h5py
import torch
import smplx
import shutil
import base64
import numpy as np
import pandas as pd
import open3d as o3d
import altair as alt
import pickle as pkl
import pytorch_lightning as pl
from enum import Enum
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from omegaconf import DictConfig
from torch import nn
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Tuple, Literal, List, Dict, Any, Optional, Union
from torch.nn import functional as F
from attrdict import AttrDict
from pytorch_lightning import Callback
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
    BlendParams,
    Materials,
)
from clip2mesh.three_dmm.smal_layer import get_smal_layer
from clip2mesh.three_dmm.flame import FLAME


def get_root_dir():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return root_dir

def append_to_root_dir(path: str) -> str:
    root_dir = get_root_dir()
    return os.path.join(root_dir, path)


def get_model_abs_path(
    model: Literal["smplx", "smpl", "flame", "smal"],
    specific: Optional[Literal["male", "female", "neutral", "expression", "shape"]] = None,
) -> str:
    root_dir = get_root_dir()
    if model in ["smplx", "smpl"]:
        assert specific in ["male", "female", "neutral"], "unrecognized model type"
        model_path = os.path.join(root_dir, "models_ckpts", model, f"{model}_{specific}.ckpt")
    elif model == "flame":
        assert specific in ["expression", "shape"], "unrecognized model type"
        model_path = os.path.join(
            root_dir,
            "models_ckpts",
            model,
            specific,
            f"{model}_{specific}.ckpt",
        )
    elif model == "smal":
        model_path = os.path.join(root_dir, "models_ckpts", model, f"{model}.ckpt")
    else:
        raise ValueError("unrecognized model type")
    return model_path


def get_model_feature_name(
    model: Literal["smplx", "smpl", "flame", "smal"],
    specific: Optional[Literal["expression", "shape"]] = None,
) -> str:
    if model in ["smplx", "smpl"]:
        feature_name = "betas"
    elif model == "flame":
        assert specific in ["expression", "shape"], "unrecognized model type"
        if specific == "expression":
            feature_name = "expression_params"
        else:
            feature_name = "shape_params"
    elif model == "smal":
        feature_name = "beta"
    else:
        raise ValueError("unrecognized model type")
    return feature_name


class C2M(nn.Module):
    def __init__(self, num_stats: int, hidden_size: int, out_features: int, num_hiddens: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(num_stats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size - 200)
        self.fc5 = nn.Linear(hidden_size - 200, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.dropout(self.fc2(x)))
        x = F.relu(self.fc2(x))
        x = self.fc5(x)
        return x


class C2M_new(nn.Module):
    def __init__(self, num_stats: int, hidden_size: Union[List[int], int], out_features: int, num_hiddens: int = 0):
        super().__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        self.fc_layers = []
        self.fc_layers.extend([nn.Linear(num_stats, hidden_size[0]), nn.ReLU()])  # , nn.Dropout(0.2)])
        if num_hiddens > 0:
            for i in range(num_hiddens):
                self.fc_layers.extend([nn.Linear(hidden_size[i], hidden_size[i + 1]), nn.ReLU()])
        self.fc_layers = nn.Sequential(*self.fc_layers)
        self.out_layer = nn.Linear(hidden_size[-1], out_features)

    def forward(self, x):
        x = self.fc_layers(x)

        return self.out_layer(x)


class Open3dRenderer:
    def __init__(
        self,
        verts: torch.tensor,
        faces: torch.tensor,
        vt: torch.tensor = None,
        ft: torch.tensor = None,
        texture: str = None,
        light_on: bool = True,
        for_image: bool = True,
        img_size: Tuple[int, int] = (512, 512),
        paint_vertex_colors: bool = False,
    ):
        self.verts = verts
        self.faces = faces
        self.height, self.width = img_size
        self.paint_vertex_colors = paint_vertex_colors
        self.texture = cv2.cvtColor(cv2.imread(texture), cv2.COLOR_BGR2RGB) if texture is not None else None
        self.vt = vt
        self.ft = ft
        if self.vt is not None and self.ft is not None:
            uvs = np.concatenate([self.vt[self.ft[:, ind]][:, None] for ind in range(3)], 1).reshape(-1, 2)
            uvs[:, 1] = 1 - uvs[:, 1]
        else:
            uvs = None
        self.uvs = uvs
        self.for_image = for_image
        self.visualizer = o3d.visualization.Visualizer()
        self.default_zoom_value = 0.55
        self.default_y_rotate_value = 70.0
        self.default_up_translate_value = 0.3
        self.visualizer.create_window(width=self.width, height=self.height)
        opt = self.visualizer.get_render_option()
        if self.paint_vertex_colors:
            opt.background_color = np.asarray([255.0, 255.0, 255.0])
        else:
            opt.background_color = np.asarray([0.0, 0.0, 0.0])
        self.visualizer.get_render_option().light_on = light_on
        self.ctr = self.visualizer.get_view_control()
        self.ctr.set_zoom(self.default_zoom_value)
        self.ctr.camera_local_rotate(0.0, self.default_y_rotate_value, 0.0)
        self.ctr.camera_local_translate(0.0, 0.0, self.default_up_translate_value)
        self.mesh = self.get_initial_mesh()
        self.visualizer.add_geometry(self.mesh)
        self.mesh.compute_vertex_normals()

    def get_texture(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:

        if self.texture is not None and isinstance(self.texture, str):
            self.texture = cv2.cvtColor(cv2.imread(self.texture), cv2.COLOR_BGR2RGB)
        mesh.textures = [o3d.geometry.Image(self.texture)]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(self.uvs)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def get_initial_mesh(self) -> o3d.geometry.TriangleMesh:
        verts = (self.verts.squeeze() - self.verts.min()) / (self.verts.max() - self.verts.min())
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.squeeze())
        if self.texture is not None:
            mesh = self.get_texture(mesh)

        if self.uvs is not None:
            mesh.triangle_uvs = o3d.utility.Vector2dVector(self.uvs)

        if self.paint_vertex_colors:
            mesh.paint_uniform_color([0.2, 0.8, 0.2])

        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(self.faces)), dtype=np.int32))
        return mesh

    def render_mesh(self, verts: torch.tensor = None, texture: np.array = None):

        if verts is not None:
            verts = (verts.squeeze() - verts.min()) / (verts.max() - verts.min())
            self.mesh.vertices = o3d.utility.Vector3dVector(verts.squeeze())
            self.visualizer.update_geometry(self.mesh)
        if texture is not None:
            self.texture = texture
            self.mesh = self.get_texture(self.mesh)
            self.visualizer.update_geometry(self.mesh)
        if self.for_image:
            self.visualizer.update_renderer()
            self.visualizer.poll_events()
        else:
            self.visualizer.run()

    def close(self):
        self.visualizer.close()

    def remove_texture(self):
        self.mesh.textures = []
        self.visualizer.update_geometry(self.mesh)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def zoom(self, zoom_value: float):
        self.ctr.set_zoom(zoom_value)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def rotate(self, y: float = None, x: float = 0.0, z: float = 0.0):
        if y is None:
            y = self.default_y_rotate_value
        self.ctr.camera_local_rotate(x, y, z)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def translate(self, right: float = 0.0, up: float = None):
        if up is None:
            up = self.default_up_translate_value
        self.ctr.camera_local_translate(0.0, right, up)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def reset_camera(self):
        self.ctr.set_zoom(self.default_zoom_value)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def save_mesh(self, path: str):
        o3d.io.write_triangle_mesh(path, self.mesh)


class SMPLXParams:
    def __init__(
        self,
        betas: torch.tensor = None,
        expression: torch.tensor = None,
        body_pose: torch.tensor = None,
        global_orient: torch.tensor = None,
        transl: torch.tensor = None,
        smpl_model: bool = False,
        num_coeffs: int = 10,
    ):
        if betas is not None:
            betas: torch.Tensor = betas
        else:
            betas = torch.zeros(1, num_coeffs)
        if expression is not None:
            expression: torch.Tensor = expression
        else:
            expression: torch.Tensor = torch.zeros(1, 10)
        if body_pose is not None:
            body_pose: torch.Tensor = body_pose
        else:
            body_pose = torch.eye(3).expand(1, 21, 3, 3)
        left_hand_pose: torch.Tensor = torch.eye(3).expand(1, 15, 3, 3)
        right_hand_pose: torch.Tensor = torch.eye(3).expand(1, 15, 3, 3)
        if global_orient is not None:
            global_orient: torch.Tensor = global_orient
        else:
            global_orient: torch.Tensor = torch.eye(3).expand(1, 1, 3, 3)
        if transl is not None:
            transl: torch.Tensor = transl
        else:
            transl: torch.Tensor = torch.zeros(1, 3)
        jaw_pose: torch.Tensor = torch.eye(3).expand(1, 1, 3, 3)
        if smpl_model:
            self.params = {"betas": betas, "body_pose": torch.cat([body_pose, torch.eye(3).expand(1, 2, 3, 3)], dim=1)}
        else:
            self.params = {
                "betas": betas,
                "body_pose": body_pose,
                "left_hand_pose": left_hand_pose,
                "right_hand_pose": right_hand_pose,
                "global_orient": global_orient,
                "transl": transl,
                "jaw_pose": jaw_pose,
                "expression": expression,
            }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}


class FLAMEParams:
    def __init__(
        self,
        shape_params: torch.tensor = None,
        expression_params: torch.tensor = None,
        jaw_pose: torch.Tensor = None,
    ):
        if shape_params is None:
            shape_params = torch.zeros(1, 100, dtype=torch.float32)
        if expression_params is None:
            expression_params = torch.zeros(1, 50, dtype=torch.float32)
        shape_params = shape_params.cuda()
        expression_params = expression_params.cuda()
        if jaw_pose is None:
            pose_params_t = torch.zeros(1, 6, dtype=torch.float32)
        else:
            pose_params_t = torch.cat([torch.zeros(1, 3), torch.tensor([[jaw_pose, 0.0, 0.0]])], 1)
        pose_params = pose_params_t.cuda()
        self.params = {
            "shape_params": shape_params,
            "expression_params": expression_params,
            "pose_params": pose_params,
        }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}


class SMALParams:
    def __init__(self, beta: torch.tensor = None):
        if beta is None:
            beta = torch.zeros(1, 41, dtype=torch.float32)
        beta = beta.cuda()
        theta = torch.eye(3).expand(1, 35, 3, 3).to("cuda")
        self.params = {
            "beta": beta,
            "theta": theta,
        }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}


class TexturesPaths(Enum):
    SMPLX = append_to_root_dir("SMPLX/textures/smplx_texture_m_alb.png")
    SMPL = append_to_root_dir("SMPLX/textures/smplx_texture_m_alb.png")
    FLAME = append_to_root_dir("Flame/flame2020/mesh.png")
    SMAL = None


class Pytorch3dRenderer:
    def __init__(
        self,
        device="cuda",
        dist: float = 0.5,
        elev: float = 0.0,
        azim: float = 0.0,
        img_size: Tuple[int, int] = (224, 224),
        texture_optimization: bool = False,
        use_tex: bool = False,
        model_type: Literal["smplx", "smal", "flame", "smpl"] = None,
        background_color: Tuple[float, float, float] = (255.0, 255.0, 255.0),
    ):

        self.device = device
        self.background_color = background_color
        self.texture_optimization = texture_optimization
        tex_path = TexturesPaths[model_type.upper()].value if model_type is not None and use_tex else None
        self.tex_map = cv2.cvtColor(cv2.imread(tex_path), cv2.COLOR_BGR2RGB) if tex_path is not None else None
        self.height, self.width = img_size

        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev)
        self.cameras = FoVPerspectiveCameras(znear=0.1, T=T, R=R, fov=30).to(self.device)
        lights = self.get_lights(self.device)
        materials = self.get_default_materials(self.device)
        blend_params = self.get_default_blend_params()
        self.shader = SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        )
        self._load_renderer()
        self._load_depth_rasterizer()

    def _load_renderer(self):
        self.raster_settings = RasterizationSettings(image_size=(self.height, self.width))
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=self.shader,
        )

    @staticmethod
    def get_texture(device, vt, ft, texture):
        verts_uvs = torch.as_tensor(vt, dtype=torch.float32, device=device)
        faces_uvs = torch.as_tensor(ft, dtype=torch.long, device=device)

        texture_map = torch.as_tensor(texture, device=device, dtype=torch.float32) / 255.0

        texture = TexturesUV(
            maps=texture_map[None],
            faces_uvs=faces_uvs[None],
            verts_uvs=verts_uvs[None],
        )
        return texture

    def _load_depth_rasterizer(self):
        self.depth_rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

    @staticmethod
    def get_lights(device):
        lights = PointLights(
            device=device,
            ambient_color=((0.8, 0.8, 0.8),),
            specular_color=((0.2, 0.2, 0.2),),
            location=[[0.0, 2.0, 2.0]],
        )
        return lights

    @staticmethod
    def get_default_materials(device):
        materials = Materials(device=device)  # , shininess=12)
        return materials

    def get_default_blend_params(self):
        blend_params = BlendParams(sigma=1e-6, gamma=1e-6, background_color=self.background_color)
        return blend_params

    @staticmethod
    def rotate_3dmm_verts(
        verts: Union[torch.Tensor, np.ndarray], degrees: float, axis: Literal["x", "y", "z"]
    ) -> Meshes:
        convert_back_to_numpy = False
        if isinstance(verts, np.ndarray):
            convert_back_to_numpy = True
            verts = torch.tensor(verts).float()
        rotation_matrix = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
        axis = 0 if verts.dim() == 2 else 1
        mesh_center = verts.mean(axis=axis)
        mesh_center_cloned = torch.tensor(mesh_center.clone().detach()).to(verts.device).float()
        rotation_matrix = torch.tensor(rotation_matrix).to(verts.device).float()
        verts = verts - mesh_center_cloned
        verts = verts @ rotation_matrix
        verts = verts + mesh_center_cloned
        if convert_back_to_numpy:
            verts = verts.cpu().numpy()
        return verts

    def render_mesh(
        self,
        verts: Union[torch.Tensor, np.ndarray] = None,
        faces: Union[torch.Tensor, np.ndarray] = None,
        vt: Optional[Union[torch.Tensor, np.ndarray]] = None,
        ft: Optional[Union[torch.Tensor, np.ndarray]] = None,
        mesh: Meshes = None,
        texture_color_values: Optional[torch.Tensor] = None,
        rotate_mesh: List[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        assert mesh is not None or (
            verts is not None and faces is not None
        ), "either mesh or verts and faces must be provided"
        if mesh is None:
            if rotate_mesh is not None:
                if isinstance(rotate_mesh, dict):
                    rotate_mesh = [rotate_mesh]
                for rot_action in rotate_mesh:
                    verts = self.rotate_3dmm_verts(verts, **rot_action)
            mesh = self.get_mesh(verts, faces, vt, ft, texture_color_values)

        rendered_mesh = self.renderer(mesh, cameras=self.cameras)
        return rendered_mesh

    def get_mesh(self, verts, faces, vt=None, ft=None, texture_color_values: torch.Tensor = None) -> Meshes:
        verts = torch.as_tensor(verts, dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        if self.tex_map is not None:
            assert vt is not None and ft is not None, "vt and ft must be provided if texture is provided"
            texture = self.get_texture(self.device, vt, ft, self.tex_map)
        else:
            if len(verts.shape) == 2:
                verts = verts[None]

            if self.texture_optimization and texture_color_values is not None:
                texture = TexturesVertex(verts_features=texture_color_values)
            else:
                texture = TexturesVertex(
                    verts_features=torch.ones(*verts.shape, device=self.device)
                    * torch.tensor([0.7, 0.7, 0.7], device=self.device)
                )
        if len(verts.size()) == 2:
            verts = verts[None]
        if len(faces.size()) == 2:
            faces = faces[None]
        mesh = Meshes(verts=verts, faces=faces, textures=texture)
        return mesh

    def save_rendered_image(self, image, path, resize: bool = True):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().squeeze()
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        if resize:
            image = cv2.resize(image, (512, 512))
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class Utils:
    def __init__(self, device: str = "cuda", comparison_mode: bool = False):
        self.device = device
        self.body_pose = torch.tensor(np.load(append_to_root_dir("SMPLX/a_pose.npy")))
        self.production_dir = append_to_root_dir("pre_production")
        self.comparison_mode = comparison_mode

    @staticmethod
    def find_multipliers(value: int) -> list:
        """
        Description
        -----------
        finds all of the pairs that their product is the value
        Args
        ----
        value (int) = a number that you would like to get its multipliers
        Returns
        -------
        list of the pairs that their product is the value
        """
        factors = []
        for i in range(1, int(value**0.5) + 1):
            if value % i == 0:
                factors.append((i, value / i))
        return factors

    def get_plot_shape(self, value: int) -> Tuple[Tuple[int, int], int]:
        """
        Description
        -----------
        given a number it finds the best pair of integers that their product
        equals the given number.
        for example, given an input 41 it will return 5 and 8
        """
        options_list = self.find_multipliers(value)
        if len(options_list) == 1:
            while len(options_list) == 1:
                value -= 1
                options_list = self.find_multipliers(value)

        chosen_multipliers = None
        min_distance = 100
        for option in options_list:
            if abs(option[0] - option[1]) < min_distance:
                chosen_multipliers = (option[0], option[1])

        # it is better that the height will be the largest value since the image is wide
        chosen_multipliers = (
            int(chosen_multipliers[np.argmax(chosen_multipliers)]),
            int(chosen_multipliers[1 - np.argmax(chosen_multipliers)]),
        )

        return chosen_multipliers, int(value)

    @staticmethod
    def flatten_list_of_lists(list_of_lists):
        return [l[0] for l in list_of_lists]

    @staticmethod
    def create_metadata(metadata: Dict[str, torch.tensor], file_path: str):
        # write tensors to json
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            metadata[key] = value

        with open(file_path, "w") as f:
            json.dump(metadata, f)

    def _get_smplx_layer(self, gender: str, num_coeffs: int, get_smpl: bool):
        if get_smpl:
            if gender == "male":
                smplx_path = append_to_root_dir("SMPL/smpl_male.pkl")
            elif gender == "female":
                smplx_path = append_to_root_dir("SMPL/smpl_female.pkl")
            else:
                smplx_path = append_to_root_dir("SMPL/SMPL_NEUTRAL.pkl")
        else:
            if gender == "neutral":
                smplx_path = append_to_root_dir("SMPLX/SMPLX_NEUTRAL_2020.npz")
            elif gender == "male":
                smplx_path = append_to_root_dir("SMPLX/SMPLX_MALE.npz")
            else:
                smplx_path = append_to_root_dir("SMPLX/SMPLX_FEMALE.npz")
        self.smplx_layer = smplx.build_layer(model_path=smplx_path, num_expression_coeffs=10, num_betas=num_coeffs)
        if get_smpl:
            self.smplx_faces = self.smplx_layer.faces_tensor
        else:
            model_data = np.load(smplx_path, allow_pickle=True)
            self.smplx_faces = model_data["f"].astype(np.int32)

    def _get_flame_layer(self, gender: Literal["male", "female", "neutral"]) -> FLAME:
        cfg = self.get_flame_model_kwargs(gender)
        self.flame_layer = FLAME(cfg).cuda()

    @property
    def smplx_offset_tensor(self):
        return torch.tensor([0.0, 0.4, 0.0], device=self.device)

    @property
    def smplx_offset_numpy(self):
        return np.array([0.0, 0.4, 0.0])

    @property
    def smpl_offset_numpy(self):
        return np.array([0.0, 0.4, 0.0])

    @property
    def smpl_offset_tensor(self):
        return torch.tensor([0.0, 0.4, 0.0], device=self.device)

    def get_smplx_model(
        self,
        betas: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        expression: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        gender: Literal["neutral", "male", "female"] = "neutral",
        device: Optional[Literal["cuda", "cpu"]] = "cpu",
        num_coeffs: int = 10,
        get_smpl: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        smplx_model = SMPLXParams(
            betas=betas,
            body_pose=body_pose,
            expression=expression,
            global_orient=global_orient,
            transl=transl,
            smpl_model=get_smpl,
            num_coeffs=num_coeffs,
        )
        if self.comparison_mode:
            self._get_smplx_layer(gender, num_coeffs, get_smpl)
        else:
            if not hasattr(self, "smplx_layer") or not hasattr(self, "smplx_faces"):
                self._get_smplx_layer(gender, num_coeffs, get_smpl)

        if device == "cuda":
            smplx_model.params = smplx_model.to(device)
            self.smplx_layer = self.smplx_layer.cuda()
            verts = self.smplx_layer(**smplx_model.params).vertices
        else:
            verts = self.smplx_layer(**smplx_model.params).vertices
            verts = verts.detach().cpu().numpy()
        verts = verts.squeeze()
        if not hasattr(self, "vt_smplx") and not hasattr(self, "ft_smplx"):
            model_type = "smpl" if get_smpl else "smplx"
            self._get_vt_ft(model_type)

        return verts, self.smplx_faces, self.vt_smplx, self.ft_smplx

    def _get_vt_ft(self, model_type: Literal["smplx", "flame", "smpl"]) -> Tuple[np.ndarray, np.ndarray]:
        if model_type == "smplx":
            self.vt_smplx = np.load("/home/nadav2/dev/repos/CLIP2Shape/SMPLX/textures/smplx_vt.npy")
            self.ft_smplx = np.load("/home/nadav2/dev/repos/CLIP2Shape/SMPLX/textures/smplx_ft.npy")
        elif model_type == "smpl":
            self.vt_smplx = np.load("/home/nadav2/dev/repos/CLIP2Shape/SMPLX/textures/smpl_uv_map.npy")
            self.ft_smplx = self.smplx_faces
        else:
            flame_uv_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/flame_texture_data_v6.pkl"
            flame_uv = np.load(flame_uv_path, allow_pickle=True)
            self.vt_flame = flame_uv["vt_plus"]
            self.ft_flame = flame_uv["ft_plus"]

    def _get_flame_faces(self) -> np.ndarray:
        flame_uv_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/flame_texture_data_v6.pkl"
        flame_uv = np.load(flame_uv_path, allow_pickle=True)
        self.flame_faces = flame_uv["f_plus"]

    def _get_smal_faces(self) -> np.ndarray:
        smal_model_path = "/home/nadav2/dev/repos/CLIP2Shape/SMAL/smal_CVPR2017.pkl"
        with open(smal_model_path, "rb") as f:
            smal_model = pkl.load(f, encoding="latin1")
        self.smal_faces = smal_model["f"].astype(np.int32)

    @staticmethod
    def init_flame_params_dict(device: str = "cuda") -> Dict[str, torch.tensor]:
        flame_dict = {}
        flame_dict["shape_params"] = torch.zeros(1, 300)
        flame_dict["expression_params"] = torch.zeros(1, 100)
        flame_dict["global_rot"] = torch.zeros(1, 3)
        flame_dict["jaw_pose"] = torch.zeros(1, 3)
        flame_dict["neck_pose"] = torch.zeros(1, 3)
        flame_dict["transl"] = torch.zeros(1, 3)
        flame_dict["eye_pose"] = torch.zeros(1, 6)
        flame_dict["shape_offsets"] = torch.zeros(1, 5023, 3)
        flame_dict = {k: v.to(device) for k, v in flame_dict.items()}
        return flame_dict

    @staticmethod
    def get_flame_model_kwargs(gender: Literal["male", "female", "neutral"]) -> Dict[str, Any]:
        if gender == "male":
            flame_model_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/male_model.pkl"
        elif gender == "female":
            flame_model_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame/female_model.pkl"
        else:
            flame_model_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/generic_model.pkl"

        kwargs = {
            "batch_size": 1,
            "use_face_contour": False,
            "use_3D_translation": True,
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "shape_params": 100,
            "expression_params": 50,
            "flame_model_path": flame_model_path,
            "ring_margin": 0.5,
            "ring_loss_weight": 1.0,
            "static_landmark_embedding_path": "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/flame_static_embedding_68.pkl",
            "pose_params": 6,
        }
        return AttrDict(kwargs)

    def get_flame_model(
        self,
        shape_params: torch.tensor = None,
        expression_params: torch.tensor = None,
        jaw_pose: float = None,
        gender: Literal["male", "female", "neutral"] = "neutral",
        device: Optional[Literal["cuda", "cpu"]] = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "flame_layer"):
            self._get_flame_layer(gender)
        if shape_params is not None and shape_params.shape == (1, 10):
            shape_params = torch.cat([shape_params, torch.zeros(1, 90).to(device)], dim=1)
        if expression_params is not None and expression_params.shape == (1, 10):
            expression_params = torch.cat([expression_params, torch.zeros(1, 40).to(device)], dim=1)
        flame_params = FLAMEParams(shape_params=shape_params, expression_params=expression_params, jaw_pose=jaw_pose)
        if device == "cuda":
            flame_params.params = flame_params.to(device)
        verts, _ = self.flame_layer(**flame_params.params)
        if device == "cpu":
            verts = verts.cpu()
        if not hasattr(self, "flame_faces"):
            self._get_flame_faces()

        if not hasattr(self, "vt_flame") and not hasattr(self, "ft_flame"):
            self._get_vt_ft("flame")

        return verts, self.flame_faces, self.vt_flame, self.ft_flame

    def get_smal_model(
        self, beta: torch.tensor, device: Optional[Literal["cuda", "cpu"]] = "cpu", py3d: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, None, None]:
        if not hasattr(self, "smal_layer"):
            self.smal_layer = get_smal_layer()
        smal_params = SMALParams(beta=beta)
        if device == "cuda":
            smal_params.params = smal_params.to(device)
            self.smal_layer = self.smal_layer.cuda()
            verts = self.smal_layer(**smal_params.params)[0]
        else:
            verts = self.smal_layer(**smal_params.params)[0].detach().cpu().numpy()
        verts = self.rotate_mesh_smal(verts, py3d)
        if not hasattr(self, "smal_faces"):
            self._get_smal_faces()
        return verts, self.smal_faces, None, None

    def get_body_pose(self) -> torch.Tensor:
        return self.body_pose

    @staticmethod
    def rotate_mesh_smal(verts: Union[np.ndarray, torch.Tensor], py3d: bool = True) -> np.ndarray:
        rotation_matrix_x = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        rotation_matrix_y = Rotation.from_euler("y", 75, degrees=True).as_matrix()
        rotation_matrix = np.matmul(rotation_matrix_x, rotation_matrix_y)
        if not py3d:
            rotation_matrix_z = Rotation.from_euler("x", -15, degrees=True).as_matrix()
            rotation_matrix = np.matmul(rotation_matrix, rotation_matrix_z)
        mesh_center = verts.mean(axis=1)
        if isinstance(verts, torch.Tensor):
            mesh_center = torch.tensor(mesh_center).to(verts.device).float()
            rotation_matrix = torch.tensor(rotation_matrix).to(verts.device).float()
        verts = verts - mesh_center
        verts = verts @ rotation_matrix
        verts = verts + mesh_center
        return verts

    @staticmethod
    def translate_mesh_smplx(
        verts: Union[np.ndarray, torch.tensor],
        translation_vector: Union[np.ndarray, torch.tensor] = np.array([-0.55, -0.3, 0.0]),
    ) -> Union[np.ndarray, torch.tensor]:
        if isinstance(verts, torch.Tensor):
            translation_vector = torch.tensor(translation_vector).to(verts.device)
        verts += translation_vector
        return verts

    @staticmethod
    def get_labels() -> List[List[str]]:
        # labels = [["big cat"], ["cow"], ["donkey"], ["hippo"], ["dog"]]  # SMAL animals
        labels = [
            "big",
            "fat",
            "broad shoulders",
            "built",
            "curvy",
            "feminine",
            "fit",
            "heavyset",
            "lean",
            "long legs",
            "tall",
            "long torso",
            "long",
            "masculine",
            "muscular",
            "pear shaped",
            "petite",
            "proportioned",
            "rectangular",
            "round apple",
            "short legs",
            "short torso",
            "short",
            "skinny",
            "small",
            "stocky",
            "sturdy",
            "tall",
            "attractive",
            "sexy",
            "hourglass",
        ]
        if not isinstance(labels[0], list):
            labels = [[label] for label in labels]
        return labels

    @staticmethod
    def get_random_betas_smplx(num_coeffs: int = 10, tall_data: bool = False) -> torch.Tensor:
        """SMPLX body shape"""
        random_offset = torch.randint(-2, 2, (1, num_coeffs)).float()
        if tall_data:
            random_offset[:, 0] = 4.0
        return torch.randn(1, num_coeffs) * random_offset

    @staticmethod
    def get_random_betas_smal(num_coeffs: int = 10) -> torch.Tensor:
        """SMAL body shape"""
        shape = torch.rand(1, num_coeffs) * torch.randint(-2, 2, (1, num_coeffs)).float()
        if num_coeffs < MaxCoeffs.SMAL.value:
            shape = torch.cat([shape, torch.zeros(1, MaxCoeffs.SMAL.value - num_coeffs)], 1)
        return shape

    @staticmethod
    def get_random_shape(num_coeffs: int = 10) -> torch.Tensor:
        """FLAME face shape"""
        shape = torch.randn(1, num_coeffs) * torch.randint(-2, 2, (1, num_coeffs)).float()
        if num_coeffs < MaxCoeffs.FLAME_SHAPE.value:
            shape = torch.cat([shape, torch.zeros(1, MaxCoeffs.FLAME_SHAPE.value - num_coeffs)], 1)
        # return torch.cat([shape, torch.zeros(1, 90)], dim=1)
        return shape

    @staticmethod
    def get_random_expression_flame(num_coeffs: int = 10) -> torch.Tensor:
        """FLAME face expression"""
        expression = torch.randn(1, num_coeffs)  # * torch.randint(-3, 3, (1, num_coeffs)).float()
        if num_coeffs < MaxCoeffs.FLAME_EXPRESSION.value:
            expression = torch.cat([expression, torch.zeros(1, MaxCoeffs.FLAME_EXPRESSION.value - num_coeffs)], 1)
        return expression

    @staticmethod
    def get_random_jaw_pose() -> torch.Tensor:
        """FLAME jaw pose"""
        # jaw_pose = torch.randn(1, 1).abs() * torch.tensor(np.random.choice([0.5, 0], p=[0.1, 0.9]))
        jaw_pose = F.relu(torch.randn(1, 1)) * torch.tensor([0.1])
        return jaw_pose

    @staticmethod
    def convert_str_list_to_float_tensor(strs_list: List[str]) -> torch.Tensor:
        stats = [float(stat) for stat in strs_list[0].split(" ")]
        return torch.tensor(stats, dtype=torch.float32)[None]

    @staticmethod
    def normalize_data(data, min_max_dict):
        for key, value in data.items():
            min_val, max_val, _ = min_max_dict[key]
            data[key] = (value - min_val) / (max_val - min_val)
        return data

    def filter_params_hack(self, ckpt: Dict[str, Any], convert_legacy: bool = False) -> Dict[str, Any]:
        hack = {key.split("model.")[-1]: ckpt["state_dict"][key] for key in ckpt["state_dict"].keys() if "model" in key}
        if convert_legacy:
            hack = self.convert_legacy(hack)
        return hack

    @staticmethod
    def convert_legacy(legacy_dict: Dict[str, Any]) -> Dict[str, Any]:
        converted_dict = {}
        layers_counter = 0
        last_layer = (legacy_dict.keys().__len__() // 2) - 1
        layers_renumbered = np.linspace(0, last_layer, 2)
        for key in legacy_dict.keys():
            if layers_counter == last_layer:
                prefix = "out_layer"
            else:
                prefix = f"fc_layers.{int(layers_renumbered[layers_counter])}"
            converted_dict[f"{prefix}.{key.split('.')[-1]}"] = legacy_dict[key]
            if "bias" in key and not prefix == "out_layer":
                layers_counter += 1
        return converted_dict

    def get_model_to_eval(self, model_path: str) -> Tuple[nn.Module, List[List[str]]]:
        model_meta_path = model_path.replace(".ckpt", "_metadata.json")
        with open(model_meta_path, "r") as f:
            model_meta = json.load(f)

        # kwargs from metadata
        labels = model_meta["labels"]
        model_meta.pop("labels")
        if "lr" in model_meta:
            model_meta.pop("lr")

        # load model
        ckpt = torch.load(model_path)
        convert_legacy = False if "fc_layers" in list(ckpt["state_dict"].keys())[0] else True
        filtered_params_hack = self.filter_params_hack(ckpt, convert_legacy=convert_legacy)
        model = C2M_new(**model_meta).to(self.device)
        model.load_state_dict(filtered_params_hack)
        model.eval()

        return model, labels

    @staticmethod
    def get_default_parameters(body_pose: bool = False, num_coeffs: int = 10) -> torch.Tensor:
        if body_pose:
            return torch.eye(3).expand(1, 21, 3, 3)
        return torch.zeros(1, num_coeffs)

    @staticmethod
    def get_default_face_shape() -> torch.Tensor:
        return torch.zeros(1, 100)

    @staticmethod
    def get_default_face_expression() -> torch.Tensor:
        return torch.zeros(1, 50)

    @staticmethod
    def get_default_shape_smal() -> torch.Tensor:
        return torch.zeros(1, 41)

    @staticmethod
    def get_min_max_values(working_dir: str) -> Dict[str, Tuple[float, float, float]]:
        stats = {}
        min_max_dict = {}

        for file in Path(working_dir).rglob("*_labels.json"):
            with open(file.as_posix(), "r") as f:
                data = json.load(f)
            for key, value in data.items():
                if key not in stats:
                    stats[key] = []
                stats[key].append(value)

        for key, value in stats.items():
            stats[key] = np.array(value)
            # show min and max
            min_max_dict[key] = (np.min(stats[key]), np.max(stats[key]), np.mean(stats[key]))
        return min_max_dict

    @staticmethod
    def video_to_frames(video_path: Union[str, Path]):
        video_path = Path(video_path)
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        vidcap = cv2.VideoCapture(video_path.as_posix())
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"{frames_dir}/{count:05d}.png", image)
            success, image = vidcap.read()
            count += 1


class C2M_pl(pl.LightningModule):
    def __init__(
        self,
        num_stats: int,
        lr: float = 0.0001,
        out_features: int = 10,
        hidden_size: Union[int, List[int]] = 300,
        num_hiddens: int = 0,
        labels: List[List[str]] = None,
    ):
        super().__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        self.save_hyperparameters()
        self.model = C2M_new(
            num_stats=num_stats, out_features=out_features, hidden_size=hidden_size, num_hiddens=num_hiddens
        )
        self.lr = lr
        self.utils = Utils()
        self.out_features = out_features
        self.labels = labels

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        parameters, clip_labels = batch
        b = parameters.shape[0]
        parameters_pred = self(clip_labels)
        parameters_pred = parameters_pred.reshape(b, 1, self.out_features)
        loss = F.mse_loss(parameters, parameters_pred)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class CreateModelMeta(Callback):
    def __init__(self):
        self.utils = Utils()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path is None:
            return
        ckpt_new_name = f"{trainer.logger.name}.ckpt"
        ckpt_new_path = ckpt_path.replace(ckpt_path.split("/")[-1], ckpt_new_name)
        os.rename(ckpt_path, ckpt_new_path)
        shutil.copy(ckpt_new_path, f"{self.utils.production_dir}/{ckpt_new_name}")
        pl_module.hparams.hidden_size = list(pl_module.hparams.hidden_size)
        metadata = dict(pl_module.hparams)
        if pl_module.hparams.labels is None:
            metadata = {"labels": self.utils.get_labels()}
        with open(
            f"{self.utils.production_dir}/{ckpt_new_path.split('/')[-1].replace('.ckpt', '_metadata.json')}", "w"
        ) as f:
            json.dump(metadata, f)


class ModelsFactory:
    def __init__(self, model_type: Literal["flame", "smplx", "smal", "smpl"]):
        self.model_type = model_type
        self.utils = Utils()

    def get_model(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.model_type == "smplx" or self.model_type == "smpl":
            return self.utils.get_smplx_model(**kwargs)
        elif self.model_type == "flame":
            if "num_coeffs" in kwargs:
                kwargs.pop("num_coeffs")
            return self.utils.get_flame_model(**kwargs)
        else:
            if "gender" in kwargs:
                kwargs.pop("gender")
            if "num_coeffs" in kwargs:
                kwargs.pop("num_coeffs")
            return self.utils.get_smal_model(**kwargs)

    def get_default_params(self, with_face: bool = False, num_coeffs: int = 10) -> Dict[str, torch.tensor]:

        params = {}

        if self.model_type == "smplx" or self.model_type == "smpl":
            params["body_pose"] = self.utils.get_default_parameters(body_pose=True)
            params["betas"] = self.utils.get_default_parameters(num_coeffs=num_coeffs)
            expression = None
            if with_face:
                expression = self.utils.get_default_face_expression()
            params["expression"] = expression

        elif self.model_type == "flame":
            params["shape_params"] = self.utils.get_default_face_shape()
            params["expression_params"] = self.utils.get_default_face_expression()

        else:
            params["beta"] = self.utils.get_default_parameters()

        return params

    def get_vt_ft(self):
        return self.utils.get_vt_ft(self.model_type)

    def get_renderer(self, py3d: bool = False, **kwargs) -> Union[Open3dRenderer, Pytorch3dRenderer]:
        if py3d:
            return Pytorch3dRenderer(**kwargs)
        return Open3dRenderer(**kwargs)

    def get_random_params(
        self, with_face: bool = False, num_coeffs: int = 10, tall_data: bool = False
    ) -> Dict[str, torch.tensor]:
        params = {}
        if self.model_type in ["smplx", "smpl"]:
            params["betas"] = self.utils.get_random_betas_smplx(num_coeffs, tall_data=tall_data)
        elif self.model_type == "flame":
            if with_face:
                params["expression_params"] = self.utils.get_random_expression_flame(num_coeffs)
            else:
                params["shape_params"] = self.utils.get_random_shape(num_coeffs)

        else:
            params["beta"] = self.utils.get_random_betas_smal(num_coeffs)

        return params

    def get_key_name_for_model(self, with_face: bool = False) -> str:
        if self.model_type == "smplx":
            if with_face:
                return "expression"
            return "betas"
        elif self.model_type == "flame":
            if with_face:
                return "expression_params"
            return "shape_params"
        else:
            return "beta"


def plot_scatter_with_thumbnails(
    data_2d: np.ndarray,
    thumbnails: List[np.ndarray],
    labels: Optional[List[np.ndarray]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (1200, 1200),
    mark_size: int = 40,
):
    """
    Plot an interactive scatter plot with the provided thumbnails as tooltips.
    Args:
    - data_2d: 2D array of shape (n_samples, 2) containing the 2D coordinates of the data points.
    - thumbnails: List of thumbnails to be displayed as tooltips, each thumbnail should be a numpy array.
    - labels: List of labels to be used for coloring the data points, if None, no coloring is applied.
    - title: Title of the plot.
    - figsize: Size of the plot.
    - mark_size: Size of the data points.
    Returns:
    - Altair chart object.
    """

    def _return_thumbnail(img_array, size=100):
        """Return a thumbnail of the image array."""
        image = Image.fromarray(img_array)
        image.thumbnail((size, size), Image.ANTIALIAS)
        return image

    def _image_formatter(img):
        """Return a base64 encoded image."""
        with BytesIO() as buffer:
            img.save(buffer, "png")
            data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{data}"

    dataframe = pd.DataFrame(
        {
            "x": data_2d[:, 0],
            "y": data_2d[:, 1],
            "image": [_image_formatter(_return_thumbnail(thumbnail)) for thumbnail in thumbnails],
            "label": labels,
        }
    )

    chart = (
        alt.Chart(dataframe, title=title)
        .mark_circle(size=mark_size)
        .encode(
            x="x", y=alt.Y("y", axis=None), tooltip=["image"], color="label"
        )  # Must be a list for the image to render
        .properties(width=figsize[0], height=figsize[1])
        .configure_axis(grid=False)
        .configure_legend(orient="top", titleFontSize=20, labelFontSize=10, labelLimit=0)
    )

    if labels is not None:
        chart = chart.encode(color="label:N")

    return chart.display()


class Image2ShapeUtils:
    def __init__(self):
        pass

    @staticmethod
    def _gender_decider(arg: str) -> Literal["male", "female", "neutral"]:
        possible_gender = arg.split("smplx_")[1]
        assert possible_gender in ["male", "female", "neutral"], f"{possible_gender} is not supported"
        return possible_gender

    def _load_renderer(self, kwargs: Union[DictConfig, Dict[str, Any]]):
        self.renderer: Pytorch3dRenderer = Pytorch3dRenderer(**kwargs)

    def _load_body_pose(self, body_pose_path: str):
        self.body_pose: torch.Tensor = torch.from_numpy(np.load(body_pose_path))

    def _load_smplx_models(self, smplx_models_paths: Dict[str, str]):
        self.model: Dict[str, nn.Module] = {}
        self.labels: Dict[str, List[str]] = {}
        for model_name, model_path in smplx_models_paths.items():
            model, labels = self.utils.get_model_to_eval(model_path)
            labels = self._flatten_list_of_lists(labels)
            gender = self._gender_decider(model_name)
            self.model[gender] = model
            self.labels[gender] = labels

    def _load_flame_smal_models(self, model_path: str):
        self.model, labels = self.utils.get_model_to_eval(model_path)
        self.labels = self._flatten_list_of_lists(labels)

    def _from_h5_to_img(
        self, h5_file_path: Union[str, Path], gender: Literal["male", "female", "neutral"], renderer: Pytorch3dRenderer
    ) -> Tuple[np.ndarray, torch.Tensor]:
        data = h5py.File(h5_file_path, "r")
        shape_vector = torch.tensor(data["betas"])[None].float()
        render_mesh_kwargs = self.get_render_mesh_kwargs(shape_vector, gender=gender)
        rendered_img = renderer.render_mesh(**render_mesh_kwargs)
        rendered_img = self.adjust_rendered_img(rendered_img)
        return rendered_img, shape_vector

    def _load_comparison_data(self, path: Union[str, Path]):
        self.comparison_data: torch.Tensor = torch.from_numpy(np.load(path))

    def _load_images_generator(self):
        self.images_generator: List[Path] = sorted(list(self.data_dir.rglob(f"*.{self.suffix}")))

    def _load_clip_model(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def _encode_labels(self):
        if isinstance(self.labels, dict):
            self.encoded_labels: Dict[str, torch.Tensor] = {
                gender: clip.tokenize(self.labels[gender]).to(self.device) for gender in self.labels.keys()
            }
        else:
            self.encoded_labels: torch.Tensor = clip.tokenize(self.labels).to(self.device)

    @staticmethod
    def _flatten_list_of_lists(list_of_lists: List[List[str]]) -> List[str]:
        return [item for sublist in list_of_lists for item in sublist]

    def _get_smplx_attributes(
        self,
        betas: torch.Tensor,
        gender: Literal["male", "female", "neutral"],
        get_smpl: bool = False,
        body_pose=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        betas = betas.cpu()
        if hasattr(self, "body_pose"):
            body_pose = self.body_pose
        verts, faces, vt, ft = self.utils.get_smplx_model(
            betas=betas, gender=gender, body_pose=body_pose, get_smpl=get_smpl
        )
        if get_smpl:
            verts += self.utils.smpl_offset_numpy
        else:
            verts += self.utils.smplx_offset_numpy
        return verts, faces, vt, ft

    def _get_flame_attributes(
        self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.with_face:
            flame_out = self.utils.get_flame_model(expression_params=pred_vec.cpu(), gender=gender)
        else:
            flame_out = self.utils.get_flame_model(shape_params=pred_vec.cpu(), gender=gender)
        return flame_out

    def _get_smal_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, None, None]:
        smal_out = self.utils.get_smal_model(beta=pred_vec.cpu())
        return smal_out

    def get_render_mesh_kwargs(
        self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"], get_smpl: bool = False
    ) -> Dict[str, np.ndarray]:
        if self.model_type == "smplx" or self.model_type == "smpl":
            out = self._get_smplx_attributes(pred_vec=pred_vec, gender=gender, get_smpl=get_smpl)
        elif self.model_type == "flame":
            out = self._get_flame_attributes(pred_vec=pred_vec)
        elif self.model_type == "smal":
            out = self._get_smal_attributes(pred_vec=pred_vec)

        kwargs = {"verts": out[0], "faces": out[1], "vt": out[2], "ft": out[3]}

        return kwargs

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor) -> np.ndarray:
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def create_video_from_dir(self, dir_path: Union[str, Path], image_shape: Tuple[int, int]):
        dir_path = Path(dir_path)
        out_vid_path = dir_path.parent / "out_vid.mp4"
        out_vid = cv2.VideoWriter(out_vid_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), 30, image_shape[::-1])
        sorted_frames = sorted(dir_path.iterdir(), key=lambda x: int(x.stem))
        for frame in tqdm(sorted_frames, desc="Creating video", total=len(sorted_frames)):
            out_vid.write(cv2.imread(frame.as_posix()))
        out_vid.release()

    def _save_images_collage(self, images: List[np.ndarray]):
        collage_shape = self.utils.get_plot_shape(len(images))[0]
        images_collage = []
        for i in range(collage_shape[0]):
            images_collage.append(np.hstack(images[i * collage_shape[1] : (i + 1) * collage_shape[1]]))
        images_collage = np.vstack([image for image in images_collage])
        cv2.imwrite(self.images_out_path.as_posix(), images_collage)
        self.num_img += 1
        self.images_out_path = self.images_out_path.parent / f"{self.num_img}.png"

    @staticmethod
    def mesh_attributes_to_kwargs(
        attributes: Union[
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
        to_tensor: bool = False,
    ) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        if isinstance(attributes, dict):
            kwargs = {}
            for key, value in attributes.items():
                kwargs[key] = {"verts": value[0], "faces": value[1], "vt": value[2], "ft": value[3]}
                if to_tensor:
                    kwargs[key] = {k: torch.tensor(v)[None] for k, v in kwargs[key].items()}
        else:
            kwargs = {"verts": attributes[0], "faces": attributes[1], "vt": attributes[2], "ft": attributes[3]}
            if to_tensor:
                kwargs = {k: torch.tensor(v)[None] for k, v in kwargs.items()}
        return kwargs


class MaxCoeffs(Enum):
    """Enum for max coeffs for each model type."""

    SMPLX = 100
    SMPL = 100
    FLAME_SHAPE = 100
    FLAME_EXPRESSION = 50
    SMAL = 41
    JAW_POSE = 3


class VertsIdx(Enum):

    TOP_LIP_MIN = 3531
    TOP_LIP_MAX = 3532
    BOTTOM_LIP_MIN = 3504
    BOTTOM_LIP_MAX = 3505
