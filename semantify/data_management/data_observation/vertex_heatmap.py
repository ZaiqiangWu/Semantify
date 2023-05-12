import cv2
import torch
import hydra
import shutil
import trimesh
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from itertools import combinations
from typing import List, Literal, Tuple
from clip2mesh.utils import Utils, Pytorch3dRenderer, ModelsFactory


class VertexHeatmap:
    def __init__(
        self,
        descriptors_dir: str,
        iou_threshold: float,
        corr_threshold: float,
        model_type: Literal["flame", "smplx", "smpl", "smal"],
        compare_to_default_mesh: bool,
        gender: Literal["male", "female", "neutral"],
        method: Literal["L2", "diff_coords"],
        effect_threshold: float,
        optimize_feature: Literal["betas", "shape_params", "expression_params"],
        renderer_kwargs: DictConfig,
        color_map: str = "YlOrRd",
        save_color_bar: bool = False,
    ):

        self.utils: Utils = Utils()
        self.descriptors_dir: Path = Path(descriptors_dir)
        self.iou_threshold = iou_threshold
        self.corr_threshold = corr_threshold
        self.model_type = model_type
        self.compare_to_default_mesh = compare_to_default_mesh
        self.gender = gender
        self.method = method
        self.effect_threshold = effect_threshold
        self.optimize_feature = optimize_feature
        self.color_map = plt.get_cmap(color_map)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.view_angles = range(0, 360, 45)
        self.models_factory = ModelsFactory(model_type=model_type)
        self.save_color_bar = save_color_bar

        self.num_rows, self.num_cols = self.get_collage_shape()

        self._assertions()
        self._load_renderer(renderer_kwargs)
        self._load_def_mesh()
        self._initialize_df()

    def _load_renderer(self, kwargs: DictConfig):
        self.renderer = Pytorch3dRenderer(**kwargs)

    def _initialize_df(self):
        self.df = pd.DataFrame(columns=["descriptor", "effect", "effective_vertices"])

    def _assertions(self):
        assert self.method in ["L2", "diff_coords"], "method must be either L2 or diff_coords"
        assert self.model_type in [
            "flame",
            "smplx",
            "smpl",
            "smal",
        ], "model_type must be either flame, smplx, smpl or smal"
        assert self.optimize_feature in [
            "betas",
            "shape_params",
            "expression_params",
        ], "optimize_feature must be either betas, shape_params or expression_params"
        assert self.gender in ["male", "female", "neutral"], "gender must be either male, female or neutral"

    def _load_def_mesh(self):
        verts, faces, vt, ft = self.models_factory.get_model(gender=self.gender)
        if self.model_type == "smplx":
            verts += self.utils.smplx_offset_numpy
        self.def_verts = torch.tensor(verts).to(self.device)
        if self.def_verts.dim() == 2:
            self.def_verts = self.def_verts.unsqueeze(0)
        self.def_faces = torch.tensor(faces).to(self.device)
        if self.def_faces.dim() == 2:
            self.def_faces = self.def_faces.unsqueeze(0)
        self.def_vt = torch.tensor(vt).to(self.device)
        self.def_ft = torch.tensor(ft).to(self.device)
        if self.method == "diff_coords":
            if verts.shape.__len__() == 3:
                verts = verts[0]
            if faces.shape.__len__() == 3:
                faces = faces[0]
            self.def_diff_coords = self.get_verts_diff_coords(
                verts, trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            )

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def get_collage_shape(self):
        num_rows, num_cols = self.utils.get_plot_shape(len(self.view_angles))[0]
        if num_rows > num_cols:
            return num_cols, num_rows
        return num_rows, num_cols

    def get_collage(self, images_list: List[np.ndarray]) -> np.ndarray:
        imgs_collage = [cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR) for rend_img in images_list]
        collage = np.concatenate(
            [
                np.concatenate(imgs_collage[i * self.num_cols : (i + 1) * self.num_cols], axis=1)
                for i in range(self.num_rows)
            ],
            axis=0,
        )
        return collage

    def save_color_bar_with_threshold(self, path):
        scale = np.linspace(0, 1, 11)
        fig = plt.figure(figsize=(10, 1))
        plt.imshow([scale], cmap=self.color_map, aspect=0.1, extent=[0, 1, 0, 1])
        plt.axvline(x=self.effect_threshold, color="black", linewidth=1)
        plt.axis("off")
        # annotate threshold
        plt.annotate(
            f"threshold: {self.effect_threshold}",
            xy=(self.effect_threshold, 0.5),
            xytext=(self.effect_threshold + 0.1, 0.5),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )
        fig.savefig(path)

    def get_model(self, shape_vec):
        model_kwargs = {self.optimize_feature: shape_vec, "gender": self.gender}
        verts, faces, _, _ = self.models_factory.get_model(**model_kwargs)
        if self.model_type == "smplx":
            verts += self.utils.smplx_offset_numpy
        return verts, faces

    def get_verts_diff_coords(self, verts: np.ndarray, mesh: trimesh.Trimesh) -> List[torch.Tensor]:
        g = nx.from_edgelist(mesh.edges_unique)
        one_ring = [list(g[i].keys()) for i in range(len(mesh.vertices))]
        verts_diff_coords = np.zeros_like(verts)
        for i, v in enumerate(verts):
            verts_diff_coords[i] = v - verts[one_ring[i]].mean(axis=0)
        return verts_diff_coords

    def create_ious_csv(self, path):
        df = self.df
        df["effective_vertices"] = df["effective_vertices"].apply(lambda x: np.array(x))
        df["vertex_coverage"] = df["effective_vertices"].apply(lambda x: x.shape[0])
        combs = list(combinations(df["descriptor"].values, 2))
        ious_df = pd.DataFrame()
        for comb in combs:
            comb_df = df[df["descriptor"].isin(comb)]
            comb_df = comb_df.sort_values("descriptor")
            iou = len(
                np.intersect1d(comb_df["effective_vertices"].values[0], comb_df["effective_vertices"].values[1])
            ) / len(np.union1d(comb_df["effective_vertices"].values[0], comb_df["effective_vertices"].values[1]))
            ious_df = pd.concat([ious_df, pd.DataFrame((comb[0], comb[1], iou)).T])

        columns = ("descriptor_1", "descriptor_2", "iou")
        ious_df.columns = columns
        ious_df = ious_df.sort_values("iou", ascending=False)
        ious_df.to_csv(path, index=False)
        df.to_csv(path.as_posix().replace("ious.csv", "summary.csv"), index=False)

    def get_verts_faces_by_model_type(self, verts, faces):
        if self.model_type in ["smpl", "smplx"]:
            return verts, faces
        return verts[0], faces

    def get_diffs(
        self, regular_features: Tuple[np.ndarray, np.ndarray], to_compare_features: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        verts_regular, faces_regular = self.get_verts_faces_by_model_type(*regular_features)

        if self.method == "L2":
            diffs = np.linalg.norm(verts_regular - to_compare_features[0], axis=-1)
            if diffs.ndim == 2:
                diffs = diffs.squeeze()
        else:
            diff_coords_regular = self.get_verts_diff_coords(
                verts_regular, trimesh.Trimesh(vertices=verts_regular, faces=faces_regular, process=False)
            )
            if not self.compare_to_default_mesh:
                to_compare_features = self.get_verts_faces_by_model_type(*to_compare_features)
                diff_coords_to_compare = self.get_verts_diff_coords(
                    to_compare_features[0],
                    trimesh.Trimesh(vertices=to_compare_features[0], faces=to_compare_features[1], process=False),
                )
                diffs = np.linalg.norm(diff_coords_regular - diff_coords_to_compare, axis=-1)
                return diffs

            diffs = np.linalg.norm(self.def_diff_coords - diff_coords_regular, axis=-1)
        return diffs

    def __call__(self):
        descriptors_generator = list(self.descriptors_dir.iterdir())
        progress_bar = tqdm(descriptors_generator, desc="descriptors", total=len(descriptors_generator))
        for descriptor in progress_bar:
            progress_bar.set_description(f"descriptor: {descriptor.name}")
            regular = torch.tensor(np.load(descriptor / f"{descriptor.name}.npy")).float()
            if regular.dim() == 1:
                regular = regular.unsqueeze(0)
            regular_verts, regular_faces = self.get_model(regular)

            if self.compare_to_default_mesh:
                to_compare_verts, to_compare_faces = self.def_verts.cpu().numpy(), self.def_faces.cpu().numpy()
                if to_compare_verts.ndim == 2:
                    to_compare_verts = to_compare_verts.unsqueeze(0)
                if to_compare_faces.ndim == 2:
                    to_compare_faces = to_compare_faces.unsqueeze(0)
            else:
                inverse = torch.tensor(np.load(descriptor / f"{descriptor.name}_inverse.npy")).float()
                if inverse.dim() == 1:
                    inverse = inverse.unsqueeze(0)
                to_compare_verts, to_compare_faces = self.get_model(inverse)

            diffs = self.get_diffs((regular_verts, regular_faces), (to_compare_verts, to_compare_faces))

            diffs = diffs / diffs.max()

            vertex_colors = self.color_map(diffs)[:, :3]
            effective_indices = np.where(diffs > self.effect_threshold)[0]

            vertex_colors = torch.tensor(vertex_colors).float().to(self.device)

            if vertex_colors.dim() == 2:
                vertex_colors = vertex_colors.unsqueeze(0)

            rend_imgs = []
            angles_dir = descriptor / "angles"
            angles_dir.mkdir(exist_ok=True)
            for angle in self.view_angles:
                rend_img = self.renderer.render_mesh(
                    self.def_verts,
                    self.def_faces,
                    self.def_vt,
                    self.def_ft,
                    texture_color_values=vertex_colors,
                    rotate_mesh={"degrees": float(angle), "axis": "y"},
                )
                rend_img = self.adjust_rendered_img(rend_img)
                rend_img = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(angles_dir / f"{angle}.png"), cv2.resize(rend_img, (512, 512)))
                rend_imgs.append(rend_img)
            collage = self.get_collage(rend_imgs)
            collage = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(descriptor / f"vertex_heatmap.png"), collage)
            temp_df = pd.DataFrame((descriptor.name, diffs.sum(), effective_indices.tolist())).T
            temp_df.columns = self.df.columns
            self.df = pd.concat([self.df, temp_df])

        vertex_heatmaps_dir = self.descriptors_dir / "vertex_heatmaps"
        vertex_heatmaps_dir.mkdir(exist_ok=True)
        self.create_ious_csv(vertex_heatmaps_dir / "ious.csv")
        for vertex_heatmap_file in self.descriptors_dir.rglob("vertex_heatmap.png"):
            shutil.copy(vertex_heatmap_file, vertex_heatmaps_dir / (vertex_heatmap_file.parent.name + ".png"))
        if self.save_color_bar:
            self.save_color_bar_with_threshold(vertex_heatmaps_dir / "color_bar_w_threshold.png")


@hydra.main(config_path="../../config", config_name="vertex_heatmap")
def main(cfg: DictConfig):
    vertex_heatmap = VertexHeatmap(**cfg)
    vertex_heatmap()


if __name__ == "__main__":
    main()
