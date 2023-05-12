import torch
import hydra
import tkinter
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from clip2mesh.utils import Utils, ModelsFactory


class Text2MeshApp:
    def __init__(self, cfg):

        self.device = cfg.device
        self.texture = cfg.texture
        self.data_dir = cfg.data_dir

        assert cfg.model_type in ["smplx", "flame", "smal"], "Model type should be smplx, flame or smal"
        self.model_type = cfg.model_type

        if cfg.out_dir is not None:
            try:
                img_id = int(sorted(list(Path(cfg.out_dir).glob("*.png")), key=lambda x: int(x.stem))[-1].stem) + 1
            except IndexError:
                img_id = 0
            self.outpath = Path(cfg.out_dir) / f"{img_id}.png"

        self.utils = Utils()
        self.models_factory = ModelsFactory(self.model_type)
        self.gender = cfg.gender
        self.with_face = cfg.with_face
        self.model_kwargs = self.models_factory.get_default_params(cfg.with_face)
        verts, faces, vt, ft = self.models_factory.get_model(**self.model_kwargs)
        default_renderer_kwargs = self.get_renderer_kwargs(verts, faces, vt, ft)
        self.renderer = self.models_factory.get_renderer(**default_renderer_kwargs)

        self.default_zoom_value = 0.7
        if self.model_type == "smal":
            self.default_zoom_value = 0.1

        self.renderer.render_mesh()

        assert cfg.model_path is not None, "Please provide a path to a pretrained model"
        self.text_model = self.load_text_model()
        self.model, labels = self.utils.get_model_to_eval(cfg.model_path)

        self.labels = self.utils.flatten_list_of_lists(labels)
        values = self.get_mean_values()
        self.encoded_labels = self.encode_labels(self.labels)
        self.mean_values = {label: values[label][-1] for label in self.labels}
        self.input_for_model = torch.tensor(list(self.mean_values.values()), dtype=torch.float32)[None]

    def get_renderer_kwargs(self, verts, faces, vt, ft) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        return {
            "verts": verts,
            "faces": faces,
            "vt": vt,
            "ft": ft,
            "texture": self.texture,
            "paint_vertex_colors": True if self.model_type == "smal" else False,
        }

    def load_text_model(self):
        return SentenceTransformer("all-MiniLM-L6-v2")

    def encode_labels(self, labels):
        labels = self.text_model.encode(labels, convert_to_tensor=True)
        return labels

    def get_mean_values(self) -> Dict[str, Tuple[float, float, float]]:
        return self.utils.get_min_max_values(self.data_dir)

    @staticmethod
    def normalize_data(data):
        for key, value in data.items():
            min_val, max_val = 0.2, 0.7
            data[key] = (value - min_val) / (max_val - min_val)
        return data

    def refresh_open_3d(self):
        self.renderer.render_mesh()

    @staticmethod
    def inverse_normalize_data(data):
        for key, value in data.items():
            min_val, max_val = 0, 80
            data[key] = value * (max_val - min_val) + min_val
        return data

    def generate_mesh(self):
        text = self.text_entry.get("1.0", "end-1c")
        if text:
            text = [text]
            encoded_text = self.text_model.encode(text, convert_to_tensor=True)

            for idx, (label_name, labels_embedding) in enumerate(zip(self.labels, self.encoded_labels)):
                label_similarity = self.normalize_data(
                    {label_name: util.cos_sim(encoded_text, labels_embedding.unsqueeze(0)).item()}
                )
                label_similarity_inversed = self.inverse_normalize_data(label_similarity)
                self.input_for_model[0, idx] = label_similarity_inversed[label_name]

            with torch.no_grad():
                out = self.model(self.input_for_model.to(self.device))
                if self.model_type == "smplx":
                    betas = out.cpu()
                    expression = torch.zeros(1, 10)
                    body_pose = torch.eye(3).expand(1, 21, 3, 3)
                    verts, _, _, _ = self.utils.get_smplx_model(
                        betas=betas, body_pose=body_pose, expression=expression, gender=self.gender
                    )
                elif self.model_type == "flame":
                    if self.with_face:
                        verts, _, _, _ = self.utils.get_flame_model(
                            expression_params=out.cpu(), jaw_pose=torch.tensor([[0.08]])
                        )
                    else:
                        verts, _, _, _ = self.utils.get_flame_model(shape_params=out.cpu())

                else:
                    verts, _, _, _ = self.utils.get_smal_model(beta=out.cpu())

                self.renderer.render_mesh(verts=verts)

    def create_application(self):

        self.root = tkinter.Tk()
        self.root.title("Text 2 Mesh")
        self.root.geometry("300x300")
        button_canvas_coords = (30, 30)

        main_frame = tkinter.Frame(self.root, bg="white")
        main_frame.pack(fill=tkinter.BOTH, expand=True)
        text_frame = tkinter.Frame(self.root, highlightbackground="white", highlightthickness=0, bg="white")
        text_canvas = tkinter.Canvas(main_frame, bg="white", highlightbackground="white")
        text_canvas.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)
        text_canvas.create_window(button_canvas_coords, window=text_frame, anchor=tkinter.NW)

        l = tkinter.Label(text_frame, text="Mesh Description", font=("Helvetica", 12))
        l.config(font=("Courier", 14))
        l.pack(side=tkinter.TOP, pady=10)

        self.text_entry = tkinter.Text(text_frame, font=("Helvetica", 12), height=5, width=20)
        self.text_entry.pack(side=tkinter.TOP, pady=10)

        generate_mesh_button = tkinter.Button(text_frame, text="Generate Mesh", command=lambda: self.generate_mesh())
        generate_mesh_button.pack(side=tkinter.TOP, pady=10)

        refresh_button = tkinter.Button(text_frame, text="Refresh", command=lambda: self.refresh_open_3d())
        refresh_button.pack(side=tkinter.TOP, pady=10)

        self.root.mainloop()


@hydra.main(config_path="config", config_name="text_to_mesh_demo")
def main(cfg):
    app = Text2MeshApp(cfg.demo_kwargs)
    app.create_application()


if __name__ == "__main__":
    main()
