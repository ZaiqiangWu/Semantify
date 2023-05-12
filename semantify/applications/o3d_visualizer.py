import argparse
from clip2mesh.utils import ModelsFactory


def main(args):

    models_factory = ModelsFactory(args.model_type)

    model_kwargs = models_factory.get_default_params()

    verts, faces, vt, ft = models_factory.get_model(**model_kwargs)

    renderer_kwargs = {
        "verts": verts,
        "faces": faces,
        "vt": vt,
        "ft": ft,
        "texture": args.texture_path,
        "for_image": args.for_image,
        "paint_vertex_colors": True if args.model_type == "smal" else False,
    }

    open3d_renderer = models_factory.get_renderer(**renderer_kwargs)
    open3d_renderer.render_mesh()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--for_image", action="store_true")
    parser.add_argument("--model_type", type=str, default="smal")
    parser.add_argument("--texture_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
