import cv2
import clip
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.io import load_objs_as_meshes
from clip2mesh.utils import Pytorch3dRenderer
from clip2mesh.optimization.optimization import CLIPLoss

device = "cuda" if torch.cuda.is_available() else "cpu"


class OptimizeVertexColors(nn.Module):
    def __init__(self, params_size: Tuple[int, int]):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(params_size))

    def forward(self):
        return self.weights


initial_mesh = load_objs_as_meshes(["/home/nadav2/dev/data/CLIP2Shape/outs/objs/0.obj"]).to(device)
subdivide_mesh = SubdivideMeshes(initial_mesh)
initial_mesh = subdivide_mesh(initial_mesh)
initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)
# initial_mesh = subdivide_mesh(initial_mesh)

verts = initial_mesh.verts_packed().unsqueeze(0)
faces = initial_mesh.faces_packed().unsqueeze(0)
print("the number of vertices is:", verts.shape[1])
print("the number of faces is:", faces.shape[1])
frontal_renderer = Pytorch3dRenderer(
    tex_path=None, azim=2.1, elev=20.0, dist=4.0, texture_optimization=True, img_size=(224, 224)
)
left_renderer = Pytorch3dRenderer(
    tex_path=None, azim=-20.1, elev=20.0, dist=4.0, texture_optimization=True, img_size=(224, 224)
)
right_renderer = Pytorch3dRenderer(
    tex_path=None, azim=40.1, elev=20.0, dist=4.0, texture_optimization=True, img_size=(224, 224)
)
upper_renderer = Pytorch3dRenderer(
    tex_path=None, azim=2.1, elev=60.0, dist=4.0, texture_optimization=True, img_size=(224, 224)
)
lower_renderer = Pytorch3dRenderer(
    tex_path=None, azim=2.1, elev=-20.0, dist=4.0, texture_optimization=True, img_size=(224, 224)
)


clip_model, image_encoder = clip.load("ViT-B/32", device=device)
encoded_text = clip.tokenize("cow").to(device)
loss_fn = CLIPLoss()

model = OptimizeVertexColors(params_size=(initial_mesh.verts_packed().shape[0], 3)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

progress_bar = tqdm(range(3000))

for i in progress_bar:

    # Forward pass: render the image
    verts_rgb = model().unsqueeze(0)

    # render the image from all 3 views
    front_image = frontal_renderer.render_mesh(verts=verts, faces=faces, texture_color_values=verts_rgb)
    left_image = left_renderer.render_mesh(verts=verts, faces=faces, texture_color_values=verts_rgb)
    right_image = right_renderer.render_mesh(verts=verts, faces=faces, texture_color_values=verts_rgb)
    upper_image = upper_renderer.render_mesh(verts=verts, faces=faces, texture_color_values=verts_rgb)
    lower_image = lower_renderer.render_mesh(verts=verts, faces=faces, texture_color_values=verts_rgb)
    # # image = renderer.render_mesh(verts=verts, faces=faces, texture_color_values=verts_rgb)

    # Forward pass: compute the loss
    loss = loss_fn(front_image[..., :3].permute(0, 3, 1, 2), encoded_text)
    loss += loss_fn(left_image[..., :3].permute(0, 3, 1, 2), encoded_text)
    loss += loss_fn(right_image[..., :3].permute(0, 3, 1, 2), encoded_text)
    loss += loss_fn(upper_image[..., :3].permute(0, 3, 1, 2), encoded_text)
    loss += loss_fn(lower_image[..., :3].permute(0, 3, 1, 2), encoded_text)

    # Backward pass: compute the gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # Manually zero the gradients after running the backward pass
    optimizer.zero_grad()

    progress_bar.desc = f"Loss: {loss.item():.4f}"

    if i % 10 == 0:

        # display the image
        image = front_image.squeeze().detach().cpu().numpy()[..., :3]
        image = (image * 255).astype(np.uint8)
        # resize the image to 512x512
        image = cv2.resize(image, (512, 512))
        cv2.imshow("image", image)
        cv2.waitKey(1)
