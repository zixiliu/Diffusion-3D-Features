from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
from .normal_shading import HardPhongNormalShader
import torch
import math
import time


@torch.no_grad()
def run_rendering(device, mesh, mesh_vertices, num_views, H, W, add_angle_azi=0, add_angle_ele=0, use_normal_map=False):
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor

    # Generate exactly num_views camera angles
    if num_views == 1:
        elevation = torch.tensor([0.0]) + add_angle_ele
        azimuth = torch.tensor([0.0]) + add_angle_azi
    else:
        # For multiple views, distribute them evenly around the sphere
        # Use a spiral pattern or uniform distribution
        azimuth = torch.linspace(0, 360 * (1 - 1/num_views), num_views) + add_angle_azi

        # For elevation, use a pattern that covers different heights
        if num_views <= 4:
            elevation = torch.linspace(0, 45, num_views) + add_angle_ele
        else:
            # Create a more complex pattern for better coverage
            elevation = torch.zeros(num_views)
            for i in range(num_views):
                # Alternate between different elevation angles
                elevation[i] = (i % 3) * 30  # 0, 30, 60 degrees pattern
            elevation = elevation + add_angle_ele

    # Ensure bbox_center has the right shape: flatten it first, then expand to match batch size
    batch_size = azimuth.shape[0]
    bbox_center = bbox_center.flatten()  # Ensure it's 1D: [3]
    bbox_center = bbox_center.unsqueeze(0).expand(batch_size, -1)  # Now [batch_size, 3]

    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    rasterization_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_centre,
        device=device,
    )
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    batch_mesh = mesh.extend(num_views)
    normal_batched_renderings = None
    batched_renderings = batch_renderer(batch_mesh)
    if use_normal_map:
        normal_shader = HardPhongNormalShader(device=device, cameras=camera, lights=lights)
        normal_batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=normal_shader)
        normal_batched_renderings = normal_batch_renderer(batch_mesh)
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf
    return batched_renderings, normal_batched_renderings, camera, depth


def batch_render(device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    while trials < 5:
        try:
            return run_rendering(device, mesh, mesh_vertices, num_views, H, W, add_angle_azi=add_angle_azi, add_angle_ele=add_angle_ele, use_normal_map=use_normal_map)
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = torch.randn(1)
            add_angle_ele = torch.randn(1)
            continue
