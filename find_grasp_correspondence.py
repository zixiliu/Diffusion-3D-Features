import torch
from diff3f import get_features_per_vertex
from time import time
# Handle both package import and direct script execution
try:
    from .utils import convert_mesh_container_to_torch_mesh, cosine_similarity, double_plot, get_colors, generate_colors
    from .dataloaders.mesh_container import MeshContainer
    from .diffusion import init_pipe
    from .dino import init_dino
    from .functional_map import compute_surface_map
    from .point_to_mesh_similarity import run_point_similarity_analysis, point_similarity_colormap, visualize_point_similarity, run_multi_point_correspondence_analysis
    from .utils import remesh_mesh_pair
except ImportError:
    from utils import convert_mesh_container_to_torch_mesh, cosine_similarity, double_plot, get_colors, generate_colors
    from dataloaders.mesh_container import MeshContainer
    from diffusion import init_pipe
    from dino import init_dino
    from functional_map import compute_surface_map
    from point_to_mesh_similarity import run_point_similarity_analysis, point_similarity_colormap, visualize_point_similarity, run_multi_point_correspondence_analysis
    from utils import remesh_mesh_pair

import importlib
import meshplot as mp
import numpy as np
import open3d


def main(source_mesh_path, target_mesh_path, source_points_3d, remesh=True, source_prompt="beaker", target_prompt="bottle", num_views = 8, meshplot_browser=True):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing diffusion pipeline...")
    pipe = init_pipe(device)

    print("Initializing DINO model...")
    dino_model = init_dino(device)

    print("Models initialized successfully!")


    if remesh:
        source_mesh_path, target_mesh_path = remesh_mesh_pair(
            source_mesh_path,
            target_mesh_path,
            num_points=10000,
            alpha=0.005
        )
    source_mesh = MeshContainer().load_from_file(source_mesh_path)
    target_mesh = MeshContainer().load_from_file(target_mesh_path)

    print(f"Source mesh: {len(source_mesh.vert)} vertices, {len(source_mesh.face)} faces")
    print(f"Target mesh: {len(target_mesh.vert)} vertices, {len(target_mesh.face)} faces")

    # Run the complete analysis pipeline
    similarity_colors, raw_similarities, source_point_idx, closest_distance, target_points_3d = run_multi_point_correspondence_analysis(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_points_3d=source_points_3d,
        device=device,
        pipe=pipe,
        dino_model=dino_model,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        num_views=num_views,
        meshplot_browser=meshplot_browser
    )

    print("\n➡️ target_points_3d = \n", target_points_3d)
    print("\n✅ Analysis complete!")

    return target_points_3d

if __name__ == "__main__":

    # example source points 3d
    p1 = [ 0.00886742, -0.04124315, -0.01754964]
    p2 = [-0.01223545,  0.0363251 ,  0.00719686]
    p3 = [-0.01694481,  0.03424783, -0.01735426]
    p4 = [-0.01596961,  0.03464369, -0.02075735]
    source_points_3d = np.array([p1, p2, p3, p4])

    main(source_mesh_path="meshes/oakink_beaker_decomp2.obj", target_mesh_path="meshes/oakink_bowl_decomp2.obj", source_points_3d=source_points_3d, remesh=True, source_prompt="beaker", target_prompt="bowl", num_views=4)
