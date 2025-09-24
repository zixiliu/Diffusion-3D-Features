import torch
import numpy as np
from diff3f import get_features_per_vertex
from utils import convert_mesh_container_to_torch_mesh, cosine_similarity
import meshplot as mp

def find_closest_vertex(point_3d, mesh_vertices):
    """
    Find the closest vertex to a given 3D point.

    Args:
        point_3d: 3D coordinates as array-like (x, y, z) or torch tensor
        mesh_vertices: mesh vertices as numpy array or torch tensor (N, 3)

    Returns:
        closest_vertex_idx: index of the closest vertex
        closest_distance: distance to the closest vertex
    """
    # Convert to numpy arrays for consistent processing
    if isinstance(point_3d, torch.Tensor):
        point_3d = point_3d.cpu().numpy()
    if isinstance(mesh_vertices, torch.Tensor):
        mesh_vertices = mesh_vertices.cpu().numpy()

    point_3d = np.array(point_3d).reshape(1, 3)  # Ensure shape (1, 3)

    # Compute distances to all vertices
    distances = np.linalg.norm(mesh_vertices - point_3d, axis=1)

    # Find closest vertex
    closest_vertex_idx = np.argmin(distances)
    closest_distance = distances[closest_vertex_idx]

    return int(closest_vertex_idx), float(closest_distance)

def point_similarity_colormap(
    device,
    pipe,
    dino_model,
    source_mesh,
    target_mesh,
    source_point_3d,
    prompt,
    num_views=100,
    **kwargs
):
    """
    Color-map target mesh based on similarity to a specific 3D point on source mesh.

    Args:
        device: torch device
        pipe: diffusion pipeline
        dino_model: DINO model
        source_mesh: source mesh container
        target_mesh: target mesh container
        source_point_3d: 3D coordinates (x, y, z) as array-like or torch tensor
        prompt: text prompt for feature extraction
        num_views: number of views for rendering
        **kwargs: additional arguments for get_features_per_vertex

    Returns:
        similarity_colors: color values for target mesh vertices based on similarity
        raw_similarities: raw similarity scores
        source_point_idx: index of the closest vertex to the input 3D point
        closest_distance: distance from input point to closest vertex
    """

    # Find the closest vertex to the input 3D point
    source_point_idx, closest_distance = find_closest_vertex(source_point_3d, source_mesh.vert)

    print(f"Input 3D point: {np.array(source_point_3d)}")
    print(f"Closest vertex: {source_point_idx} (distance: {closest_distance:.4f})")
    print(f"Closest vertex coordinates: {source_mesh.vert[source_point_idx]}")

    # Convert mesh containers to torch meshes
    source_torch_mesh = convert_mesh_container_to_torch_mesh(source_mesh, device)
    target_torch_mesh = convert_mesh_container_to_torch_mesh(target_mesh, device)

    print("Computing features for source mesh...")
    source_features = get_features_per_vertex(
        device, pipe, dino_model, source_torch_mesh, prompt, num_views, **kwargs
    )

    print("Computing features for target mesh...")
    target_features = get_features_per_vertex(
        device, pipe, dino_model, target_torch_mesh, prompt, num_views, **kwargs
    )

    # Get the feature vector for the specific source point
    source_point_feature = source_features[source_point_idx:source_point_idx+1]  # Keep as 2D tensor

    # Compute cosine similarity between the source point and all target vertices
    similarities = cosine_similarity(source_point_feature, target_features)
    similarity_scores = similarities[0].cpu().numpy()  # Shape: (num_target_vertices,)

    # Normalize similarity scores to [0, 1] for color mapping
    min_sim = similarity_scores.min()
    max_sim = similarity_scores.max()
    normalized_similarities = (similarity_scores - min_sim) / (max_sim - min_sim)

    return normalized_similarities, similarity_scores, source_point_idx, closest_distance

def visualize_point_similarity(
    source_mesh,
    target_mesh,
    source_point_idx,
    similarity_colors,
    colormap='viridis'
):
    """
    Visualize the source mesh with highlighted point and target mesh with similarity colors.

    Args:
        source_mesh: source mesh container
        target_mesh: target mesh container
        source_point_idx: index of the specific point on source mesh
        similarity_colors: normalized similarity values for coloring target mesh
        colormap: matplotlib colormap name
    """

    # Create colors for source mesh - highlight the specific point
    source_colors = np.ones((len(source_mesh.vert), 3)) * 0.7  # Gray color
    source_colors[source_point_idx] = [1.0, 0.0, 0.0]  # Red for the specific point

    # Create subplot with source mesh (left) and target mesh (right) with similarity colors
    d = mp.subplot(source_mesh.vert, source_mesh.face, c=source_colors, s=[2, 2, 0])
    mp.subplot(target_mesh.vert, target_mesh.face, c=similarity_colors, s=[2, 2, 1], data=d, colormap=colormap)

    print(f"Source point (red): vertex {source_point_idx}")
    print(f"Target mesh colored by similarity to source point")
    print(f"Similarity range: {similarity_colors.min():.3f} to {similarity_colors.max():.3f}")

# Example usage function
def run_point_similarity_analysis(
    source_mesh,
    target_mesh,
    source_point_3d,
    device,
    pipe,
    dino_model,
    prompt="a textured 3D model",
    num_views=50
):
    """
    Complete pipeline to analyze and visualize point-to-mesh similarity using 3D coordinates.

    Args:
        source_mesh: source mesh container
        target_mesh: target mesh container
        source_point_3d: 3D coordinates (x, y, z) as array-like or torch tensor
        device: torch device
        pipe: diffusion pipeline
        dino_model: DINO model
        prompt: text prompt for feature extraction
        num_views: number of views for rendering

    Returns:
        similarity_colors: normalized similarity colors for target mesh
        raw_similarities: raw similarity scores
        source_point_idx: index of closest vertex to input 3D point
        closest_distance: distance from input point to closest vertex
    """

    # Compute similarity colors
    similarity_colors, raw_similarities, source_point_idx, closest_distance = point_similarity_colormap(
        device=device,
        pipe=pipe,
        dino_model=dino_model,
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_point_3d=source_point_3d,
        prompt=prompt,
        num_views=num_views
    )

    # Visualize results
    visualize_point_similarity(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_point_idx=source_point_idx,
        similarity_colors=similarity_colors
    )

    return similarity_colors, raw_similarities, source_point_idx, closest_distance
