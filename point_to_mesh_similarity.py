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

def find_closest_vertices_batch(points_3d, mesh_vertices):
    """
    Find the closest vertices to multiple 3D points.

    Args:
        points_3d: 3D coordinates as array-like (N, 3) or torch tensor
        mesh_vertices: mesh vertices as numpy array or torch tensor (M, 3)

    Returns:
        closest_vertex_indices: array of indices of closest vertices (N,)
        closest_distances: array of distances to closest vertices (N,)
    """
    # Convert to numpy arrays for consistent processing
    if isinstance(points_3d, torch.Tensor):
        points_3d = points_3d.cpu().numpy()
    if isinstance(mesh_vertices, torch.Tensor):
        mesh_vertices = mesh_vertices.cpu().numpy()

    points_3d = np.array(points_3d)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, 3)  # Handle single point case

    # Compute distances from all points to all vertices
    distances = np.linalg.norm(mesh_vertices[np.newaxis, :, :] - points_3d[:, np.newaxis, :], axis=2)

    # Find closest vertices
    closest_vertex_indices = np.argmin(distances, axis=1)
    closest_distances = np.min(distances, axis=1)

    return closest_vertex_indices.astype(int), closest_distances

def generate_distinct_colors(n):
    """
    Generate n distinct colors for visualization.

    Args:
        n: number of colors to generate

    Returns:
        colors: array of RGB colors (n, 3)
    """
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8 + 0.2 * (i % 2)  # Alternate between high and very high saturation
        value = 0.9 + 0.1 * ((i + 1) % 2)  # Alternate between high and very high value
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return np.array(colors)

def create_sphere(center, radius=0.02, resolution=16):
    """
    Create a sphere mesh at given center with specified radius.

    Args:
        center: 3D coordinates (x, y, z) for sphere center
        radius: sphere radius
        resolution: sphere resolution (higher = smoother)

    Returns:
        vertices: sphere vertices (N, 3)
        faces: sphere faces (M, 3)
    """
    # Create sphere using parametric equations
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    # Convert to vertices
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Create faces for the sphere (more robust face generation)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Two triangles per quad
            v1 = i * resolution + j
            v2 = i * resolution + (j + 1)
            v3 = (i + 1) * resolution + j
            v4 = (i + 1) * resolution + (j + 1)

            # Ensure faces are properly oriented
            if v1 < len(vertices) and v2 < len(vertices) and v3 < len(vertices) and v4 < len(vertices):
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

    return vertices, np.array(faces, dtype=np.int32)

def calculate_mesh_scale(mesh_vertices):
    """
    Calculate appropriate sphere radius based on mesh size.

    Args:
        mesh_vertices: mesh vertices (N, 3)

    Returns:
        sphere_radius: appropriate radius for highlighting spheres
    """
    # Calculate bounding box
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    bbox_size = np.linalg.norm(bbox_max - bbox_min)

    # Set sphere radius as a fraction of the bounding box diagonal
    sphere_radius = bbox_size * 0.02  # 2% of mesh size

    return sphere_radius

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

def create_highlight_regions(mesh_vertices, mesh_faces, highlight_indices, colors, radius_scale=0.05):
    """
    Create enlarged regions around specific vertices for better visibility.

    Args:
        mesh_vertices: mesh vertices (N, 3)
        mesh_faces: mesh faces (M, 3)
        highlight_indices: vertex indices to highlight
        colors: colors for each highlight (len(highlight_indices), 3)
        radius_scale: scale factor for highlight region size

    Returns:
        highlight_vertices: vertices for highlight regions
        highlight_faces: faces for highlight regions
        highlight_colors: colors for highlight regions
    """
    if len(highlight_indices) == 0:
        return np.array([]), np.array([]), np.array([])

    # Calculate mesh scale for consistent sizing
    bbox_size = np.linalg.norm(np.max(mesh_vertices, axis=0) - np.min(mesh_vertices, axis=0))
    radius = bbox_size * radius_scale

    all_vertices = []
    all_faces = []
    all_colors = []

    for i, (vertex_idx, color) in enumerate(zip(highlight_indices, colors)):
        center = mesh_vertices[vertex_idx]

        # Create a simple octahedron for highlighting (6 vertices, 8 faces)
        # This is much simpler than a sphere and more reliable
        vertices = np.array([
            center + [radius, 0, 0],      # +X
            center + [-radius, 0, 0],     # -X
            center + [0, radius, 0],      # +Y
            center + [0, -radius, 0],     # -Y
            center + [0, 0, radius],      # +Z
            center + [0, 0, -radius]      # -Z
        ])

        # Define octahedron faces
        faces = np.array([
            [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],  # Right side
            [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]   # Left side
        ])

        # Offset face indices for this highlight
        faces = faces + len(all_vertices)

        # Colors for all vertices of this highlight
        vertex_colors = np.tile(color, (len(vertices), 1))

        all_vertices.extend(vertices)
        all_faces.extend(faces)
        all_colors.extend(vertex_colors)

    return np.array(all_vertices), np.array(all_faces), np.array(all_colors)

def visualize_point_similarity(
    source_mesh,
    target_mesh,
    source_point_idx,
    similarity_colors,
    raw_similarities=None,
    colormap='viridis'
):
    """
    Visualize the source mesh with highlighted point and target mesh with similarity colors.
    Uses robust octahedron highlighting instead of complex spheres.

    Args:
        source_mesh: source mesh container
        target_mesh: target mesh container
        source_point_idx: index of the specific point on source mesh
        similarity_colors: normalized similarity values for coloring target mesh
        raw_similarities: raw similarity scores (optional, for highlighting most similar vertex)
        colormap: matplotlib colormap name
    """
    import matplotlib.pyplot as plt

    # Convert similarity colors to RGB using the specified colormap
    cmap = plt.get_cmap(colormap)
    target_colors_rgb = cmap(similarity_colors)[:, :3]  # Get RGB, ignore alpha

    # Create base mesh visualization with similarity colors
    source_colors = np.ones((len(source_mesh.vert), 3)) * 0.7  # Gray color
    source_colors[source_point_idx] = [1.0, 0.0, 0.0]  # Red highlight for source point

    # Handle target mesh coloring
    most_similar_idx = None
    if raw_similarities is not None:
        most_similar_idx = np.argmax(raw_similarities)
        target_colors_rgb[most_similar_idx] = [1.0, 0.0, 0.0]  # Red highlight for most similar vertex

    # Create base visualization
    d = mp.subplot(source_mesh.vert, source_mesh.face, c=source_colors, s=[2, 2, 0])
    mp.subplot(target_mesh.vert, target_mesh.face, c=target_colors_rgb, s=[2, 2, 1], data=d)

    # Add enhanced highlighting with octahedrons (more reliable than spheres)
    try:
        # Source highlight
        source_highlight_verts, source_highlight_faces, source_highlight_colors = create_highlight_regions(
            source_mesh.vert, source_mesh.face, [source_point_idx], [[1.0, 0.0, 0.0]]
        )

        if len(source_highlight_verts) > 0:
            mp.subplot(source_highlight_verts, source_highlight_faces, c=source_highlight_colors, s=[2, 2, 0], data=d)

        # Target highlight (if most similar vertex found)
        if most_similar_idx is not None:
            target_highlight_verts, target_highlight_faces, target_highlight_colors = create_highlight_regions(
                target_mesh.vert, target_mesh.face, [most_similar_idx], [[1.0, 0.0, 0.0]]
            )

            if len(target_highlight_verts) > 0:
                mp.subplot(target_highlight_verts, target_highlight_faces, c=target_highlight_colors, s=[2, 2, 1], data=d)

        print(f"🎯 Enhanced visualization with 3D highlighting...")

    except Exception as e:
        print(f"ℹ️  Using vertex highlighting (3D markers unavailable: {str(e)})")

    # Print results
    if raw_similarities is not None:
        print(f"Source point (red): vertex {source_point_idx}")
        print(f"Target mesh: colored by similarity + MOST SIMILAR vertex {most_similar_idx} in RED")
        print(f"Most similar vertex similarity: {raw_similarities[most_similar_idx]:.4f}")
        print(f"Similarity range: {similarity_colors.min():.3f} to {similarity_colors.max():.3f}")
    else:
        print(f"Source point highlighted in red: vertex {source_point_idx}")
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

    # Visualize results with highlighting of most similar vertex
    visualize_point_similarity(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_point_idx=source_point_idx,
        similarity_colors=similarity_colors,
        raw_similarities=raw_similarities  # Pass raw similarities for highlighting
    )

    return similarity_colors, raw_similarities, source_point_idx, closest_distance

def multi_point_correspondence_analysis(
    device,
    pipe,
    dino_model,
    source_mesh,
    target_mesh,
    source_points_3d,
    prompt,
    num_views=100,
    **kwargs
):
    """
    Find correspondences for multiple 3D points and return color-coded results.

    Args:
        device: torch device
        pipe: diffusion pipeline
        dino_model: DINO model
        source_mesh: source mesh container
        target_mesh: target mesh container
        source_points_3d: 3D coordinates (N, 3) as array-like or torch tensor
        prompt: text prompt for feature extraction
        num_views: number of views for rendering
        **kwargs: additional arguments for get_features_per_vertex

    Returns:
        correspondences: list of (source_idx, target_idx, similarity_score) tuples
        correspondence_colors: RGB colors for each correspondence pair (N, 3)
        source_vertex_indices: indices of closest vertices to input 3D points (N,)
        closest_distances: distances from input points to closest vertices (N,)
    """

    # Convert input to numpy array
    source_points_3d = np.array(source_points_3d)
    if source_points_3d.ndim == 1:
        source_points_3d = source_points_3d.reshape(1, 3)

    num_points = len(source_points_3d)
    print(f"Finding correspondences for {num_points} source points...")

    # Find closest vertices to input 3D points
    source_vertex_indices, closest_distances = find_closest_vertices_batch(source_points_3d, source_mesh.vert)

    print("Closest vertices to input 3D points:")
    for i, (point, vertex_idx, distance) in enumerate(zip(source_points_3d, source_vertex_indices, closest_distances)):
        print(f"  Point {i}: {point} -> Vertex {vertex_idx} (distance: {distance:.4f})")

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

    # Find correspondences for each source point
    correspondences = []
    for i, source_vertex_idx in enumerate(source_vertex_indices):
        # Get feature vector for this source vertex
        source_point_feature = source_features[source_vertex_idx:source_vertex_idx+1]

        # Compute similarity to all target vertices
        similarities = cosine_similarity(source_point_feature, target_features)
        similarity_scores = similarities[0].cpu().numpy()

        # Find most similar target vertex
        target_vertex_idx = np.argmax(similarity_scores)
        max_similarity = similarity_scores[target_vertex_idx]

        correspondences.append((int(source_vertex_idx), int(target_vertex_idx), float(max_similarity)))
        print(f"  Point {i}: Source vertex {source_vertex_idx} -> Target vertex {target_vertex_idx} (similarity: {max_similarity:.4f})")

    # Generate distinct colors for each correspondence
    correspondence_colors = generate_distinct_colors(num_points)

    return correspondences, correspondence_colors, source_vertex_indices, closest_distances

def visualize_multi_point_correspondences(
    source_mesh,
    target_mesh,
    correspondences,
    correspondence_colors,
    source_vertex_indices
):
    """
    Visualize multiple point correspondences with colored 3D markers at correspondence locations.

    Args:
        source_mesh: source mesh container
        target_mesh: target mesh container
        correspondences: list of (source_idx, target_idx, similarity_score) tuples
        correspondence_colors: RGB colors for each correspondence pair (N, 3)
        source_vertex_indices: indices of source vertices (N,)
    """

    print(f"🎨 Visualizing {len(correspondences)} correspondences with colored 3D markers:")
    for i, ((source_idx, target_idx, similarity), color) in enumerate(zip(correspondences, correspondence_colors)):
        print(f"  Correspondence {i+1}: Source vertex {source_idx} ↔ Target vertex {target_idx} (similarity: {similarity:.4f})")

    # Start with mesh visualization (gray background)
    source_mesh_colors = np.ones((len(source_mesh.vert), 3)) * 0.7  # Gray
    target_mesh_colors = np.ones((len(target_mesh.vert), 3)) * 0.7  # Gray

    # Add vertex highlighting as backup
    for i, ((source_idx, target_idx, similarity), color) in enumerate(zip(correspondences, correspondence_colors)):
        source_mesh_colors[source_idx] = color
        target_mesh_colors[target_idx] = color

    # Create the base mesh visualization
    d = mp.subplot(source_mesh.vert, source_mesh.face, c=source_mesh_colors, s=[2, 2, 0])
    mp.subplot(target_mesh.vert, target_mesh.face, c=target_mesh_colors, s=[2, 2, 1], data=d)

    # Try to add 3D markers (octahedrons) for better visibility
    try:
        # Extract correspondence indices and colors
        source_indices = [corr[0] for corr in correspondences]
        target_indices = [corr[1] for corr in correspondences]

        # Create 3D markers for source mesh
        source_highlight_verts, source_highlight_faces, source_highlight_colors = create_highlight_regions(
            source_mesh.vert, source_mesh.face, source_indices, correspondence_colors
        )

        # Create 3D markers for target mesh
        target_highlight_verts, target_highlight_faces, target_highlight_colors = create_highlight_regions(
            target_mesh.vert, target_mesh.face, target_indices, correspondence_colors
        )

        # Add the 3D markers to visualization
        if len(source_highlight_verts) > 0:
            mp.subplot(source_highlight_verts, source_highlight_faces, c=source_highlight_colors, s=[2, 2, 0], data=d)

        if len(target_highlight_verts) > 0:
            mp.subplot(target_highlight_verts, target_highlight_faces, c=target_highlight_colors, s=[2, 2, 1], data=d)

        print(f"🎯 Enhanced visualization with 3D octahedron markers...")

    except Exception as e:
        print(f"ℹ️  Using vertex highlighting (3D markers unavailable: {str(e)})")

    print(f"\n📊 Correspondence Statistics:")
    similarities = [corr[2] for corr in correspondences]
    print(f"  - Average similarity: {np.mean(similarities):.4f}")
    print(f"  - Min similarity: {np.min(similarities):.4f}")
    print(f"  - Max similarity: {np.max(similarities):.4f}")
    print(f"  - Each correspondence pair has matching colors on both meshes")

def run_multi_point_correspondence_analysis(
    source_mesh,
    target_mesh,
    source_points_3d,
    device,
    pipe,
    dino_model,
    prompt="a textured 3D model",
    num_views=50
):
    """
    Complete pipeline for multi-point correspondence analysis.

    Args:
        source_mesh: source mesh container
        target_mesh: target mesh container
        source_points_3d: 3D coordinates (N, 3) as array-like or torch tensor
        device: torch device
        pipe: diffusion pipeline
        dino_model: DINO model
        prompt: text prompt for feature extraction
        num_views: number of views for rendering

    Returns:
        correspondences: list of (source_idx, target_idx, similarity_score) tuples
        correspondence_colors: RGB colors for each correspondence pair (N, 3)
        source_vertex_indices: indices of closest vertices to input 3D points (N,)
        closest_distances: distances from input points to closest vertices (N,)
    """

    # Find correspondences
    correspondences, correspondence_colors, source_vertex_indices, closest_distances = multi_point_correspondence_analysis(
        device=device,
        pipe=pipe,
        dino_model=dino_model,
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_points_3d=source_points_3d,
        prompt=prompt,
        num_views=num_views
    )

    # Visualize results
    visualize_multi_point_correspondences(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        correspondences=correspondences,
        correspondence_colors=correspondence_colors,
        source_vertex_indices=source_vertex_indices
    )

    return correspondences, correspondence_colors, source_vertex_indices, closest_distances
