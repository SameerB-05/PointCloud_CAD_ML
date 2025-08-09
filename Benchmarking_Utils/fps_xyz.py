import numpy as np
import open3d as o3d
from tqdm import tqdm


def load_xyz(file_path):
    """Load .xyz point cloud (assumes 3 columns: x y z)"""
    return np.loadtxt(file_path)[:, :3]


def farthest_point_sampling(points, n_samples):
    """
    Perform FPS to sample 'n_samples' points from 'points'.
    Input: points (N, 3), Output: (n_samples, 3)
    """
    N = points.shape[0]
    sampled_pts = np.zeros((n_samples, 3))
    selected_idxs = np.zeros(n_samples, dtype=int)

    distances = np.full(N, np.inf)
    farthest_idx = np.random.randint(N)

    for i in tqdm(range(n_samples), desc="FPS sampling"):
        selected_idxs[i] = farthest_idx
        sampled_pts[i] = points[farthest_idx]

        dist = np.linalg.norm(points - points[farthest_idx], axis=1)
        distances = np.minimum(distances, dist)

        farthest_idx = np.argmax(distances)

    return sampled_pts


def visualize_points(points, title="Point Cloud"):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([pcd], window_name=title)


def save_xyz(points, out_path):
    np.savetxt(out_path, points, fmt="%.6f")
    print(f"Saved: {out_path}")


# ========== USAGE ==========

# 📁 Change this to your input .xyz file path
input_xyz_path = r"C:\Users\samee\Documents\IITB_INTERN\point2cad\BenchmarkingInputs\abc_00949.xyz"
output_xyz_path = r"C:\Users\samee\Documents\IITB_INTERN\point2cad\BenchmarkingInputs\abc_00949b.xyz"

points = load_xyz(input_xyz_path)
print(f"Loaded {points.shape[0]} points.")

# 🎯 Apply FPS to get 7000 points
sampled_points = farthest_point_sampling(points, 7000)

# 💾 Save & 👁️ Visualize
save_xyz(sampled_points, output_xyz_path)
visualize_points(sampled_points, title="FPS 7000 Points")
