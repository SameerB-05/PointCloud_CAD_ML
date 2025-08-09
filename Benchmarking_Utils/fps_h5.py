import numpy as np
import h5py
from tqdm import tqdm
import open3d as o3d


def load_h5(file_path):
    """Load .h5 file with 'xyz' and 'normals' datasets"""
    with h5py.File(file_path, 'r') as f:
        xyz = f['xyz'][:]        # (N, 3)
        normals = f['normals'][:]  # (N, 3)
    return xyz, normals


def farthest_point_sampling(points, n_samples):
    """
    FPS sampling: returns indices of sampled points
    """
    N = points.shape[0]
    indices = np.zeros(n_samples, dtype=np.int32)
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)

    for i in tqdm(range(n_samples), desc="FPS"):
        indices[i] = farthest
        centroid = points[farthest]
        dist = np.linalg.norm(points - centroid, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return indices


def visualize(points, title="Sampled Points"):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([pcd], window_name=title)


def save_to_npz(file_path, xyz, normals):
    np.savez(file_path, xyz=xyz, normals=normals)
    print(f"Saved: {file_path}")


# ========== USAGE ==========

input_h5 = r"C:\Users\samee\Documents\IITB_INTERN\HPNet\BenchmarkingInputs\abc_00470_normal.h5"
output_npz = r"C:\Users\samee\Documents\IITB_INTERN\HPNet\BenchmarkingInputs\abc_00470b_normal.h5"

xyz, normals = load_h5(input_h5)
print(f"Loaded: {xyz.shape[0]} points")

sampled_idx = farthest_point_sampling(xyz, 7000)

xyz_sampled = xyz[sampled_idx]
normals_sampled = normals[sampled_idx]

# Save and visualize
save_to_npz(output_npz, xyz_sampled, normals_sampled)
visualize(xyz_sampled, title="FPS 7000 Points")
