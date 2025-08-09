import numpy as np
import open3d as o3d
import h5py
import os

# === CHANGE THIS LINE TO YOUR INPUT FILE ===
xyz_path = r"C:\Users\samee\Documents\IITB_INTERN\HPNet\BenchmarkingInputs\abc_00007.xyz"

# === Load XYZ File ===
points = np.loadtxt(xyz_path)  # shape [N, 3]

# === Create Open3D Point Cloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd], window_name="Loaded Point Cloud", width=800, height=600)

# === Compute Normals ===
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))

# === Ensure consistent orientation of normals ===
pcd.orient_normals_consistent_tangent_plane(100)


# (Optional) Orient normals consistently (can skip if not needed)
# pcd.orient_normals_consistent_tangent_plane(50)

# === Visualize ===
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# === Convert to numpy arrays ===
xyz_np = np.asarray(pcd.points).astype(np.float32)
normal_np = np.asarray(pcd.normals).astype(np.float32)

# === Save to .h5 ===
save_path = xyz_path.replace(".xyz", "_normal2.h5")

with h5py.File(save_path, "w") as f:
    f.create_dataset("xyz", data=xyz_np)
    f.create_dataset("normal", data=normal_np)

print(f"✅ Saved H5 file with normals: {save_path}")
