import open3d as o3d
import numpy as np

# Path to the output PLY file with color-coded clusters
ply_file_path = r"C:\Users\samee\Documents\IITB_INTERN\HPNet\BenchmarkingOutputs\abc_00007_normal2_prediction.ply"
# Load the point cloud
pcd = o3d.io.read_point_cloud(ply_file_path)

# Sanity check: print point and color info
print(f"Loaded {len(pcd.points)} points")
print(f"Has colors? {'Yes' if pcd.has_colors() else 'No'}")

# Convert colors to NumPy array
colors = np.asarray(pcd.colors)
# Count unique colors
unique_colors = np.unique(colors, axis=0)
print(f"Number of unique labels (based on unique colors): {len(unique_colors)}")




# Visualize the point cloud with colors
o3d.visualization.draw_geometries(
    [pcd],
    window_name="Segmented Output - Cluster Colors",
    width=900,
    height=700,
    point_show_normal=False
)
