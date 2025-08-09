import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Set path to .xyzc file
file_path = r"C:\Users\samee\Documents\IITB_INTERN\point2cad\assets\abc_00470.xyzc"

# Load .xyzc (expects 4 columns: x, y, z, segment_id)
data = np.loadtxt(file_path)
points = data[:, :3]
labels = data[:, 3].astype(np.int32)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Map each segment ID to a unique color
unique_labels = np.unique(labels)
#colors = plt.get_cmap("tab20")(labels % 20)[:, :3]  # tab20 supports up to 20 unique colors

np.random.seed(22)
label_to_color = {label: np.random.rand(3) for label in unique_labels}
colors = np.array([label_to_color[l] for l in labels])


# Assign colors
pcd.colors = o3d.utility.Vector3dVector(colors)

# Display basic info
print(f"Loaded point cloud with {points.shape[0]} points and {len(unique_labels)} segments")

# Visualize
o3d.visualization.draw_geometries([pcd], window_name="Segmented Point Cloud Viewer")
