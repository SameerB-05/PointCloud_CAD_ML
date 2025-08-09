import trimesh
import yaml
import numpy as np
import os

# === Step 1: File paths ===
obj_path = r"C:\Users\samee\Documents\IITB_INTERN\ABC_Dataset_Samples\Objects\00000003\00000003_1ffb81a71e5b402e966b9341_trimesh_002.obj"
yml_path = r"C:\Users\samee\Documents\IITB_INTERN\ABC_Dataset_Samples\Features\00000003\00000003_1ffb81a71e5b402e966b9341_features_002.yml"

# === Step 2: Load mesh ===
mesh = trimesh.load(obj_path, process=False)

# === Step 3: Sample points from mesh ===
num_points = 7000
points, face_indices = mesh.sample(num_points, return_index=True)

# === Step 4: Parse feat.yml ===
with open(yml_path, "r") as f:
    feat_data = yaml.safe_load(f)

surfaces = feat_data.get("surfaces", [])

# === Step 5: Build face → label mapping ===
face_to_label = {}  # face_id → (semantic_class, instance_id)

for instance_id, surf in enumerate(surfaces):
    label = surf.get("type", "None")
    for face_id in surf.get("face_indices", []):
        face_to_label[face_id] = (label, instance_id)

# === Step 6: Assign semantic labels to sampled points ===
semantic_labels = []
instance_labels = []

for fidx in face_indices:
    if fidx in face_to_label:
        label, inst_id = face_to_label[fidx]
    else:
        label, inst_id = "None", -1
    semantic_labels.append(label)
    instance_labels.append(inst_id)

# === Step 7: Map semantic class names to integer indices ===
unique_labels = sorted(set(semantic_labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
semantic_class_indices = [label_to_index[lbl] for lbl in semantic_labels]

# === Step 8: Output file names (next to .obj file) ===
output_dir = os.path.dirname(obj_path)
xyz_path = os.path.join(output_dir, "pointcloud_unlabeled.xyz")
#label_map_path = os.path.join(output_dir, "semantic_label_map2.txt")
xyzc_path = os.path.join(output_dir, "pointcloud_labelled.xyzc")

np.savetxt(xyz_path, points, fmt="%.6f %.6f %.6f")

# === Step 9: Save point clouds with only instance IDs ===
instance_labels = np.array(instance_labels).reshape(-1, 1)
points_with_instances = np.hstack((points, instance_labels))
np.savetxt(xyzc_path, points_with_instances, fmt="%.6f %.6f %.6f %d")


print("✅ Done. Files saved:")
print(f"→ {xyz_path}")
print(f"→ {xyzc_path}")

