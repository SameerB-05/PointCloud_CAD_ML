import trimesh
import matplotlib.pyplot as plt

# Load the OBJ file
mesh_path = r"C:\Users\samee\Documents\IITB_INTERN\ABC_Dataset_Samples\Objects\00000010\00000010_b4b99d35e04b4277931f9a9c_trimesh_000.obj"
mesh = trimesh.load(mesh_path)

# Show basic info
print(mesh)

# Show using built-in viewer (opens an interactive window)
mesh.show()
