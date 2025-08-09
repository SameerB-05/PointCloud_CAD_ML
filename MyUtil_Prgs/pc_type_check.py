import numpy as np
import torch

print(torch.__version__)

points = np.loadtxt(r"C:\Users\samee\OneDrive\Documents\INTERN_IITB\Armadillo_pointcloud.xyz")
print(points.shape[1])
