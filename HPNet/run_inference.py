import open3d as o3d
import random
import h5py
import numpy as np
import torch
from trainer import Trainer
from option import parser
from utils.loss_utils import compute_miou, compute_type_miou_abc


def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    # S = eigenvalues, U = eigenvectors
    return S, U

def rotation_matrix_a_to_b(A, B, eps=1e-8):
    """
    Rotates vector A to align with vector B in 3D.
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + eps)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + eps)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def inference_on_custom_input(self, xyz, normals, save_path=None):


    # 1. Center the point cloud
    xyz_mean = xyz.mean(axis=0)
    xyz = xyz - xyz_mean

    # 2. PCA alignment
    S, U = pca_numpy(xyz)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    xyz = (R @ xyz.T).T
    normals = (R @ normals.T).T

    # 3. Normalize to unit sphere
    xyz_scale = np.linalg.norm(xyz, axis=1).max()
    xyz = xyz / (xyz_scale + 1e-8)

    # 4. Normalize normals
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)


    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.eval()

    with torch.no_grad():
        # Prepare input tensors
        # Convert numpy arrays to torch tensors and permute dimensions
        xyz_th = torch.from_numpy(xyz).float().unsqueeze(0).permute(0, 2, 1).cuda()     # [1, 3, N]
        normals_th = torch.from_numpy(normals).float().unsqueeze(0).permute(0, 2, 1).cuda()  # [1, 3, N]

        # Forward through model
        if self.opt.input_normal:
            affinity_feat, type_per_point, normal_per_point, param_per_point, sub_idx = self.model(xyz_th, normals_th, postprocess=True)
        else:
            affinity_feat, type_per_point, param_per_point, sub_idx = self.model(xyz_th, normals_th, postprocess=True)
        
        print("affinity_feat:", affinity_feat.shape)
        print("type_per_point:", type_per_point.shape)
        print("normal_per_point:", normal_per_point.shape)
        print("param_per_point:", param_per_point.shape)
        print("sub_idx:", sub_idx.shape if hasattr(sub_idx, 'shape') else type(sub_idx))

        # Gather subsampled points
        xyz_sub = torch.gather(xyz_th, -1, sub_idx.unsqueeze(1).repeat(1, 3, 1))  # [1, 3, S]
        N_gt = torch.gather(normals_th.permute(0, 2, 1), 1, sub_idx.unsqueeze(-1).repeat(1, 1, 3))  # [1, S, 3]

        # Build affinity matrices
        from utils.abc_utils import construction_affinity_matrix_type, construction_affinity_matrix_normal, compute_entropy, mean_shift

        print("Building affinity matrices...")
        affinity_matrix = construction_affinity_matrix_type(xyz_sub, type_per_point, param_per_point, self.opt.sigma)
        print("Affinity_type matrix shape:", affinity_matrix.shape)
        affinity_matrix_normal = construction_affinity_matrix_normal(xyz_sub, N_gt, sigma=self.opt.normal_sigma, knn=self.opt.edge_knn)
        print("Affinity_normal matrix shape:", affinity_matrix.shape)


        # Spectral embedding
        spec_embedding_list = []
        weight_ent = []

        feat_ent = self.opt.feat_ent_weight - float(compute_entropy(affinity_feat))
        weight_ent.append(feat_ent)
        spec_embedding_list.append(affinity_feat)

        topk = self.opt.topK
        e, v = torch.lobpcg(affinity_matrix, k=topk, niter=10)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-16)
        dis_ent = self.opt.dis_ent_weight - float(compute_entropy(v))
        weight_ent.append(dis_ent)
        spec_embedding_list.append(v)

        edge_topk = self.opt.edge_topK
        e, v = torch.lobpcg(affinity_matrix_normal, k=edge_topk, niter=10)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-16)
        edge_ent = self.opt.edge_ent_weight - float(compute_entropy(v))
        weight_ent.append(edge_ent)
        spec_embedding_list.append(v)

        norm_weight_ent = weight_ent / np.linalg.norm(weight_ent)
        weighted_list = [spec_embedding_list[i] * weight_ent[i] for i in range(len(spec_embedding_list))]
        spectral_embedding = torch.cat(weighted_list, dim=-1)

        # Mean shift clustering
        cluster_pred = mean_shift(spectral_embedding, bandwidth=self.opt.bandwidth).squeeze().cpu().numpy()  # [S,]

        # Get final segmented point cloud in original xyz_sub order
        xyz_sub_np = xyz_sub.squeeze().permute(1, 0).cpu().numpy()  # [S, 3]

        # Step 1: Undo scaling
        xyz_sub_np = xyz_sub_np * xyz_scale
        # Step 2: Undo PCA rotation
        xyz_sub_np = (R.T @ xyz_sub_np.T).T
        # Step 3: Undo centering
        xyz_sub_np = xyz_sub_np + xyz_mean.squeeze()

        # Save colored point cloud
        if save_path is not None:
            colors = np.array([_random_color(label) for label in cluster_pred]) / 255.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_sub_np)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(save_path, pcd)

        print("Unique cluster labels:", np.unique(cluster_pred))
        return cluster_pred


def _random_color(label):
    random.seed(label)
    return [random.randint(64, 255), random.randint(64, 255), random.randint(64, 255)]



if __name__ == "__main__":
    # Load xyz + normals from .h5 file
    input_h5_path = '/workspace/HPNet/BenchmarkingInputs/abc_00007_normal2.h5'
    file_name = input_h5_path.split('/')[-1].split('.')[0]  # Extract filename without extension
    print(f"Processing file: {file_name}")
    output_ply_path = f'/workspace/HPNet/BenchmarkingOutputs/{file_name}_prediction.ply'

    # Extract xyz and normals from the .h5 file
    with h5py.File(input_h5_path, 'r') as f:
        print("Available keys in .h5 file:", list(f.keys()))
        xyz = f['xyz'][:]
        normals = f['normal'][:]

    # Minimal args needed for inference
    args = [
        '--checkpoint_path', '/workspace/HPNet/abc_normal/abc_normal',
        '--log_dir', './log',  # Required only because Trainer uses it during init
        '--input_normal', '1',
        '--bandwidth', '0.85',
    ]
    FLAGS = parser.parse_args(args)

    trainer = Trainer(FLAGS, skip_data=True)  # skip_data=True to avoid building dataloader

    # Bind the inference method to the trainer instance using dynamic method binding
    # (monkey patching)
    trainer.inference_on_custom_input = inference_on_custom_input.__get__(trainer)

    # Run inference and save output PLY
    trainer.inference_on_custom_input(xyz, normals, save_path=output_ply_path)
