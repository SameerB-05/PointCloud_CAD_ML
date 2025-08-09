import numpy as np
from scipy.optimize import linear_sum_assignment # Hungarian algorithm for optimal assignment

def load_xyzc(path):

    data = np.loadtxt(path)
    if data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns (x,y,z,c), got {data.shape[1]}")
    points = data[:, :3]
    labels = data[:, 3].astype(int)
    return points, labels

def compute_iou_matrix(gt_labels, pred_labels):
    """
    Build an [n_gt x n_pred] IoU matrix.
    """
    gt_ids   = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels)

    M = len(gt_ids)
    N = len(pred_ids)
    iou_mat = np.zeros((M, N), dtype=float)

    # Precompute masks
    gt_masks   = {i: (gt_labels == i) for i in gt_ids}
    pred_masks = {j: (pred_labels == j) for j in pred_ids}

    for i, gi in enumerate(gt_ids):
        for j, pj in enumerate(pred_ids):
            inter = np.sum(gt_masks[gi] & pred_masks[pj])
            union = np.sum(gt_masks[gi] | pred_masks[pj])
            if union > 0:
                iou_mat[i, j] = inter / union
    return gt_ids, pred_ids, iou_mat

def match_instances(iou_mat):

    cost = -iou_mat  # maximize IoU → minimize cost
    gt_idx, pred_idx = linear_sum_assignment(cost)
    return gt_idx, pred_idx

def mean_iou(gt_ids, pred_ids, iou_mat, matched_gt_idx, matched_pred_idx):
    """
    Compute mean IoU:
      - Take IoU for each matched pair
      - Assign IoU=0 for any GT with no match
    """
    M = len(gt_ids)
    matched_set = set(matched_gt_idx)
    ious = [iou_mat[i, j] for i, j in zip(matched_gt_idx, matched_pred_idx)]
    # add zeros for unmatched GTs
    num_unmatched = M - len(matched_set)
    ious.extend([0.0] * num_unmatched)
    return float(np.mean(ious))

def coverage(gt_labels, pred_labels):
    """
    For each GT instance, compute:
       coverage_i = max_j |G_i ∩ P_j| / |G_i|
    Return average over GT instances.
    """
    gt_ids   = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels)

    pred_masks = {j: (pred_labels == j) for j in pred_ids}
    coverages = []

    for gi in gt_ids:
        G = (gt_labels == gi)
        size_G = G.sum()
        if size_G == 0:
            continue
        # Fraction of G covered by each P_j
        frac = [ np.sum(G & pred_masks[pj]) / size_G for pj in pred_ids ]
        coverages.append(max(frac) if frac else 0.0)

    return float(np.mean(coverages)) if coverages else 0.0

if __name__ == "__main__":
    # === Example usage ===
    gt_path   = r"C:\Users\samee\Documents\IITB_INTERN\BenchmarkingGTs\abc_00006_gt.xyzc"
    pred_path = r"C:\Users\samee\Documents\IITB_INTERN\point2cad\BenchmarkingOutputs\abc_00006b_prediction.xyzc"

    _, gt_lbl = load_xyzc(gt_path) # ground truth labels
    _, pr_lbl = load_xyzc(pred_path) # predicted labels

    gt_ids, pred_ids, iou_mat = compute_iou_matrix(gt_lbl, pr_lbl)
    gt_idx, pr_idx = match_instances(iou_mat)

    mIoU = mean_iou(gt_ids, pred_ids, iou_mat, gt_idx, pr_idx)
    cov  = coverage(gt_lbl, pr_lbl)

    print(f"Mean IoU      : {mIoU:.4f}")
    print(f"GT Coverage   : {cov:.4f}")
