import numpy as np
import scipy.linalg
import torch
from torch import linalg
import sys

def compute_mmdist(text_embeddings: np.ndarray, motion_embeddings: np.ndarray) -> float:
    """
    Computes Multi-Modal Distance (MMDist) between paired text and motion embeddings.

    Args:
        text_embeddings (np.ndarray): shape (N, D), text features
        motion_embeddings (np.ndarray): shape (N, D), motion features

    Returns:
        float: mean Euclidean distance between aligned pairs
    """
    assert text_embeddings.shape == motion_embeddings.shape, "Shape mismatch"
    
    distances = np.linalg.norm(text_embeddings - motion_embeddings, axis=1)
    mmdist = np.mean(distances)
    
    return mmdist

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)


def sqrtm(input):
    m = input.detach().cpu().numpy().astype(np.float64_)
    sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m)).to(input)
    return sqrtm


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = torch.sum(torch.square(matrix1), axis=1,
                   keepdims=True)  # shape (num_test, 1)
    d3 = torch.sum(torch.square(matrix2), axis=1)  # shape (num_train, )
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def euclidean_distance_matrix_np(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1,
                keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = (torch.unsqueeze(torch.arange(size),
                              1).to(mat.device).repeat_interleave(size, 1))
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = torch.cat(top_k_list, dim=1)
    return top_k_mat


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma



def calculate_activation_statistics_np(activations):
    """
    Computes the mean and covariance of activation features.

    Args:
        activations (np.ndarray): Shape (N, D)

    Returns:
        mu: (D,) mean vector
        sigma: (D, D) covariance matrix
    """
    assert not np.isnan(activations).any(), "Found NaNs in activations"
    assert not np.isinf(activations).any(), "Found Infs in activations"
    assert activations.ndim == 2, "Activations must be 2D"
    # assert activations.shape[0] > activations.shape[1], "More features than samples — unstable stats"

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculates Fréchet Distance with robust numerical stability.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Stabilize
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1 += offset
    sigma2 += offset

    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        print("⚠️ sqrtm returned NaNs or Infs. Adding more offset...")
        sigma1 += offset
        sigma2 += offset
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # If imaginary values are too high, fallback
    if np.iscomplexobj(covmean):
        imag_comp = np.abs(np.imag(covmean)).max()
        if imag_comp > 1e-3:
            print(f"⚠️ High imaginary component in sqrtm. Using real part only.")
            # print(f"⚠️ High imaginary component in sqrtm: {imag_comp:.3f}. Using real part only.")
        covmean = np.real(covmean)

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)



def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples,
                                     diversity_times,
                                     replace=False)
    second_indices = np.random.choice(num_samples,
                                      diversity_times,
                                      replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices],
                       axis=1)
    return dist.mean()


def calculate_diversity_np(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples,
                                     diversity_times,
                                     replace=False)
    second_indices = np.random.choice(num_samples,
                                      diversity_times,
                                      replace=False)
    dist = scipy.linalg.norm(activation[first_indices] -
                             activation[second_indices],
                             axis=1)
    return dist.mean()


def calculate_multimodality_np(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent,
                                   multimodality_times,
                                   replace=False)
    second_dices = np.random.choice(num_per_sent,
                                    multimodality_times,
                                    replace=False)
    dist = scipy.linalg.norm(activation[:, first_dices] -
                             activation[:, second_dices],
                             axis=2)
    return dist.mean()


# motion reconstructions metrics


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat, (scale, R, t)


def compute_mpjpe(preds,
                  target,
                  valid_mask=None,
                  pck_joints=None,
                  sample_wise=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance)
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape,
                                              target.shape)  # BxJx3
    mpjpe = torch.norm(preds - target, p=2, dim=-1)  # BxJ

    if pck_joints is None:
        if sample_wise:
            mpjpe_seq = ((mpjpe * valid_mask.float()).sum(-1) /
                         valid_mask.float().sum(-1)
                         if valid_mask is not None else mpjpe.mean(-1))
        else:
            mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_seq
    else:
        mpjpe_pck_seq = mpjpe[:, pck_joints]
        return mpjpe_pck_seq


def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)


def calc_mpjpe(preds, target, align_inds=[0], sample_wise=True, trans=None):
    # Expects BxJx3
    valid_mask = target[:, :, 0] != -2.0
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    if align_inds is not None:
        preds_aligned = align_by_parts(preds, align_inds=align_inds)
        if trans is not None:
            preds_aligned += trans
        target_aligned = align_by_parts(target, align_inds=align_inds)
    else:
        preds_aligned, target_aligned = preds, target
    mpjpe_each = compute_mpjpe(preds_aligned,
                               target_aligned,
                               valid_mask=valid_mask,
                               sample_wise=sample_wise)
    return mpjpe_each


def calc_accel(preds, target):
    """
    Mean joint acceleration error
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape,
                                              target.shape)  # BxJx3
    assert preds.dim() == 3
    # Expects BxJx3
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
    normed = torch.linalg.norm(accel_pred - accel_gt, dim=-1)
    accel_seq = normed.mean(1)
    return accel_seq


def calc_pampjpe(preds, target, sample_wise=True, return_transform_mat=False):
    # Expects BxJx3
    target, preds = target.float(), preds.float()
    # extracting the keypoints that all samples have valid annotations
    # valid_mask = (target[:, :, 0] != -2.).sum(0) == len(target)
    # preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(preds[:, valid_mask], target[:, valid_mask])
    # pa_mpjpe_each = compute_mpjpe(preds_tranformed, target[:, valid_mask], sample_wise=sample_wise)

    preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(
        preds, target)
    pa_mpjpe_each = compute_mpjpe(preds_tranformed,
                                  target,
                                  sample_wise=sample_wise)

    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each


# from action2motion
def calculate_diversity_multimodality(activations,
                                      labels,
                                      num_labels,
                                      diversity_times=200,
                                      multimodality_times=20):
    labels = labels.long()
    num_motions = activations.shape[0]  # len(labels)

    diversity = 0

    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    multimodality = 0
    label_quotas = np.zeros(num_labels)
    label_quotas[labels.unique(
    )] = multimodality_times  # if a label does not appear in batch, its quota remains zero
    while np.any(label_quotas > 0):
        # print(label_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not label_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        label_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation, second_activation)

    multimodality /= (multimodality_times * num_labels)

    return diversity, multimodality


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance_np(statistics_1[0], statistics_1[1],
                                         statistics_2[0], statistics_2[1])


# from: https://github.com/abdulfatir/gan-metrics-pytorch/blob/master/kid_score.py
def polynomial_mmd_averages(codes_g,
                            codes_r,
                            n_subsets=50,
                            subset_size=64,
                            ret_var=True,
                            output=sys.stdout,
                            **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    replace = subset_size < len(codes_g)

    for i in range(n_subsets):
        g = codes_g[choice(len(codes_g), subset_size, replace=replace)]
        r = codes_r[choice(len(codes_r), subset_size, replace=replace)]
        o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
        if ret_var:
            mmds[i], vars[i] = o
        else:
            mmds[i] = o

    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g,
                   codes_r,
                   degree=3,
                   gamma=None,
                   coef0=1,
                   var_at_m=None,
                   ret_var=True):
    from sklearn.metrics.pairwise import polynomial_kernel
    
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX,
                              K_XY,
                              K_YY,
                              var_at_m=var_at_m,
                              ret_var=ret_var)


def _mmd2_and_variance(K_XX,
                       K_XY,
                       K_YY,
                       unit_diagonal=False,
                       mmd_est='unbiased',
                       block_size=1024,
                       var_at_m=None,
                       ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m) + (Kt_YY_sum + sum_diag_Y) /
                (m * m) - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) *
        (_sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum) - 1 /
        (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2) + 1 / (m * m * m1) *
        (_sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum) -
        2 / m**4 * K_XY_sum**2 - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX) +
        2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum)
    zeta2_est = (1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum) - 1 / (m * m1)**2 *
                 (Kt_XX_sum**2 + Kt_YY_sum**2) + 2 / (m * m) * K_XY_2_sum -
                 2 / m**4 * K_XY_sum**2 - 4 / (m * m * m1) *
                 (dot_XX_XY + dot_YY_YX) + 4 / (m**3 * m1) *
                 (Kt_XX_sum + Kt_YY_sum) * K_XY_sum)
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est +
               2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def calculate_kid(real_activations, generated_activations):
    kid_values = polynomial_mmd_averages(real_activations,
                                         generated_activations,
                                         n_subsets=50)
    results = (kid_values[0].mean(), kid_values[0].std())
    return results
