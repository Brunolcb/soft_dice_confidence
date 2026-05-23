from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
import pandas as pd
from sklearn.metrics import auc
from src.metrics import rc_curve, dice_coef, hd95, accuracy,  rc_curve_max
from typing import Sequence, Union, List
import SimpleITK as sitk
import radiomics
import bisect

ArrayLike = Union[np.ndarray, Sequence[np.ndarray]]

def sum_entropy(y_hat, epsilon=1e-12, thresh=0.95):
    sum_uncert_array = []
    for mask in y_hat:
        H = entropy_func(mask, epsilon=1e-12)
        U = entropy_func(mask, epsilon=1e-12)>thresh
        sum_uncert_array.append(np.sum(U))
    return np.array(sum_uncert_array)

def _is_list_of_arrays(x) -> bool:
    return isinstance(x, (list, tuple)) and all(isinstance(a, np.ndarray) for a in x)

def _entropy_nd(a: np.ndarray, epsilon: float) -> np.ndarray:
    # Works for boolean or probability arrays in [0,1]
    p = a.astype(np.float64, copy=False)
    return -(p * np.log(p + epsilon) + (1.0 - p) * np.log(1.0 - p + epsilon)) / np.log(2.0)

def entropy_func(mask: ArrayLike, epsilon: float = 1e-12) -> ArrayLike:
    """
    If `mask` is a numpy array, returns an array of the same shape with entropy per element.
    If `mask` is a list/tuple of numpy arrays (possibly with different shapes), returns a list
    with the entropy computed elementwise for each array.
    """
    if isinstance(mask, np.ndarray):
        return _entropy_nd(mask, epsilon)
    if _is_list_of_arrays(mask):
        return [_entropy_nd(m, epsilon) for m in mask]
    # Fallback: try to coerce to array (kept for backward compatibility)
    mask_arr = np.asarray(mask)
    return _entropy_nd(mask_arr, epsilon)

def get_noise_with_Lcov(Lcov_path: str, img_shape: tuple, n=None):
    L = da.from_npy_stack(Lcov_path)

    if n is None:
        da.random.seed(0)
        eta = da.random.normal(0, size=L.shape[1])
        eta = L @ eta
        eta = eta.reshape(*img_shape)
    else:
        da.random.seed(0)
        eta = da.random.normal(0, size=(n, L.shape[1]))
        eta = eta @ L.T
        eta = eta.reshape(n, *img_shape)

    return eta.compute()

def plot_rc_curves(confidences: dict, errors: np.array, ax):
    random_aurc = np.mean(errors)

    ideal_coverage, ideal_risk, _ = rc_curve(-errors, errors, ideal=True)
    ideal_aurc = auc(ideal_coverage, ideal_risk)

    ax = plot_baselines(errors, ax)

    for name, confidence in confidences.items():
        plot_rc_curve(confidence, errors, name, ax, low_aurc=ideal_aurc, high_aurc=random_aurc)

    ax.set_xlim(0,1)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid()
    ax.legend()

    return ax

def plot_baselines(errors, ax, **kwargs):
    colors = [plt.colormaps['tab20'](i) for i in range(20)]
    
    markersize = 3
    styles = dict(
    random=dict(linestyle='--', color='gray', label="Random"),
    amsp=dict(color=colors[0], markersize=markersize, marker="v"),
    ane=dict(color=colors[1], markersize=markersize, marker="p"),
    mmmc=dict(color=colors[2], markersize=markersize, marker="D"),
    tla=dict(color=colors[3], markersize=markersize, marker="^"),
    pla=dict(color=colors[4], markersize=markersize, marker="o"),
    sdc=dict(color=colors[5], markersize=markersize, marker="s"),
    aef=dict(color=colors[6], markersize=markersize, marker="*"),
    aef_sdc=dict(color=colors[7], markersize=markersize, marker="h"),
    oracle=dict(linestyle=':',color='gray', markersize=markersize, label="Oracle"),
    idc_true=dict(color=colors[9], marker="*", zorder=-2),
    idc_hat=dict(color=colors[10], markersize=markersize+2, marker=".", zorder=-1),
    )
    ax.hlines(np.mean(errors), 0, 1, **styles['random'], **kwargs)

    coverages, risks, _ = rc_curve(-errors, errors, expert=False, ideal=True)

    ax.plot(coverages, risks, **styles['oracle'], **kwargs)

    return ax

def plot_rc_curve(confidence, errors, label, ax, i, low_aurc=0, high_aurc=None,
                  **kwargs):
    total_markers = 8
    
    colors = [plt.colormaps['tab20'](i) for i in range(20)]

    markersize = 3
    styles = dict(
    random=dict(linestyle='--', color='gray', label="Random"),
    amsp=dict(color=colors[0], markersize=markersize, marker="v"),
    ane=dict(color=colors[1], markersize=markersize, marker="p"),
    mmmc=dict(color=colors[2], markersize=markersize, marker="D"),
    tla=dict(color=colors[3], markersize=markersize, marker="^"),
    pla=dict(color=colors[4], markersize=markersize, marker="o"),
    sdc=dict(color=colors[5], markersize=markersize, marker="s"),
    aef=dict(color=colors[6], markersize=markersize, marker="*"),
    aef_sdc=dict(color=colors[7], markersize=markersize, marker="h"),
    oracle=dict(linestyle=':',color='gray', markersize=markersize, label="Oracle"),
    idc_true=dict(color=colors[9], marker="*", zorder=-2),
    idc_hat=dict(color=colors[10], markersize=markersize+2, marker=".", zorder=-1),
    sdc_trust_model= dict(color=colors[11], markersize=markersize, marker="p"),
    sdc_all_classes= dict(color=colors[12], markersize=markersize, marker="o"),
    sdc_ideal= dict(color=colors[13], markersize=markersize, marker="*"),
    )

    coverages, risks, _ = rc_curve(confidence, errors)

    aurc = auc(coverages, risks)
    aurc -= low_aurc
    
    if high_aurc is not None:
        high_aurc -= low_aurc
        aurc = aurc / high_aurc
     
    ax.plot(coverages, risks, label=f"{label}", **styles[label.lower()], markevery=calculate_markers(len(coverages), total_markers))
    #ax.plot(coverages, risks, label=f"{label} ({aurc:.3f})")

def plot_segmentation_performance_report(results_fpath):
    _results_fpath = Path(results_fpath)

    data = np.load(_results_fpath)
    y = data['y']
    y_hat = data['y_hat']

    assert np.equal(y.shape, y_hat.shape).all()

    hd95s = list(map(
        lambda ys_i: hd95(ys_i[0].squeeze(), ys_i[1].squeeze()),
        zip(y, y_hat)
    ))

    dices = list(map(
        lambda ys_i: dice_coef(ys_i[0].flatten(), ys_i[1].flatten()),
        zip(y, y_hat)
    ))

    fig, axs = plt.subplots(3,1)
    fig.set_size_inches(8,10)
    fig.suptitle(results_fpath.name)

    def plot_performance_hist(values, ax):
        ax.hist(values, bins=20)
        ylims = ax.get_ylim()
        mean_performance = np.mean(values)
        ax.vlines(mean_performance, *ylims, color='red', label=f"{mean_performance:.2f}")
        ax.set_ylim(*ylims)

        return ax

    axs[0].remove()
    ax_hd95 = plot_performance_hist(hd95s, fig.add_subplot(3,2,1))
    ax_hd95.set_xlabel('Hausdorff95')
    ax_hd95.legend()

    ax_dice = plot_performance_hist(dices, fig.add_subplot(3,2,2))
    ax_dice.set_xlabel('Dice')
    ax_dice.legend()

    gt_sizes = y.reshape(y.shape[0],-1).sum(1)
    # axs[1].hist(gt_sizes, bins=[0,]+np.linspace(1,max(gt_sizes),20,endpoint=True),
    axs[1].hist(gt_sizes, bins=20,
                label=f"# of empty = {np.sum(np.array(gt_sizes) == 0)}")
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].set_xlabel('gt size')

    probs = y_hat.flatten()
    axs[2].hist(probs, bins=20)
    axs[2].set_yscale('log')
    axs[2].set_xlabel('y_hat probability')

    fig.tight_layout()

    return fig

def calculate_markers(data_length, num_markers):
    if data_length == 0:
        return np.zeros(data_length, dtype=bool)  # Empty mask for no data
    if num_markers >= data_length:
        return np.ones(data_length, dtype=bool)  # All points get markers if fewer than requested
    # Create a boolean mask
    indices = np.linspace(0, data_length - 1, num_markers, dtype=int)
    mask = np.zeros(data_length, dtype=bool)
    mask[indices] = True
    return mask

def plot_aurc_curves(confidences: dict, errors: np.array, ax):
    random_aurc = np.mean(errors)

    ideal_coverage, ideal_risk, _ = rc_curve(-errors, errors, ideal=True)
    ideal_aurc = auc(ideal_coverage, ideal_risk)

    ax = plot_baselines(errors, ax)

    for i, name_confidence in enumerate(confidences.items()):
        name, confidence = name_confidence
        plot_rc_curve(confidence, errors, name, ax, i)

    ax.set_xlim(0,1)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid()
    ax.legend()

    return ax

def write_aurc_curves(confidences: dict, errors: np.array):
    aurcs = {}
    random_aurc = np.mean(errors)

    ideal_coverage, ideal_risk, _ = rc_curve(-errors, errors, ideal=True)
    ideal_aurc = auc(ideal_coverage, ideal_risk)
    aurcs['Random'] = random_aurc
    aurcs['Ideal'] = ideal_aurc
    for name, confidence in confidences.items():
        coverage, risk, _ = rc_curve(confidence, errors)
        aurc = auc(coverage, risk)
        aurcs[name] = aurc
    return aurcs

def write_aurc_curves_by_min_coverage(confidences: dict, errors: np.array, min_coverage=0.5):
    def filter_coverage_and_risk(coverage, risk, min_coverage):
        """Filter coverage and risk for values where coverage >= min_value"""
        mask = coverage >= min_coverage
        filtered_coverage = coverage[mask]
        filtered_risk = risk[mask]
        if len(filtered_coverage) == 1:
            # Duplicate the single point to ensure at least two points
            filtered_coverage = np.concatenate([filtered_coverage, filtered_coverage])
            filtered_risk = np.concatenate([filtered_risk, filtered_risk])

        return filtered_coverage, filtered_risk
    aurcs = {}
    random_aurc = np.mean(errors)

    # Ideal curve
    ideal_coverage, ideal_risk, _ = rc_curve(-errors, errors, ideal=True)
    filtered_ideal_coverage, filtered_ideal_risk = filter_coverage_and_risk(ideal_coverage, ideal_risk, min_coverage)
    ideal_aurc = auc(filtered_ideal_coverage, filtered_ideal_risk)
    aurcs['Random'] = random_aurc*(1-min(ideal_coverage))
    aurcs['Ideal'] = ideal_aurc

    # Compute AURCs for each confidence
    for name, confidence in confidences.items():
        coverage, risk, _ = rc_curve(confidence, errors)
        filtered_coverage, filtered_risk = filter_coverage_and_risk(coverage, risk, min_coverage)
        aurc = auc(filtered_coverage, filtered_risk)
        aurcs[name] = aurc

    return aurcs

def write_naurc_curves(confidences: dict, errors: np.array):
    naurcs = {}
    random_aurc = np.mean(errors)

    ideal_coverage, ideal_risk, _ = rc_curve(-errors, errors, ideal=True)
    ideal_aurc = auc(ideal_coverage, ideal_risk)
    for name, confidence in confidences.items():
        coverage, risk, _ = rc_curve(confidence, errors)
        aurc = auc(coverage, risk)
        naurc = (aurc - ideal_aurc) / (random_aurc - ideal_aurc)
        naurcs[name] = naurc
    return naurcs

def calculating_aurc_from_dataframe(dataframe: pd.DataFrame, risks =['dice risk', 'ndice risk','hd95 risk']):
    ideal_coverage = {}
    ideal_risk = {}
    aurcs = {}
    list_all_risks = []
    errors = {name:dataframe[name] for name in risks}
    for c in dataframe.columns:
        if c.endswith(' risk'):
            list_all_risks.append(c)
    confidences =  dataframe.drop(list_all_risks, axis=1)
    errors = {name:dataframe[name] for name in risks}
    random_aurc = {name: np.mean(errors[name]) for name in errors.keys()}

    for name in risks:
        ideal_coverage[name], ideal_risk[name], _ = rc_curve(-errors[name], errors[name], ideal=True)
        ideal_aurc = auc(ideal_coverage[name], ideal_risk[name])
        aurcs[f'Random_{name}'] = random_aurc[name]
        aurcs[f'Ideal_{name}'] = ideal_aurc
        for name_conf, confidence in confidences.items():
            coverage, risk, _ = rc_curve(confidence, errors[name])
            aurc = auc(coverage, risk)
            aurcs[name_conf+'_'+name] = aurc
    return aurcs



def plot_segmentation_performance_report_UAI(y, y_hat,name, threshold=0.5):


    dices = list(map(
        lambda ys_i: 1-dice_coef(ys_i[0].flatten(), ys_i[1].flatten(), threshold=threshold),
        zip(y_hat,y)
    ))
    
    accs = list(map(
        lambda ys_i: 1-accuracy(ys_i[0].flatten(), ys_i[1].flatten(), threshold=threshold),
        zip(y_hat,y)
    ))


    def plot_performance_hist_acc(values, ax):
        ax.hist(values, bins=20)
        ylims = ax.get_ylim()
        mean_performance = np.mean(values)
        ax.vlines(mean_performance, *ylims, color='red', label=f"{mean_performance:.4E}")
        ax.set_ylim(*ylims)

        return ax
    
    def plot_performance_hist(values, ax):
        ax.hist(values, bins=20)
        ylims = ax.get_ylim()
        mean_performance = np.mean(values)
        ax.vlines(mean_performance, *ylims, color='red', label=f"{mean_performance:.2f}")
        ax.set_ylim(*ylims)

        return ax
    
    fig1 = plt.figure(figsize=(8, 5))
    ax_acc = fig1.add_subplot(111)

    ax_acc = plot_performance_hist_acc(accs, ax_acc)
    ax_acc.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax_acc.set_xlabel('Accuracy risk')
    ax_acc.set_ylabel('Frequency')
    ax_acc.legend()

    fig2 = plt.figure(figsize=(8, 5))
    ax_dice = fig2.add_subplot(111)

    ax_dice = plot_performance_hist(dices, ax_dice)
    ax_dice.set_xlabel('Dice risk')
    ax_dice.set_ylabel('Frequency')
    ax_dice.legend()

    fig3 = plt.figure(figsize=(8, 5))
    ax_gt_size = fig3.add_subplot(111)
    gt_lesion_load = np.array([yi.sum() / len(yi.flatten()) for yi in y_hat])

    ax_gt_size.hist(gt_lesion_load, bins=20)
    ylims = ax_gt_size.get_ylim()
    mean_performance = np.mean(gt_lesion_load)
    std_performance = np.std(gt_lesion_load, ddof=1)
    ax_gt_size.vlines(mean_performance, *ylims, color='red', label=f"Mean:{mean_performance:.5f}, Std:{std_performance:.5f}")
    ax_gt_size.set_xlabel('Ground truth size')
    ax_gt_size.set_ylabel('Frequency')
    ax_gt_size.legend()

    fig4 = plt.figure(figsize=(8, 5))
    ax_probs = fig4.add_subplot(111)
    probs = np.concatenate([y_hat_i.flatten() for y_hat_i in y_hat], axis=None)

    ax_probs.hist(probs, bins=20)
    ax_probs.set_yscale('log')
    ax_probs.set_xlabel('y_hat probability')
    ax_probs.set_ylabel('Frequency')

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    return fig1, fig2, fig3, fig4, mean_performance

def plot_max_baselines(errors, ax, **kwargs):
    ax.hlines(np.max(errors), 0, 1, colors='gray', linestyles='dashed', **kwargs)

    coverages, risks, _ = rc_curve_max(-errors, errors, expert=False, ideal=True)

    ax.plot(coverages, risks, linestyle='dashed', c='gray', **kwargs)

    return ax

def plot_rc_max_curve(confidence, errors, label, ax, low_aurc=0, high_aurc=None,
                  **kwargs):
    coverages, risks, _ = rc_curve_max(confidence, errors)

    aurc = auc(coverages, risks)
    aurc -= low_aurc
    
    if high_aurc is not None:
        high_aurc -= low_aurc
        aurc = aurc / high_aurc
     
    ax.plot(coverages, risks, label=f"{label}")

def plot_max_aurc_curves(confidences: dict, errors: np.array, ax):
    random_aurc = np.max(errors)

    ideal_coverage, ideal_risk, _ = rc_curve_max(-errors, errors, ideal=True)
    ideal_aurc = auc(ideal_coverage, ideal_risk)

    ax = plot_max_baselines(errors, ax)

    for name, confidence in confidences.items():
        plot_rc_max_curve(confidence, errors, name, ax)

    ax.set_xlim(0,1)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid()
    ax.legend()

    return ax

def thresh_alpha(p_hat, y_hat=None, threshold=0.5):
    # Case 0: If y_hat is provided and is a numpy array, compute with y_hat only
    if y_hat is not None and isinstance(y_hat, np.ndarray):
        return np.mean()
    
    # Case 1: y_hat is a list of numpy arrays
    elif y_hat is not None and isinstance(y_hat, list):
        threshold_alpha = []
        for y_hat_i in zip(y_hat):
            arr = np.asarray(y_hat_i)
            threshold_alpha.append(np.mean(arr))
        return np.mean(threshold_alpha)
    
    # Case 2: p_hat is a single numpy array
    elif y_hat is None and isinstance(p_hat, np.ndarray):
        return np.mean(p_hat > threshold)
    
    # Case 3: p_hat is a list (possibly of arrays of different shapes)
    elif y_hat is None and isinstance(p_hat, list):
        threshold_alpha = []
        for p_hat_i in p_hat:
            arr = np.asarray(p_hat_i)
            threshold_alpha.append(np.mean(arr > threshold))
        return np.mean(threshold_alpha)
    
    else:
        raise TypeError("p_hat must be a numpy array or a list of arrays")
    
def adjust_axis_limits(random, ideal=0, visual_position_random=0.1, visual_position_ideal=0.0):
    # Calculate the data range needed to satisfy the visual positions
    y_min = ideal - visual_position_ideal * (random - ideal)
    y_max = random + visual_position_random * (random - ideal)
    return y_min, y_max

def linear_interpolation(x, x0, y0, x1, y1):
    """
    Performs linear interpolation for the value x between the points (x0, y0) and (x1, y1).

    Parameters:
        x  : The x value to interpolate for (must be between x0 and x1).
        x0 : The x value of the first point.
        y0 : The y value of the first point.
        x1 : The x value of the second point.
        y1 : The y value of the second point.

    Returns:
        y  : The interpolated value corresponding to x.
    """
    if x1 - x0 == 0:
        raise ValueError("The values of x0 and x1 cannot be equal.")

    return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0)

def find_closest_indices(arr, target):
    """
    Given an array and a target number, this function returns a tuple:
    (index_of_closest_lower, index_of_closest_higher)
    
    - index_of_closest_lower: The index in the original array of the largest number less than target.
    - index_of_closest_higher: The index in the original array of the smallest number greater than target.
    
    If either does not exist, None is returned in its place.
    """
    # Create a list of tuples (value, original_index)
    arr_with_indices = [(value, idx) for idx, value in enumerate(arr)]
    
    # Sort the array by value while keeping track of original indices.
    arr_sorted = sorted(arr_with_indices, key=lambda x: x[0])
    # Create a separate list of sorted values for binary search.
    sorted_values = [pair[0] for pair in arr_sorted]
    
    # Find the insertion point for the target.
    idx = bisect.bisect_left(sorted_values, target)
    
    lower_index = None
    higher_index = None
    
    # Find the closest number less than the target.
    if idx > 0:
        lower_index = arr_sorted[idx - 1][1]
    
    # Find the closest number greater than the target.
    if idx < len(arr_sorted):
        # If target is exactly present, we take the next element.
        if sorted_values[idx] == target:
            if idx + 1 < len(arr_sorted):
                higher_index = arr_sorted[idx + 1][1]
        else:
            higher_index = arr_sorted[idx][1]
    
    return lower_index, higher_index

def uncertainty_error_overlap(
    y: ArrayLike,
    p_hat: ArrayLike,
    uncertainty: ArrayLike,
    T: float = 0.5,
    thresholds_u: Union[None, np.ndarray, Sequence[float]] = None,
    eps: float = 0.0,
) -> float:
    """
    Find the uncertainty threshold (in `thresholds_u`) that MINIMIZES the mean Dice overlap
    between (uncertainty >= thresh) and the error mask E = XOR(y, (p_hat > T)).

    Accepts either:
      - Single arrays: y, p_hat, uncertainty with the same shape (N, ...), OR
      - Lists/tuples of arrays: y[i], p_hat[i], uncertainty[i] with matching shapes per i,
        but different shapes across i are allowed.

    Returns:
      best_threshold : float
    """
    if thresholds_u is None:
        thresholds_u = np.arange(0.05, 1.0, 0.05, dtype=np.float64)
    else:
        thresholds_u = np.asarray(thresholds_u, dtype=np.float64)

    # Fast vectorized path for single-array inputs (original behavior)
    if isinstance(y, np.ndarray) and isinstance(p_hat, np.ndarray) and isinstance(uncertainty, np.ndarray):
        y_bool = y.astype(bool)
        pred = (p_hat > T)
        E = np.logical_xor(y_bool, pred)  # (N, ...)

        th = thresholds_u.reshape((-1,) + (1,) * uncertainty.ndim)     # (K, 1, 1, ...)
        U_bin = (uncertainty[None, ...] >= th)                         # (K, N, ...)
        E_exp = E[None, ...]                                           # (K, N, ...)

        # Sum over all non-(K,N) axes
        spatial_axes = tuple(range(2, U_bin.ndim))
        inter = np.sum(U_bin & E_exp, axis=spatial_axes, dtype=np.float64)      # (K, N)
        u_sum = np.sum(U_bin, axis=spatial_axes, dtype=np.float64)              # (K, N)
        e_sum = np.sum(E_exp, axis=spatial_axes, dtype=np.float64)              # (K, N)

        dice_kn = (2.0 * inter) / (u_sum + e_sum + eps)                         # (K, N)
        mean_dice_k = dice_kn.mean(axis=1)                                      # (K,)
        best_idx = int(np.argmin(mean_dice_k))
        return float(thresholds_u[best_idx])

    # Flexible path for lists/tuples of arrays with varying shapes
    if not (_is_list_of_arrays(y) and _is_list_of_arrays(p_hat) and _is_list_of_arrays(uncertainty)):
        raise ValueError("y, p_hat, and uncertainty must each be either a numpy array or a list/tuple of numpy arrays.")

    if not (len(y) == len(p_hat) == len(uncertainty)):
        raise ValueError("When passing lists/tuples, y, p_hat, and uncertainty must have the same length.")

    K = thresholds_u.shape[0]
    N = len(y)
    # We'll accumulate mean over samples: mean_dice_k = (1/N) * sum_i dice_k_i
    mean_dice_k = np.zeros(K, dtype=np.float64)

    for i, (yi, pi, ui) in enumerate(zip(y, p_hat, uncertainty)):
        if not (isinstance(yi, np.ndarray) and isinstance(pi, np.ndarray) and isinstance(ui, np.ndarray)):
            raise ValueError(f"Sample {i} is not a numpy array in one of y/p_hat/uncertainty.")

        if not (yi.shape == pi.shape == ui.shape):
            raise ValueError(f"Shapes of y[{i}], p_hat[{i}], uncertainty[{i}] must match. Got {yi.shape}, {pi.shape}, {ui.shape}.")

        yi_bool = yi.astype(bool, copy=False)
        pred_i = (pi > T)
        Ei = np.logical_xor(yi_bool, pred_i)

        # Flatten and compute per-threshold statistics with broadcasting
        uflat = ui.ravel()
        Eflat = Ei.ravel()

        # (K, n_pixels)
        U_bin = (uflat[None, :] >= thresholds_u[:, None])
        # Per threshold counts
        inter = np.sum(U_bin & Eflat[None, :], axis=1, dtype=np.float64)    # (K,)
        u_sum = np.sum(U_bin, axis=1, dtype=np.float64)                     # (K,)
        e_sum = float(np.sum(Eflat, dtype=np.float64))                      # scalar reused

        dice_k_i = (2.0 * inter) / (u_sum + e_sum + eps)                    # (K,)
        mean_dice_k += dice_k_i / N

    best_idx = int(np.argmin(mean_dice_k))
    return float(thresholds_u[best_idx])

def feature_extractor_2D(msk,u_thresh=0.95):
    # Convert images to numpy arrays
    image_np = np.squeeze(msk)
    mask_np = np.squeeze(msk>u_thresh).astype(np.uint8)

    # Convert numpy arrays to SimpleITK images
    image_sitk = sitk.GetImageFromArray(image_np)
    mask_sitk = sitk.GetImageFromArray(mask_np)
    
    # Use PyRadiomics to extract features
    try:
        features_2D = radiomics.shape2D.RadiomicsShape2D(image_sitk, mask_sitk, binWidth=25)
        features_2D = features_2D.execute()
    except:
        features_2D = {}
    try:
        first_order = radiomics.firstorder.RadiomicsFirstOrder(image_sitk, mask_sitk, binWidth=25)
        first_order = first_order.execute()
    except:
        first_order = {}
    try:
        gray_level_co = radiomics.glcm.RadiomicsGLCM(image_sitk, mask_sitk, binWidth=25)
        gray_level_co = gray_level_co.execute() 
    except:
        gray_level_co = {}
    try:
        gray_level_run = radiomics.glrlm.RadiomicsGLRLM(image_sitk, mask_sitk, binWidth=25)
        gray_level_run = gray_level_run.execute()
    except:
        gray_level_run = {}
    try:
        gray_level_size = radiomics.glszm.RadiomicsGLSZM(image_sitk, mask_sitk, binWidth=25)
        gray_level_size = gray_level_size.execute()
    except:
        gray_level_size = {}
    try:
        gray_level_depedence = radiomics.gldm.RadiomicsGLDM(image_sitk, mask_sitk, binWidth=25)
        gray_level_depedence = gray_level_depedence.execute()
    except:
        gray_level_depedence = {}

    all_features = {**features_2D, **first_order, **gray_level_co, **gray_level_run, **gray_level_size, **gray_level_depedence}
    return all_features

def feature_extractor_3D(msk,u_thresh=0.95):
    
    img = np.squeeze(msk)
    mask_np = np.squeeze(msk>u_thresh).astype(np.uint8)
    
    # Convert numpy arrays to SimpleITK images
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask_np)

    # Use PyRadiomics to extract features

    features_3D = radiomics.shape.RadiomicsShape(image_sitk, mask_sitk, binWidth=25)
    features_3D = features_3D.execute()

    first_order = radiomics.firstorder.RadiomicsFirstOrder(image_sitk, mask_sitk, binWidth=25)
    first_order = first_order.execute()

    gray_level_co = radiomics.glcm.RadiomicsGLCM(image_sitk, mask_sitk, binWidth=25)
    gray_level_co = gray_level_co.execute() 

    gray_level_run = radiomics.glrlm.RadiomicsGLRLM(image_sitk, mask_sitk, binWidth=25)
    gray_level_run = gray_level_run.execute()

    gray_level_size = radiomics.glszm.RadiomicsGLSZM(image_sitk, mask_sitk, binWidth=25)
    gray_level_size = gray_level_size.execute()

    gray_level_depedence = radiomics.gldm.RadiomicsGLDM(image_sitk, mask_sitk, binWidth=25)
    gray_level_depedence = gray_level_depedence.execute()


    all_features = {**features_3D, **first_order, **gray_level_co, **gray_level_run, **gray_level_size, **gray_level_depedence}
    return all_features

def get_tuning_size(n, num_points=16):
    max_points = int(np.floor(n/2) - 1)
    num_points = min(max_points, num_points)
    k = num_points
    while True:
        tuning_prop = np.geomspace(2/n, 0.5, k)
        tuning_size = np.unique(np.round(tuning_prop*n)).astype(int)
        if len(tuning_size) >= num_points:
            assert len(tuning_size) == num_points
            break
        k += 1
    return tuning_size