import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

def mape(pred, target, threshold=0.5, soft=False):
    gt = target.astype("float32")
    if soft:
        seg = pred
    else:
        seg = pred > threshold

    v = np.sum(gt)
    v_hat = np.sum(seg)
    
    return np.absolute(v_hat-v)/v

def sme(pred, target, threshold=0.5, soft=False):
    gt = target.astype("float32")
    if soft:
        seg = pred
    else:
        seg = pred > threshold

    v = np.sum(gt)
    v_hat = np.sum(seg)
    
    return (v_hat-v)**2

def mae(pred, target, threshold=0.5, soft=False):
    gt = target.astype("float32")
    if soft:
        seg = pred
    else:
        seg = pred > threshold

    v = np.sum(gt)
    v_hat = np.sum(seg)
    
    return np.absolute(v_hat-v)

def accuracy(pred, target, threshold=0.5):
    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred > threshold

    tp_tn = np.sum(gt == seg)

    return tp_tn/len(gt.flatten())

def balanced_accuracy(pred, target, threshold=0.5):
    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred > threshold
    tp = np.sum(seg[gt == 1])
    fp = np.sum(seg[gt == 0])
    fn = np.sum(gt[seg == 0])
    tn = len(gt.flatten()) -tp -fp -fn
    return (tp/(tp+fn) + tn/(tn+fp))/2

def soft_dice_b_metric(pred, target, r=0.0783):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      target: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      pred:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """
    smooth = 1.

    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred.astype("float32")

    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg * gt)
        fp = np.sum(seg * (1 - gt))
        fn = np.sum((1 - seg) * gt)
        fp_scaled = 1/k * fp
        dsc_norm = (2. * tp + smooth) / (fp_scaled + 2. * tp + fn + smooth)

        return dsc_norm
    
def soft_g_dice_metric(pred, target,  r1 = 0.076, r2= 0.076, gamma1=1, gamma2=1):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      target: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      pred:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """
    smooth = 1.

    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred.astype("float32")

    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k1 = 1.0
            k2 = 1.0
        else:
            k1 = (1 - r1) * np.sum(gt) / (r1 * (len(gt.flatten()) - np.sum(gt)))
            k2 = (1 - r2) * np.sum(gt) / (r2 * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg * gt)
        fp = np.sum(seg * (1 - gt))
        fn = np.sum((1 - seg) * gt)
        fp_scaled = (k1**gamma1)* fp
        fn_scaled = (k2**gamma2)*fn
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn_scaled)
        return dsc_norm
    
def g_dice_metric(predictions, ground_truth, r1 = 0.076, r2= 0.076, gamma1=1, gamma2=1, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k1 = 1.0
            k2 = 1.0
        else:
            k1 = (1 - r1) * np.sum(gt) / (r1 * (len(gt.flatten()) - np.sum(gt)))
            k2 = (1 - r2) * np.sum(gt) / (r2 * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = (k1**gamma1)* fp
        fn_scaled = (k2**gamma2)*fn
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn_scaled)
        return dsc_norm
    
def ablation_1(predictions, ground_truth, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k1 = 1.0
            k2 = 1.0
        else:
            k1 =  np.sum(gt)/(len(gt.flatten()) - np.sum(gt))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = (k1)* fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm
    
def ablation_2(predictions, ground_truth, r1 = 0.076, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k1 = 1.0
            k2 = 1.0
        else:
            k1 = (1 - r1) * np.sum(gt) / (r1 * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = (k1)* fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm 
    
def ablation_3(predictions, ground_truth, r1 = 0.076, gamma1=1, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k1 = 1.0
            k2 = 1.0
        else:
            k1 = (1 - r1) * np.sum(gt) / (r1 * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = (k1**gamma1)* fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm    
    
def ablation_4(predictions, ground_truth, r1 = 0.076, gamma1=1, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k1 = 1.0
            k2 = 1.0
        else:
            k1 = (1 - r1) * np.sum(gt) / (r1 * (len(gt.flatten()) - np.sum(gt)))
            k2 =  np.sum(gt) / (len(gt.flatten()) - np.sum(gt))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = (k1**gamma1)* fp
        fn_scaled = (k2)*fn
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn_scaled)
        return dsc_norm   
    
def ablation_5(predictions, ground_truth, r1 = 0.076, r2= 0.076, gamma1=1, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k1 = 1.0
            k2 = 1.0
        else:
            k1 = (1 - r1) * np.sum(gt) / (r1 * (len(gt.flatten()) - np.sum(gt)))
            k2 = (1 - r2) * np.sum(gt) / (r2 * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = (k1**gamma1)* fp
        fn_scaled = (k2)*fn
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn_scaled)
        return dsc_norm 
    
def new_form_g_dice_metric(pred, target, alpha=1, gamma1=0, gamma2=0):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    smooth = 1.

    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k1 = 1.0
            k2 = 1.0

        else:
            k1 = alpha*((np.sum(gt) /(len(gt.flatten()) - np.sum(gt)))**gamma1)
            k2 = (1-alpha)*((np.sum(gt) /(len(gt.flatten()) - np.sum(gt)))**gamma2)
        tp = np.sum(seg * gt)
        fp = np.sum(seg * (1 - gt))
        fn = np.sum((1 - seg) * gt)
        fp_scaled = k1*fp
        fn_scaled = k2*fn
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn_scaled)
        return dsc_norm
    
def new_form_g_dice_metric_weighted_lesion(pred, target, alpha=1, beta=1):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    smooth = 1.

    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            h = 1.0

        else:
            h = (np.sum(gt) /(len(gt.flatten()) - np.sum(gt)))
        
            
        tp = np.sum(seg * gt)
        fp = np.sum(seg * (1 - gt))
        fn = np.sum((1 - seg) * gt)
        fp_scaled = alpha*fp
        fn_scaled = (1-alpha)*fn
        dsc_norm = beta*2. * tp / (fp_scaled + 2. * tp + fn_scaled) + (1-beta)*h
        return dsc_norm

def new_form_g_dice_metric_geometric_lesion(pred, target, alpha=1, beta=1):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    smooth = 1.

    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            h = 1.0

        else:
            h = (np.sum(gt) /(len(gt.flatten()) - np.sum(gt)))
            
            
        tp = np.sum(seg * gt)
        fp = np.sum(seg * (1 - gt))
        fn = np.sum((1 - seg) * gt)
        fp_scaled = alpha*fp
        fn_scaled = (1-alpha)*fn
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn_scaled)*(h**beta)
        return dsc_norm
    
def dice_bias_metric(predictions, ground_truth, r = 0.076, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = 1/k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm

def sigmoid(x, alpha, beta, gamma):
    return gamma/ (1 + np.exp(-alpha*x+beta))

def dice_coef(pred, target, threshold=0.5):
    # Gurantee that both predictions and target are binaries
    pred = pred>threshold
    target = target>threshold
    num = 2 * (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 0
    else:
        return num / denom

def dice_norm_metric(predictions, ground_truth, r = 0.076, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == len(gt.flatten()):
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm

def soft_dice(pred, target, eps=0):
    eps = eps
    intersection = np.sum(pred*target)   
    ratio = (np.sum(pred) + np.sum(target) + eps)  
    if ratio  == 0.0:
        return 0.0
    else:
        dice = (2.*intersection + eps)/ratio  
        return dice

def soft_dice_norm_metric(pred, target, r=0.0783):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      target: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      pred:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """
    smooth = 1.

    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred.astype("float32")

    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 0.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg * gt)
        fp = np.sum(seg * (1 - gt))
        fn = np.sum((1 - seg) * gt)
        fp_scaled = k * fp
        dsc_norm = (2. * tp + smooth) / (fp_scaled + 2. * tp + fn + smooth)

        return dsc_norm

# function to compute the retention curve
def compute_retention_curve(confidence: np.ndarray, dices: np.ndarray):
    ordering = confidence.argsort()

    scores_ = confidence[ordering]
    dices_ = dices[ordering]

    retention_percentage = list()
    retention_score = list()
    for i in range(len(scores_) + 1):
        retention_percentage.append(i / len(scores_))
        ret_dices_ = np.ones_like(dices_)
        ret_dices_[:i] = dices_[:i]
        retention_score.append(np.mean(ret_dices_))

    return retention_percentage, retention_score

def rc_curve(confidence, error, expert=False, expert_cost=0, ideal=False, num_points=1001):
    error = np.array(error).reshape(-1)
    confidence = np.array(confidence).reshape(-1)
    n = len(error)
    assert len(confidence) == n
    desc_sort_indices = confidence.argsort()[::-1]
    error = error[desc_sort_indices]
    confidence = confidence[desc_sort_indices]
    idx = np.r_[np.where(np.diff(confidence))[0], n-1]
    thresholds = confidence[idx]
    coverages = (1 + idx)/n
    risks = np.cumsum(error)[idx]/n
    if expert:
        if np.any(expert_cost):
            expert_cost = np.array(expert_cost).reshape(-1)
            if expert_cost.size == 1:
                risks += (1 - coverages)*expert_cost
            else:
                assert len(expert_cost) == n
                expert_cost = np.cumsum(expert_cost)
                expert_cost = expert_cost[-1] - expert_cost
                risks += expert_cost[idx]/n
    else:
        risks /= coverages
        
    #if not ideal:
    #    coverages = np.insert(coverages, 0, 0)
    #    risks = np.insert(risks, 0, risks[0])
    #    thresholds = np.insert(thresholds, 0, 0)
    #elif ideal==True:
    #    coverages = np.insert(coverages, 0, 0)
    #    risks = np.insert(risks, 0, 0)
    #    thresholds = np.insert(thresholds, 0, 0)
    coverages = np.insert(coverages, 0, 0)
    risks = np.insert(risks, 0, risks[0])
    thresholds = np.insert(thresholds, 0, 0)
    return coverages, risks, thresholds

    '''error = np.array(error).reshape(-1)
    confidence = np.array(confidence).reshape(-1)
    n = len(error)
    assert len(confidence) == n

    # Sort errors and confidences in descending order of confidence
    desc_sort_indices = confidence.argsort()[::-1]
    error = error[desc_sort_indices]
    confidence = confidence[desc_sort_indices]

    # Find indices where confidence changes
    idx = np.r_[np.where(np.diff(confidence))[0], n - 1]
    thresholds = confidence[idx]

    # Calculate coverage
    coverages = (1 + idx) / n

    # Calculate risks without dividing by n (total error vs. coverage)
    risks = np.cumsum(error)[idx]

    # Include the point (0,0) in the curve
    risks = np.r_[0, risks]
    coverages = np.r_[0, coverages]

    # Perform linear interpolation to the desired number of points
    coverage_interp = np.linspace(0, coverages[-1], num_points)
    risks_interp = np.interp(coverage_interp, coverages, risks)

    # Safely divide risks by (n * coverage) without dividing by zero
    risks_interp = np.divide(
        risks_interp,
        n * coverage_interp,
        out=np.zeros_like(risks_interp),
        where=coverage_interp != 0
    )

    return coverage_interp, risks_interp, thresholds'''


def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def rc_curve_max(confidence, error, expert=False, expert_cost=0, ideal=False):
    error = np.array(error).reshape(-1)
    confidence = np.array(confidence).reshape(-1)
    n = len(error)
    assert len(confidence) == n
    desc_sort_indices = confidence.argsort()[::-1]
    error = error[desc_sort_indices]
    confidence = confidence[desc_sort_indices]
    idx = np.r_[np.where(np.diff(confidence))[0], n-1]
    thresholds = confidence[idx]
    coverages = (1 + idx)/n
    risks = np.maximum.accumulate(error)[idx]
    if expert:
        if np.any(expert_cost):
            expert_cost = np.array(expert_cost).reshape(-1)
            if expert_cost.size == 1:
                risks += (1 - coverages)*expert_cost
            else:
                assert len(expert_cost) == n
                expert_cost = np.maximum.accumulate(expert_cost)
                expert_cost = expert_cost[-1] - expert_cost
                risks += expert_cost[idx]/n
    else:
        risks
        
    #if len(risks)!=n and not ideal:
    if not ideal:
        coverages = np.insert(coverages, 0, 0)
        risks = np.insert(risks, 0, risks[0])
        thresholds = np.insert(thresholds, 0, 0)
    elif ideal==True:
        coverages = np.insert(coverages, 0, 0)
        risks = np.insert(risks, 0, 0)
        thresholds = np.insert(thresholds, 0, 0)
    return coverages, risks, thresholds