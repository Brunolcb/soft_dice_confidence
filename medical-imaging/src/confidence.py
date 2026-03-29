from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np
from scipy.special import softmax, logit, expit
from src.utils import get_noise_with_Lcov
from src.metrics import dice_norm_metric, dice_coef, soft_dice, sigmoid, soft_dice_norm_metric, hd95, dice_bias_metric, soft_dice_b_metric, soft_g_dice_metric, g_dice_metric, ablation_1, ablation_2, ablation_3, ablation_4, ablation_5, new_form_g_dice_metric, new_form_g_dice_metric_weighted_lesion, new_form_g_dice_metric_geometric_lesion, mae, mape
from multiprocessing import Pool
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from scipy.signal import convolve
import cv2 as cv
from torch import einsum
from typing import Optional
from skimage import segmentation as skimage_seg
import SimpleITK as sitk
import ants

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"
EPS: float = 1e-10

def _add_independent_noise_get_dice(args):
    p, sigma = args
    p_eta = expit(
        logit(p) + np.random.normal(0, sigma ,p.shape)
    )
    return dice_coef(p_eta, p)

def _add_noise_get_dice(args):
    p, sigma = args
    p_eta = expit(
        logit(p) + np.random.normal(0, sigma) * np.ones_like(p)
    )
    return dice_coef(p_eta, p)

class SegmentationConfidence(ABC):
    """Apply confidence metric to image.
    """
    @abstractmethod
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """

    def __call__(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        if probs.shape[0] == 1:  # binary classification
            background = 1. - probs[0]
            probs = np.stack([background, probs[0]], axis=0)

        return self.metric(probs)

class SegmentationConfidenceWithY_hat(ABC):
    """Apply confidence metric to image.
    """
    @abstractmethod
    def metric(self, probs: np.array, y_hat: np.array = None) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """

    def __call__(self, probs: np.array, y_hat: np.array = None) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        if probs.shape[0] == 1:  # binary classification
            background = 1. - probs[0]
            probs = np.stack([background, probs[0]], axis=0)

        return self.metric(probs, y_hat=y_hat)    

    
class SegmentationConfidenceWithNoise(ABC):
    """Apply uncertainty metric to image.
    """
    @abstractmethod
    def metric(self, probs: np.array, probs_with_noise: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """

    def __call__(self, probs: np.array, probs_with_noise: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        if probs.shape[0] == 1:  # binary classification
            background = 1. - probs[0]
            probs = np.stack([background, probs[0]], axis=0)
            
        if probs_with_noise.shape[0] == 1:  # binary classification
            background_with_noise = 1. - probs_with_noise[0]
            probs_with_noise = np.stack([background_with_noise, probs_with_noise[0]], axis=0)
            
        return self.metric(probs, probs_with_noise)

class MeanMaxConfidence(SegmentationConfidence):
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        confidence = np.max(probs, axis=0)
        return np.mean(confidence)

class ForegroundMeanMaxConfidence(SegmentationConfidence):
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        confidence = np.max(probs, axis=0)
        confidence *= p >= .5

        return np.mean(confidence)

class ExpectedEntropy(SegmentationConfidence):
    def metric(self, probs: np.array, epsilon=1e-9) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        log_probs = np.log(probs + epsilon)
        exe = -np.sum(probs * log_probs, axis=0)  # sums over classes

        return -np.mean(exe)

class PredSize(SegmentationConfidence):
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        assert probs.shape[0] == 2

        pred = np.argmax(probs, axis=0)

        return np.sum(pred)

class PredSizeAtThreshold(SegmentationConfidence):
    def __init__(self, threshold=.99):
        super().__init__()
        self.threshold = threshold
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        return np.sum(p > self.threshold)

class PredSizeChange(SegmentationConfidence):
    def __init__(self, thresholds=(.05, .95)):
        super().__init__()
        self.thresholds = thresholds
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        pred_low = p >= self.thresholds[0]
        pred_high = p >= self.thresholds[1]

        return -np.sum(pred_low ^ pred_high)

class PredChangenDSC(SegmentationConfidence):
    def __init__(self, thresholds=(.1, .9)):
        super().__init__()
        self.thresholds = thresholds
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        pred_low = p >= self.thresholds[0]
        pred_high = p >= self.thresholds[1]

        return dice_norm_metric(pred_low, pred_high)

class PredChangeDSC(SegmentationConfidence):
    def __init__(self, thresholds=(.1, .9)):
        super().__init__()
        self.thresholds = thresholds
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        pred_low = p >= self.thresholds[0]
        pred_high = p >= self.thresholds[1]

        return dice_coef(pred_low, pred_high)
    

        
class nDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, r=0.0764):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.r = r
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        if np.sum(pred)>0:
            ndscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                ndscs += dice_norm_metric(p > threshold, pred, r =self.r)

            return ndscs
        else:
            return -np.inf

class DSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += dice_coef(p > threshold, pred)

            return dscs
        else:
            return -np.inf

    
class SoftDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        dscs = 0
        for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
            dscs += soft_dice(p,p > threshold)

        return dscs

class NewDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += dice_coef(p > threshold, pred)
        else:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0]/2, self.threshold_lims[1]/2 + self.step/2, self.step/2):
                dscs += dice_coef(p > threshold, pred)
        return dscs

class BinaryCrossEntropy(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        p = np.sum(probs[1:], axis=0) 
        y_true = p > self.threshold_lim
        y_pred = np.clip(p, 1e-7, 1 - 1e-7)
        term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
        term_1 = y_true * np.log(y_pred + 1e-7)
        
        return np.mean(term_0+term_1)

class FocalLoss(SegmentationConfidence):
    def __init__(self, threshold_lim=.5, gamma=2):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.gamma =gamma
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim
        eps = 1e-12
        # If actual value is true, keep pt value as y_pred otherwise (1-y_pred)
        pt = np.where(y_true == 1, y_pred, 1-y_pred)
        # Clip values below epsilon and above 1-epsilon
        pt = np.clip(pt, eps, 1-eps)
        # FL = -(1-pt)^gamma log(pt)
        focal_loss = -np.mean(np.multiply(np.power(1-pt,self.gamma),np.log(pt)))    
        return -focal_loss   
    
class SoftDice(SegmentationConfidenceWithY_hat):
    def __init__(self, threshold_lim=.5,eps=0):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.eps=eps
    def metric(self, probs: np.array, y_hat=None) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        if y_hat is None:
            y_true = y_pred > self.threshold_lim        
        else: 
            y_true = y_hat       
        dice = soft_dice(y_pred,y_true,eps=self.eps)  
        return dice   

class SoftDiceWithNoise(SegmentationConfidenceWithNoise):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array, probs_with_noise: np.array) -> float:
        y_true = np.sum(probs[1:], axis=0) 
        y_true = y_true > self.threshold_lim
        y_pred = np.sum(probs_with_noise[1:], axis=0)
        dice = soft_dice(y_pred,y_true)  
        return dice  
    
class DSCIntegralOverThresholdWithNoise(SegmentationConfidenceWithNoise):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    
    def metric(self, probs: np.array, probs_with_noise: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pwn = np.sum(probs_with_noise[1:], axis=0)  # probability of foreground
        gt = p > .5

        dscs = 0
        for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
            dscs += dice_coef(pwn > threshold, gt)

        return dscs

class DSCIntegralOverMaxThreshold(SegmentationConfidence):
    def __init__(self, quant_step=20):
        super().__init__()
        self.quant_step = quant_step
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        thresh_lim = np.max(p)
        pred = p > .5

        dscs = []
        for threshold in np.linspace(0.0,thresh_lim,20)[1:]:
            dscs.append(dice_coef(p > threshold, pred))

        return np.mean(dscs)
    
class SoftlogDice(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim  
        y_predlog = np.where(y_pred>0.5,y_pred*(1+np.log(y_pred)),(y_pred)*(1+np.log(1-y_pred)))
        dice = soft_dice(y_predlog,y_true)  
        return dice   

class SoftDice_Var_Eps(SegmentationConfidence):
    def __init__(self, threshold_lim=.5, quant=10):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.quant = quant
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0)
        y_true = y_pred > self.threshold_lim
        if np.sum(y_true)>0:
            dice = soft_dice(y_pred,y_true) 
            return dice
        else:
            max_logit = np.max(y_pred)
            dice = soft_dice(y_pred,y_true, eps=(1-max_logit)*self.quant)
            return dice      

class DSCIntegralOverThreshold_bins(SegmentationConfidence):
    def __init__(self, threshold_lims_sup=np.array([0.25,0.5,0.75]), threshold_lims_inf=np.array([0.125,0.25,0.5])):
        super().__init__()
        self.threshold_lims_sup = threshold_lims_sup
        self.threshold_lims_inf = threshold_lims_inf

    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        dscs = 0
        if np.sum(pred)>0:
            for threshold in self.threshold_lims_sup:
                dscs += dice_coef(p > threshold, pred)
                return dscs
        else:
            for threshold in self.threshold_lims_inf:
                dscs += dice_coef(p > threshold, pred)
                return dscs

class Soft_Dice_and_DSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims_sup=np.array([0.25,0.5,0.75]), threshold_lims_inf=np.array([0.125,0.25,0.5])):
        super().__init__()
        self.threshold_lims_sup = threshold_lims_sup
        self.threshold_lims_inf = threshold_lims_inf

    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        dscs = 0
        if np.sum(pred)>0:
            for threshold in self.threshold_lims_sup:
                dscs += dice_coef(p > threshold, pred)
                return dscs
        else:
            for threshold in self.threshold_lims_inf:
                dscs += dice_coef(p > threshold, pred)
                return dscs
            
class Soft_Dice_and_DSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, threshold=.5,eps=0):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.eps = eps
        self.threshold = threshold
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > self.threshold 
        if np.sum(pred)>0:
            dscs = 0
            interval = np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step)
            for threshold in interval:
                dscs += dice_coef(p > threshold, pred)
            dscs = dscs/(interval.shape[0])
            return dscs
        else:
            dice = soft_dice(p,pred,eps=self.eps)
            return dice

class DIOT_Sigmoid(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, threshold=.5, alpha=1.0, beta= 0.0, gamma=1.0):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.alpha = alpha 
        self.beta = beta
        self.threshold = threshold
        self.gamma = gamma
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > self.threshold 
        if np.sum(pred)>0:
            dscs = 0
            interval = np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step)
            for threshold in interval:
                dscs += dice_coef(p > threshold, pred)
            dscs = dscs/(interval.shape[0])
            return dscs
        else:
            return sigmoid(np.sum(p), self.alpha, self.beta,self.gamma)
        
class ExpectedEntropy_With_If(SegmentationConfidence):
    def metric(self, probs: np.array, epsilon=1e-9) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > 0.5
        if np.sum(pred)>0:
            log_probs = np.log(probs + epsilon)
            exe = -np.sum(probs * log_probs, axis=0)  # sums over classes

            return -np.mean(exe)           
        else:
            return -np.inf
        
class PredChangeDSC_With_If(SegmentationConfidence):
    def __init__(self, thresholds=(.1, .9)):
        super().__init__()
        self.thresholds = thresholds
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > 0.5
        if np.sum(pred)>0:
            
            pred_low = p >= self.thresholds[0]
            pred_high = p >= self.thresholds[1]

            return dice_coef(pred_low, pred_high)   
        else:
            return -np.inf
        
class MeanMaxConfidence_With_If(SegmentationConfidence):
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > 0.5
        if np.sum(pred)>0:
            confidence = np.max(probs, axis=0)
            return np.mean(confidence)
        else:
            return -np.inf
        
class Lesion_Load_With_If(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim  
        if np.sum(y_true)>0:
            return np.sum(y_true)
        else:
            return -np.inf
        
class SoftDice_With_If(SegmentationConfidence):
    def __init__(self, threshold_lim=.5,eps=0):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.eps=eps
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0)
        y_true = y_pred > self.threshold_lim 
        if np.sum(y_true)>0:             
            dice = soft_dice(y_pred,y_true,eps=self.eps)  
            return dice    
        else:
            return -np.inf
        
class SoftnDice(SegmentationConfidence):
    def __init__(self, threshold_lim=.5,eps=0, r=0.0783):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.eps=eps
        self.r = r
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        ndice = soft_dice_norm_metric(y_pred,y_true,r=self.r)  
        return ndice  


class HD95IntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, threshold_lim=0.5):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.threshold_lim= threshold_lim
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > self.threshold_lim
        
        if np.sum(pred)>0:
            hd95_value = 0
            count =0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                if np.sum(p > threshold)>0:
                    hd95_value += hd95(p > threshold, pred)
                    count +=1
                
            return -hd95_value/count
        else:
            return -np.inf

        

class DSCGaussian_based_confidence(SegmentationConfidence):
    def __init__(self,  N=100, sigma=1):
        super().__init__()
        self.sigma = sigma
        self.N = N
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        
        if np.sum(pred)>0:
            dscs = 0
            for n in range(self.N):
                logits_ = logits + np.random.normal(0,self.sigma,p.shape)
                #logits_ = logits + np.random.normal(0,self.sigma)
                dscs += dice_coef(logits_ >0, pred)

            return dscs
        else:
            return -np.inf
        
class nDSCGaussian_based_confidence(SegmentationConfidence):
    def __init__(self,  N=100, sigma=1,  r = 0.076):
        super().__init__()
        self.sigma = sigma
        self.N = N
        self.r = r
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        if np.sum(pred)>0:
            dscs = 0
            for n in range(self.N):
                logits_ = logits + np.random.normal(0,self.sigma,p.shape)
                #logits_ = logits + np.random.normal(0,self.sigma)
                dscs += dice_norm_metric(logits_ >0, pred, r= self.r)

            return +dscs
        else:
            return -np.inf

class HD95Gaussian_based_confidence(SegmentationConfidence):
    def __init__(self,  N=100, sigma=1):
        super().__init__()
        self.sigma = sigma
        self.N = N
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        if np.sum(pred)>0:
            hd95_value = 0
            count = 0
            for n in range(self.N):
                logits_ = logits + np.random.normal(0,self.sigma,p.shape)
                #logits_ = logits + np.random.normal(0,self.sigma)
                if np.sum(logits_ >0)>0:
                    hd95_value += hd95(logits_ >0, pred)
                    count+=1
            return -hd95_value/count
        else:
            return np.inf       
        

class DSC_Cov_based_confidence(SegmentationConfidence):
    def __init__(self, eta):
        super().__init__()
        self.eta = eta
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        if np.sum(pred)>0:
            eta_min = np.expand_dims(np.min(self.eta, axis=(-1,-2)), axis=(1,2))
            eta_max = np.expand_dims(np.max(self.eta, axis=(-1,-2)), axis=(1,2))
            eta_nor =(self.eta-eta_min)/(eta_max-eta_min)
            logits_ = logits+eta_nor
            dscs = 0
            for logit_ in logits_:
                dscs += dice_coef(logit_>0, pred)

            return dscs
        else:
            return -np.inf
        
class DSC_Cov_based_confidence_diff_mag(SegmentationConfidence):
    def __init__(self, eta, alpha):
        super().__init__()
        self.eta = eta
        self.alpha = alpha
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        if np.sum(pred)>0:
            logits_ = logits+self.alpha*self.eta
            dscs = 0
            for logit_ in logits_:
                dscs += dice_coef(logit_>0, pred)

            return dscs
        else:
            return -np.inf

class DSC_Var_based_confidence(SegmentationConfidence):
    def __init__(self, var, alpha):
        super().__init__()
        self.var = var
        self.alpha = alpha
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        dscs = 0
        if np.sum(pred)>0:
            logits_ = logits+self.alpha*self.var
            dscs += dice_coef(logits_>0, pred)
            return dscs
        else:
            return -np.inf    
                

class nDSC_Cov_based_confidence(SegmentationConfidence):
    def __init__(self, eta,  r = 0.076):
        super().__init__()
        self.eta = eta
        self.r = r
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        if np.sum(pred)>0:
            eta_min = np.expand_dims(np.min(self.eta, axis=(-1,-2)), axis=(1,2))
            eta_max = np.expand_dims(np.max(self.eta, axis=(-1,-2)), axis=(1,2))
            eta_nor =(self.eta-eta_min)/(eta_max-eta_min)
            logits_ = logits+eta_nor
            dscs = 0
            for logit_ in logits_:
                dscs += dice_norm_metric(logit_ >0, pred, r= self.r)

            return dscs
        else:
            return -np.inf
    

class HD95_Cov_based_confidence(SegmentationConfidence):
    def __init__(self, eta):
        super().__init__()
        self.eta = eta
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        if np.sum(pred)>0:
            eta_min = np.expand_dims(np.min(self.eta, axis=(-1,-2)), axis=(1,2))
            eta_max = np.expand_dims(np.max(self.eta, axis=(-1,-2)), axis=(1,2))
            eta_nor =(self.eta-eta_min)/(eta_max-eta_min)
            logits_ = logits+eta_nor
            hd95_value = 0
            count = 0
            for logit_ in logits_:
                hd95_value += hd95(logit_ >0, pred)
                count+=1
            return -hd95_value/count
        else:
            return np.inf

class DSCIntegralOverIndependentNoise(SegmentationConfidence):
    def __init__(self, N: int, sigma: float) -> None:
        super().__init__()

        self.N = N
        self.sigma = sigma
    
    def metric(self, probs: np.array) -> float:
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        dscs = list()
        with Pool(10) as pool:
            dscs = pool.map(_add_independent_noise_get_dice, [(p, self.sigma),] * self.N)

        return np.mean(dscs)        

class DSCIntegralOverNoise(SegmentationConfidence):
    def __init__(self, N: int, sigma: float) -> None:
        super().__init__()

        self.N = N
        self.sigma = sigma
    
    def metric(self, probs: np.array) -> float:
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        dscs = list()
        with Pool(10) as pool:
            dscs = pool.map(_add_noise_get_dice, [(p, self.sigma),] * self.N)

        return np.mean(dscs)    

class bDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, r= 0.076):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.r =r
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += dice_bias_metric(p > threshold, pred,r=self.r)

            return dscs
        else:
            return -np.inf
        
class gDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, r1= 0.076, r2= 0.076, gamma1=1, gamma2=1):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.r1 =r1
        self.r2 =r2
        self.gamma1= gamma1
        self.gamma2= gamma2
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += g_dice_metric(p > threshold,pred,  r1=self.r1, r2=self.r2, gamma1=self.gamma1, gamma2=self.gamma2)

            return dscs
        else:
            return -np.inf     
        
        
class SoftgDice(SegmentationConfidence):
    def __init__(self, threshold_lim=.5,eps=0, r1= 0.076, r2= 0.076, gamma1=1, gamma2=1):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.eps=eps
        self.r1 =r1
        self.r2 =r2
        self.gamma1= gamma1
        self.gamma2= gamma2
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        ndice = soft_g_dice_metric(y_pred,y_true, r1=self.r1, r2=self.r2, gamma1=self.gamma1, gamma2=self.gamma2)  
        return ndice  
    
class SoftbDice(SegmentationConfidence):
    def __init__(self, threshold_lim=.5,eps=0, r=0.0783):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.eps=eps
        self.r = r
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        ndice = soft_dice_b_metric(y_pred,y_true,r=self.r)  
        return ndice 
    
class Lesion_Load(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim
        return np.sum(y_true)/len(y_true.flatten())
    
class SoftLesion_Load(SegmentationConfidence):
    def __init__(self):
        super().__init__()
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0)  
        return np.sum(y_pred)

class AblationgDIOT_1(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += ablation_1(p > threshold,pred)

            return dscs
        else:
            return -np.inf      
    
class AblationgDIOT_2(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, r1= 0.076):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.r1 =r1
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += ablation_2(p > threshold,pred,  r1=self.r1)

            return dscs
        else:
            return -np.inf    

class AblationgDIOT_3(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, r1= 0.076, gamma1=1):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.r1 =r1
        self.gamma1= gamma1
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += ablation_3(p > threshold,pred,  r1=self.r1, gamma1=self.gamma1)

            return dscs
        else:
            return -np.inf    
        
class AblationgDIOT_4(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, r1= 0.076, gamma1=1):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.r1 =r1
        self.gamma1= gamma1
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += ablation_4(p > threshold,pred,  r1=self.r1, gamma1=self.gamma1)

            return dscs
        else:
            return -np.inf    

class AblationgDIOT_5(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, r1= 0.076, r2= 0.076, gamma1=1):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.r1 =r1
        self.r2 =r2
        self.gamma1= gamma1
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += ablation_5(p > threshold,pred,  r1=self.r1, r2=self.r2, gamma1=self.gamma1)

            return dscs
        else:
            return -np.inf    
        
class DSC_Agreement_Noise(SegmentationConfidence):
    def __init__(self, noises, alpha,threshold=0.5):
        super().__init__()
        self.noises = noises
        self.threshold = threshold
        self.alpha = alpha
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        logits = logit(p)
        self.noises =self.noises.squeeze()
        if np.sum(pred)>0:
            dscs = 0
            for noise in self.noises:
                logits_ = logits + self.alpha*noise
                p_ = expit(logits_)
                dscs += dice_coef(p_ > self.threshold, pred)
            return dscs
        else:
            return -np.inf
        
class DSC_Agreement_Noise_with_Inf(SegmentationConfidenceWithNoise):
    def __init__(self,threshold=0.5):
        super().__init__()
        self.threshold = threshold     
    def metric(self, probs: np.array, probs_with_noise: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        logits_noise = np.sum(probs_with_noise[1:], axis=0).squeeze()
        pred = p > .5
        if np.sum(pred)>0:
            dscs = 0
            for noise in logits_noise:
                p_ = expit(noise)
                dscs += dice_coef(p_ > self.threshold, pred)
            return dscs
        else:
            return -np.inf
        
class new_form_gSoftDice(SegmentationConfidence):
    def __init__(self, alpha=1, gamma1=0, gamma2=0):
        super().__init__()
        self.alpha = alpha
        self.gamma1= gamma1
        self.gamma2= gamma2
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = new_form_g_dice_metric(p ,pred,  alpha=self.alpha, gamma1=self.gamma1, gamma2=self.gamma2)

            return dscs
        else:
            return -np.inf 

class new_form_wl_gSoftDice(SegmentationConfidence):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = new_form_g_dice_metric_weighted_lesion(p ,pred,  alpha=self.alpha, beta=self.beta)

            return dscs
        else:
            return -np.inf 
        
class new_form_geometric_gSoftDice(SegmentationConfidence):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = new_form_g_dice_metric_geometric_lesion(p ,pred,  alpha=self.alpha, beta=self.beta)

            return dscs
        else:
            return -np.inf 
        
class new_form_gDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold=0.5,threshold_lims=(.05, .95), step=.05, b1= 1, b2= 1, gamma1=0, gamma2=0):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.b1 =b1
        self.b2 =b2
        self.gamma1= gamma1
        self.gamma2= gamma2
        self.threshold=0.5
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += new_form_g_dice_metric(p > threshold,pred,  b1=self.b1, b2=self.b2, gamma1=self.gamma1, gamma2=self.gamma2)

            return dscs
        else:
            return -np.inf 


def entropy_of_expected(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_classes]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=0)

def entropy_of_expected_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(entropy_of_expected(probs, epsilon=epsilon))

def expected_entropy(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    log_probs = -np.log(probs + epsilon)
    return np.mean(np.sum(probs * log_probs, axis=1), axis=0)

def expected_entropy_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(expected_entropy(probs, epsilon=epsilon))

def max_prob_uncertainty(probs):
    """
    Maximum probability over classes.

    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: float
    """
    mean_probs = np.mean(probs, axis=0)

    if mean_probs.shape[0] == 1:  # binary classification, explicit background
                                  # probabilities
        background = 1. - mean_probs[0]
        mean_probs = np.stack([background, mean_probs[0]], axis=0)

    confidence = mean_probs.max(0)  # probability of the predicted class

    return -np.mean(confidence)

def max_confidence_uncertainty(probs):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return -np.mean(np.max(probs.reshape((probs.shape[0],-1)), axis=-1))

def max_softmax_uncertainty(probs):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return -np.mean(softmax(probs.reshape((probs.shape[0],-1)), axis=-1).max())

def mutual_information(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    exe = expected_entropy(probs, epsilon=epsilon)
    eoe = entropy_of_expected(probs, epsilon=epsilon)

    return eoe - exe

def mutual_information_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(mutual_information(probs, epsilon=epsilon))

def expected_pw_kl(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)

    exe = expected_entropy(probs, epsilon=epsilon)

    return -np.sum(mean_probs * mean_lprobs, axis=0) - exe

def expected_pw_kl_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(expected_pw_kl(probs, epsilon=epsilon))

def reverse_mutual_information(probs, epsilon=1e-10):
    epkl = expected_pw_kl(probs, epsilon=epsilon)
    mi = mutual_information(probs, epsilon=epsilon)

    return epkl - mi

def reverse_mutual_information_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(reverse_mutual_information(probs, epsilon=epsilon))

def naive_pred_size_uncertainty(probs):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: float
    """
    mean_probs = np.mean(probs, axis=0)
    return -np.sum(mean_probs > 0.5)

def ensemble_uncertainties_classification(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: Dictionary of uncertainties
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)

    mutual_info = eoe - exe

    epkl = -np.sum(mean_probs * mean_lprobs, axis=-1) - exe

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'epkl': epkl,
                   'reverse_mutual_information': epkl - mutual_info,
                   }

    return uncertainty






def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

class SSLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        Sensitivity-Specifity loss
        paper: http://www.rogertam.ca/Brosch_MICCAI_2015.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/df0f86733357fdc92bbc191c8fec0dcf49aa5499/niftynet/layer/loss_segmentation.py#L392
        """
        super(SSLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = 0.1 # weight parameter in SS paper

    def forward(self, net_output, gt, loss_mask=None):
        shp_x = net_output.shape
        shp_y = gt.shape
        # class_num = shp_x[1]
        
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.reshape((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        
        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - net_output)**2
        specificity_part = sum_tensor(squared_error*y_onehot, axes)/(sum_tensor(y_onehot, axes)+self.smooth)
        sensitivity_part = sum_tensor(squared_error*bg_onehot, axes)/(sum_tensor(bg_onehot, axes)+self.smooth)

        ss = self.r * specificity_part + (1-self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()

        return ss

class SSLossConfidence(SegmentationConfidence):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, threshold=0.5):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.square = square
        self.threshold = threshold
        self.ss = SSLoss(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth, square=square)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.ss(p, target).item()
        else:
            return -np.inf


class HausdorffDTConfidence(SegmentationConfidence):
    def __init__(self, alpha=2.0, threshold=0.5):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def distance_field(self, p: np.ndarray) -> np.ndarray:
        field = np.zeros_like(p)

        fg_mask = p > self.threshold

        if fg_mask.any():
            bg_mask = ~fg_mask

            fg_dist = edt(fg_mask)
            bg_dist = edt(bg_mask)

            field = fg_dist + bg_dist

        return field

    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > self.threshold
        if np.sum(pred)>0:
            pred_dt = self.distance_field(pred)
            p_dt = self.distance_field(p)
            pred_error = (pred - p) ** 2
            distance = pred_dt ** self.alpha + p_dt ** self.alpha
            dt_field = pred_error * distance
            loss = dt_field.mean()
            return -loss
            
        else:
            return -np.inf        

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.reshape(-1, num_class)
        target_flatten = target.reshape(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        #print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses
    
class LovaszSoftmaxConfidence(SegmentationConfidence):
    def __init__(self, reduction='mean', threshold=0.5):
        super().__init__()
        self.reduction = reduction
        self.lovasz = LovaszSoftmax(reduction=reduction)
        self.threshold = threshold
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.lovasz(p, target).item()
        else:
            return -np.inf

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22
        
        """
        super(IoULoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)


        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()

        return -iou

class IoULossConfidence(SegmentationConfidence):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, threshold=0.5):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.square = square
        self.threshold = threshold
        self.iou = IoULoss(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth, square=square)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.iou(p, target).item()
        else:
            return -np.inf
        
class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss
        
class HausdorffERLossConfidence(SegmentationConfidence):
    def __init__(self, alpha=2.0, erosions=10, threshold=0.5):
        super().__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.threshold = threshold
        self.hausdorff = HausdorffERLoss(alpha=alpha, erosions=erosions)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.hausdorff(p, target).item()
        else:
            return -np.inf

class CrossentropyND(torch.nn.BCELoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.reshape(-1,)

        return super(CrossentropyND, self).forward(inp, target)

class CrossentropyNDConfidence(SegmentationConfidence):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.ce = CrossentropyND()
        self.threshold = threshold
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.ce(p, target).item()
        else:
            return -np.inf
class TopKLoss(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.reshape((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

class TopKLossConfidence(SegmentationConfidence):
    def __init__(self, threshold=0.5, k=10):
        super().__init__()
        self.topk = TopKLoss(k=k)
        self.threshold = threshold
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.topk(p, target).item()
        else:
            return -np.inf
class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.reshape(-1, num_classes)

        target = target.reshape(-1,)
        wce_loss = torch.nn.BCELoss(weight=self.weight)

        return wce_loss(inp, target)

class WeightedCrossEntropyLossConfidence(SegmentationConfidence):
    def __init__(self, threshold=0.5, weight=None):
        super().__init__()
        self.wce = WeightedCrossEntropyLoss(weight=weight)
        self.threshold = threshold
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.wce(p, target).item()
        else:
            return -np.inf

class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, target, bound):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # print('net_output shape: ', net_output.shape)
        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = bound[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc)
        bd_loss = multipled.mean()

        return bd_loss

class BDLossConfidence(SegmentationConfidence):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.bd = BDLoss()
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.bd(p, target).item()
        else:
            return -np.inf
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.reshape((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        
        gt_temp = gt[:,0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy()>0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)
        
        tp = net_output * y_onehot
        tp = torch.sum(tp[:,1,...] * dist, (1,2,3))
        
        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:,1,...], (1,2,3)) + torch.sum(y_onehot[:,1,...], (1,2,3)) + self.smooth)

        dc = dc.mean()

        return -dc

class DistBinaryDiceLossConfidence(SegmentationConfidence):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.dice = DistBinaryDiceLoss()
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """

        target = probs[1:]  # probability of foreground
        if len(target.shape) == 4:
            p = np.expand_dims(probs, axis=0)
            target = np.expand_dims(target, axis=0)
        elif len(target.shape) == 3:
            p = np.expand_dims(probs, axis=0)
            p = np.expand_dims(p, axis=-1)
            target = np.expand_dims(target, axis=0)
            target = np.expand_dims(target, axis=-1)
        p = torch.from_numpy(p).float()
        target = torch.from_numpy(target).float()
        target = target>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.dice(p, target).item()
        else:
            return -np.inf

class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.reshape((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
    
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", net_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor =  - 2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc



def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

class GDiceLossConfidence(SegmentationConfidence):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.gdice = GDiceLoss()
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        if len(p.shape) == 4:
            p = np.expand_dims(p, axis=0)
        elif len(p.shape) == 3:
            p = np.expand_dims(p, axis=0)
            p = np.expand_dims(p, axis=-1)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.gdice(p, target).item()
        else:
            return -np.inf

class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)


        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky

class TverskyLossConfidence(SegmentationConfidence):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, threshold=0.5):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.square = square
        self.threshold = threshold
        self.tversky = TverskyLoss(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth, square=square)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.tversky(p, target).item()
        else:
            return -np.inf

'''class FocalTversky_loss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """
    def __init__(self, tversky_kwargs, gamma=0.75):
        super(FocalTversky_loss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, net_output, target):
        tversky_loss = 1 + self.tversky(net_output, target) # = 1-tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky

class FocalTversky_lossConfidence(SegmentationConfidence):
    def __init__(self, tversky_kwargs, gamma=0.75, threshold=0.5):
        super().__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.focal_tversky = FocalTversky_loss(tversky_kwargs, gamma=gamma)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.focal_tversky(p, target).item()
        else:
            return -np.inf'''

class AsymLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779
        """
        super(AsymLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.beta = 1.5

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)# shape: (batch size, class num)
        weight = (self.beta**2)/(1+self.beta**2)
        asym = (tp + self.smooth) / (tp + weight*fn + (1-weight)*fp + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                asym = asym[1:]
            else:
                asym = asym[:, 1:]
        asym = asym.mean()

        return -asym

class AsymLossConfidence(SegmentationConfidence):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, threshold=0.5):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.square = square
        self.threshold = threshold
        self.asym = AsymLoss(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth, square=square)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.asym(p, target).item()
        else:
            return -np.inf

'''class PenaltyGDiceLoss(nn.Module):
    """
    paper: https://openreview.net/forum?id=H1lTh8unKN
    """
    def __init__(self, gdice_kwargs):
        super(PenaltyGDiceLoss, self).__init__()
        self.k = 2.5
        self.gdc = GDiceLoss(apply_nonlin=softmax_helper, **gdice_kwargs)

    def forward(self, net_output, target):
        gdc_loss = self.gdc(net_output, target)
        penalty_gdc = gdc_loss / (1 + self.k * (1 - gdc_loss))

        return penalty_gdc

class PenaltyGDiceLossConfidence(SegmentationConfidence):
    def __init__(self, gdice_kwargs, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.penalty_gdc = PenaltyGDiceLoss(gdice_kwargs)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.penalty_gdc(p, target).item()
        else:
            return -np.inf'''

'''class ExpLog_loss(nn.Module):
    """
    paper: 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    https://arxiv.org/pdf/1809.00076.pdf
    """
    def __init__(self, soft_dice_kwargs, wce_kwargs, gamma=0.3):
        super(ExpLog_loss, self).__init__()
        self.wce = WeightedCrossEntropyLoss(**wce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.gamma = gamma

    def forward(self, net_output, target):
        dc_loss = -self.dc(net_output, target) # weight=0.8
        wce_loss = self.wce(net_output, target) # weight=0.2
        # with torch.no_grad():
        #     print('dc loss:', dc_loss.cpu().numpy(), 'ce loss:', ce_loss.cpu().numpy())
        #     a = torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma)
        #     b = torch.pow(-torch.log(torch.clamp(ce_loss, 1e-6)), self.gamma)
        #     print('ExpLog dc loss:', a.cpu().numpy(), 'ExpLogce loss:', b.cpu().numpy())
        #     print('*'*20)
        explog_loss = 0.8*torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma) + \
            0.2*wce_loss

        return explog_loss

class ExpLog_lossConfidence(SegmentationConfidence):
    def __init__(self, soft_dice_kwargs, wce_kwargs, gamma=0.3, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.explog = ExpLog_loss(soft_dice_kwargs, wce_kwargs, gamma=gamma)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        target = p>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.explog(p, target).item()
        else:
            return -np.inf'''

def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    GT = np.squeeze(GT)
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
        neg_edt =  distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt)-neg_edt)*negmask        
        res[i] = pos_edt/np.max(pos_edt) + neg_edt/np.max(neg_edt)
    return res
        
def nll_loss(input, target):
    """
    customized nll loss
    source: https://medium.com/@zhang_yang/understanding-cross-entropy-
    implementation-in-pytorch-softmax-log-softmax-nll-cross-entropy-416a2b200e34
    """
    loss = -input[range(target.shape[0]), target]
    return loss.mean()

class BinaryCrossEntropy_with_fp_fn(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        p = np.sum(probs[1:], axis=0) 
        y_true = p > self.threshold_lim
        y_pred = np.clip(p, 1e-7, 1 - 1e-7)
        term_0 = (1-y_true) * np.log(y_pred + 1e-7)
        term_1 = y_true *(np.log(y_pred + 1e-7)-np.log(1-y_pred + 1e-7))
        
        return np.mean(term_1-term_0)
    
def expand_onehot_labels(labels, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        elif labels.dim() == 4:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2], inds[3]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels, valid_mask


def get_region_proportion(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Get region proportion
    Args:
        x : one-hot label map/mask
        valid_mask : indicate the considered elements
    """
    if valid_mask is not None:
        x = torch.einsum("bcxyz,bxyz->bcxyz", x, valid_mask)
        cardinality = torch.einsum("bxyz->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
        # if valid_mask.dim() == 4:
        #     x = torch.einsum("bcwh, bcwh->bcwh", x, valid_mask)
        #     cardinality = torch.einsum("bcwh->bc", valid_mask)
        # else:
        #     x = torch.einsum("bcwh,bwh->bcwh", x, valid_mask)
        #     cardinality = torch.einsum("bwh->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3] * x.shape[4]

    # region_proportion = (torch.einsum("bcxyz->bc", x) + EPS) / (cardinality + EPS)
    region_proportion = (torch.sum(x, dim=(2, 3, 4)) + EPS) / (cardinality + EPS)

    return region_proportion    

class CompoundLoss(nn.Module):
    """
    The base class for implementing a compound loss:
        l = l_1 + alpha * l_2
    """
    def __init__(self, mode: str,
                 alpha: float = 1.,
                 factor: float = 1.,
                 step_size: int = 0,
                 max_alpha: float = 100.,
                 temp: float = 1.,
                 ignore_index: int = 1,
                 background_index: int = -1,
                 weight: Optional[torch.Tensor] = None) -> None:
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size
        self.temp = temp
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight

    def cross_entropy(self, inputs: torch.Tensor, labels: torch.Tensor):
        if len(labels.shape) == len(inputs.shape):
            assert labels.shape[1] == 1
            labels = labels[:, 0]
            labels = labels.unsqueeze(dim=1)
        if self.mode == MULTICLASS_MODE:
            loss = F.cross_entropy(
                inputs, labels.long(), weight=self.weight, ignore_index=self.ignore_index)
        else:
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(inputs, labels.type(torch.float32))
        return loss

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.factor, self.max_alpha)
            print(
                "CompoundLoss : Adjust the tradoff param alpha : {:.3g} -> {:.3g}".format(curr_alpha, self.alpha)
            )

    def get_gt_proportion(self, mode: str,
                          labels: torch.Tensor,
                          target_shape,
                          ignore_index: int = 255):
        if mode == MULTICLASS_MODE:
            bin_labels, valid_mask = expand_onehot_labels(labels, target_shape, ignore_index)
        else:
            valid_mask = (labels >= 0) & (labels != ignore_index)
            if labels.dim() == 4:
                labels = labels.unsqueeze(dim=0)
                valid_mask = valid_mask.unsqueeze(dim=0)
            bin_labels = labels
            valid_mask = valid_mask.squeeze(dim=0)
        gt_proportion = get_region_proportion(bin_labels, valid_mask)
        return gt_proportion, valid_mask

    def get_pred_proportion(self, mode: str,
                            logits: torch.Tensor,
                            temp: float = 1.0,
                            valid_mask=None):
        if mode == MULTICLASS_MODE:
            preds = F.log_softmax(temp * logits, dim=1).exp()
        else:
            preds = F.logsigmoid(temp * logits).exp()
        pred_proportion = get_region_proportion(preds, valid_mask)
        return pred_proportion


class CrossEntropyWithL1(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * |gt_region - prob_region|
    """
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        # ce term
        if len(labels.shape) == len(inputs.shape):
            assert labels.shape[1] == 1
            labels = labels[:, 0]
            labels = labels.unsqueeze(dim=1)
        labels = labels.long()

        loss_ce = self.cross_entropy(inputs, labels)
        # regularization
        gt_proportion, valid_mask = self.get_gt_proportion(self.mode, labels, inputs.shape)
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(dim=0)
        pred_proportion = self.get_pred_proportion(self.mode, inputs, temp=self.temp, valid_mask=valid_mask)
        loss_reg = (pred_proportion - gt_proportion).abs().mean()

        loss = loss_ce + self.alpha * loss_reg

        # return loss, loss_ce, loss_reg
        return loss

class CrossEntropyWithL1Confidence(SegmentationConfidence):
    def __init__(self, mode: str = 'binary', alpha: float = 1., factor: float = 1., step_size: int = 0, max_alpha: float = 100., temp: float = 1., ignore_index: int = 255, background_index: int = -1, weight: Optional[torch.Tensor] = None, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.cel1 = CrossEntropyWithL1(mode, alpha, factor, step_size, max_alpha, temp, ignore_index, background_index, weight)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        target = probs[1:]  # probability of foreground
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target).float()
        target = target>self.threshold
        p = np.sum(probs[1:], axis=0) 
        p = np.expand_dims(p, axis=0)
        p = np.expand_dims(p, axis=0)
        p = torch.from_numpy(p).float()
        if target.sum()>0:
            target = target.float()
            return -self.cel1(p, target).item()
        else:
            return -np.inf
        
class Linear_soft_dice(SegmentationConfidence):
    def __init__(self, threshold=.5, alpha=1, beta=1):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
    def metric(self, probs: np.array) -> float:
        p = np.sum(probs[1:], axis=0) 
        y_true = p > self.threshold
        result = ((1+self.alpha+self.beta)*y_true-self.alpha)*p - self.beta*y_true
        return np.mean(result)

class Linear_soft_dice_based_in_lesionload(SegmentationConfidence):
    def __init__(self, threshold=.5):
        super().__init__()
        self.threshold = threshold
    def metric(self, probs: np.array) -> float:
        p = np.sum(probs[1:], axis=0) 
        y_true = p > self.threshold
        lession_load_inverse = len(y_true.flatten())/np.sum(y_true)
        result = ((1+lession_load_inverse)*y_true-lession_load_inverse/2)*p - lession_load_inverse/2*y_true
        return np.mean(result)
    
class BoundaryLoss(nn.Module):
    # def __init__(self, **kwargs):
    def __init__(self, classes) -> None:
        super().__init__()
        # # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        self.idx = [i for i in range(classes)]

    def compute_sdf1_1(self, img_gt, out_shape):
        """
        compute the normalized signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM) 
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1, 1]
        """
        img_gt = img_gt.cpu().numpy()
        img_gt = img_gt.astype(np.uint8)

        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
                # ignore background
            for c in range(1, out_shape[1]):
                posmask = img_gt[b].astype(np.bool_)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance_transform_edt(posmask)
                    negdis = distance_transform_edt(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf

        return normalized_sdf

    def forward(self, outputs, gt):
        """
        compute boundary loss for binary segmentation
        input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
            gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
        output: boundary_loss; sclar
        """
        gt_sdf = self.compute_sdf1_1(gt, outputs.shape)
        pc = outputs_soft[:,self.idx,...]
        dc = torch.from_numpy(gt_sdf[:,self.idx,...]).cuda()
        if pc.dim()==4:
            multipled = torch.einsum('bxyz, bxyz->bxyz', pc, dc)
        else:
            multipled = torch.einsum('bxy, bxy->bxy', pc, dc)
        bd_loss = multipled.mean()

        return bd_loss

class BoundaryLossConfidence(SegmentationConfidence):
    def __init__(self, classes, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.bd = BoundaryLoss(classes)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        
        p = np.expand_dims(probs, axis=0)
        p = torch.from_numpy(p).float()
        target = np.expand_dims(probs[1:0], axis=0)
        target = torch.from_numpy(target).float()
        target = target>self.threshold
        if target.sum()>0:
            target = target.float()
            return -self.bd(p, target).item()
        else:
            return -np.inf
        
def get_one_hot(y_true, n_classes):
    y_true = y_true.to(torch.int64)
    y_true = one_hot(y_true, num_classes=n_classes)
    y_true = torch.transpose(y_true, dim0=5, dim1=1)
    y_true = torch.squeeze(y_true, dim=5)
    y_true = y_true.to(torch.int8)
    return y_true

def ants_to_sitk(img_ants):
    spacing = img_ants.spacing
    origin = img_ants.origin
    direction = tuple(img_ants.direction.flatten())

    img_sitk = sitk.GetImageFromArray(img_ants.numpy().T)
    img_sitk.SetSpacing(spacing)
    img_sitk.SetOrigin(origin)
    img_sitk.SetDirection(direction)

    return img_sitk


def sitk_to_ants(img_sitk):
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    direction_sitk = img_sitk.GetDirection()
    dim = int(np.sqrt(len(direction_sitk)))
    direction = np.reshape(np.array(direction_sitk), (dim, dim))

    img_ants = ants.from_numpy(sitk.GetArrayFromImage(img_sitk).T)
    img_ants.set_spacing(spacing)
    img_ants.set_origin(origin)
    img_ants.set_direction(direction)

    return img_ants

def sitk_get_min_max(image):
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    return stats_filter.GetMinimum(), stats_filter.GetMaximum()

def make_onehot(mask_ants, labels):
    spacing = mask_ants.spacing
    origin = mask_ants.origin
    direction = tuple(mask_ants.direction.flatten())
    
    mask_npy = mask_ants.numpy()
    masks_sitk = list()
    for i in range(len(labels)):
        sitk_label_i = sitk.GetImageFromArray((mask_npy == labels[i]).T.astype("float32"))
        sitk_label_i.SetSpacing(spacing)
        sitk_label_i.SetOrigin(origin)
        sitk_label_i.SetDirection(direction)
        masks_sitk.append(sitk_label_i)

    return masks_sitk

def compute_dtm(mask_ants, labels):
    dtms_sitk = list()
    masks_sitk = make_onehot(mask_ants, labels)

    for i, mask in enumerate(masks_sitk):
        dtm_i = sitk.SignedMaurerDistanceMap(sitk.Cast(masks_sitk[i], sitk.sitkUInt8),
                                             squaredDistance=False,
                                             useImageSpacing=False)

        dtm_int = sitk.Cast((dtm_i < 0), sitk.sitkFloat32)
        dtm_int *= dtm_i
        int_min, _ = sitk_get_min_max(dtm_int)

        dtm_ext = sitk.Cast((dtm_i > 0), sitk.sitkFloat32)
        dtm_ext *= dtm_i
        _, ext_max = sitk_get_min_max(dtm_ext)

        dtm_i = (dtm_ext / ext_max) - (dtm_int / int_min)

        dtms_sitk.append(dtm_i)

    dtm = sitk_to_ants(sitk.JoinSeries(dtms_sitk))
    dtm = dtm.numpy()
    return dtm

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred):
        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        num = torch.sum(torch.square(y_true - y_pred), dim=self.axes)
        den = torch.sum(torch.square(y_true), dim=self.axes) + torch.sum(torch.square(y_pred),
                                                                         dim=self.axes) + self.smooth

        loss = torch.mean(num / den, axis=1)
        loss = torch.mean(loss)
        return loss


class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_true, y_pred):
        # Dice loss
        loss_dice = self.dice_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1]).to(torch.float32)

        # Cross entropy loss
        loss_ce = self.cross_entropy(y_pred, y_true)

        return 0.5*(loss_ce + loss_dice)
    
class HDOneSidedLoss(nn.Module):
    def __init__(self):
        super(HDOneSidedLoss, self).__init__()
        self.region_loss = DiceCELoss()
        self.alpha = 0.5

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])

        # Compute boundary loss
        boundary_loss = torch.mean(torch.square(y_true - y_pred) * torch.square(dtm))

        return alpha * region_loss + (1. - alpha) * boundary_loss

class HDOneSidedLossConfidence(SegmentationConfidence):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.hd = HDOneSidedLoss()
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        target = p>self.threshold
        if target.sum()>0:
            target = target
            target_ants = target.squeeze(axis=0)
            labels = [0,1]
            dtm = compute_dtm(ants.from_numpy(data=target_ants .astype("float32")),labels)
            return -self.hd(p, target,dtm,0.5).item()
        else:
            return -np.inf
        
class GenSurfLoss(nn.Module):
    def __init__(self, class_weights):
        super(GenSurfLoss, self).__init__()
        self.region_loss = DiceCELoss()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not (class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.alpha = 0.5
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])

        if self.class_weights is None:
            class_weights = torch.sum(y_true, dim=self.axes)
            class_weights = 1. / (torch.square(class_weights) + 1.)
        else:
            class_weights = self.class_weights

        # Compute loss
        num = torch.sum(torch.square(dtm * (1 - (y_true + y_pred))), axis=self.axes)
        num *= class_weights

        den = torch.sum(torch.square(dtm), axis=self.axes)
        den *= class_weights
        den += self.smooth

        boundary_loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        boundary_loss = torch.mean(boundary_loss)
        boundary_loss = 1. - boundary_loss

        return alpha * region_loss + (1. - alpha) * boundary_loss

class GenSurfLossConfidence(SegmentationConfidence):
    def __init__(self, class_weights, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.gs = GenSurfLoss(class_weights)
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = probs[1:]  # probability of foreground
        p = np.expand_dims(p, axis=0)
        target = p>self.threshold
        if target.sum()>0:
            target = target
            dtm = compute_dtm(ants.from_numpy(data=p.astype("float32")),target)
            return -self.gs(p, target,dtm,0.5).item()
        else:
            return -np.inf

class Softmae(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        mae_value = mae(y_pred,y_true, soft=True)
        return -mae_value  
    

class Optimumquadratic(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        value = (np.sum(y_true-y_pred))**2 + np.sum(y_pred-y_pred**2)
        return -value  
    
class SoftDice_Opt_thresh(SegmentationConfidence):
    def __init__(self, threshold_lims=np.linspace(0.000, 1.000, 1001),eps=0):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.eps=eps
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        
        # Vectorized soft dice function
        def vectorized_soft_dice(y_pred, threshold_lim, eps):
            y_true = y_pred > threshold_lim
            intersection = np.sum(y_pred * y_true)
            total = np.sum(y_pred) + np.sum(y_true)
            dice = (2. * intersection + eps) / (total + eps)
            return dice
        
        # Vectorize the soft dice function
        vectorized_func = np.vectorize(vectorized_soft_dice, excluded=['y_pred','eps'], signature='(m,n),()->()')
        
        # Compute dice scores for each threshold using vectorized function
        dice_values = vectorized_func(y_pred, self.threshold_lims, eps=self.eps)
        
        #dice_values = [vectorized_soft_dice(y_pred, thresh, eps=self.eps) for thresh in self.threshold_lims]
        return np.max(dice_values), self.threshold_lims[np.argmax(dice_values)]    
    
class Optimumquadratic_Opt_thresh(SegmentationConfidence):
    def __init__(self, threshold_lims=np.linspace(0.000, 1.000, 1001)):
        super().__init__()
        self.threshold_lims = threshold_lims
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        
        # Vectorized soft dice function
        def vectorized_opt_quadratic(y_pred, threshold_lim):
            y_true = y_pred > threshold_lim
            value = (np.sum(y_true-y_pred))**2 + np.sum(y_pred-y_pred**2)
            return -value
        
        # Vectorize the soft dice function
        vectorized_func = np.vectorize(vectorized_opt_quadratic, excluded=['y_pred'], signature='(m,n),()->()')
        
        # Compute dice scores for each threshold using vectorized function
        vec_values = vectorized_func(y_pred, self.threshold_lims)
        
        #dice_values = [vectorized_soft_dice(y_pred, thresh, eps=self.eps) for thresh in self.threshold_lims]
        return np.max(vec_values), self.threshold_lims[np.argmax(vec_values)]   
    
    
class SoftDice_New_Opt_thresh(SegmentationConfidence):
    def __init__(self, threshold_lim_init = 0.5,eps=0):
        super().__init__()
        self.threshold_lim_init = threshold_lim_init
        self.eps=eps
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        
        for i in range(5):
            y_true = y_pred > self.threshold_lim_init
            intersection = np.sum(y_pred * y_true)
            total = np.sum(y_pred) + np.sum(y_true)
            dice = (2. * intersection + self.eps) / (total + self.eps)
            self.threshold_lim_init = dice
        return dice, dice
        
class Softmape(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        mape_value = mape(y_pred,y_true, soft=True)
        return -mape_value     
    
class Softmape_Opt_thresh(SegmentationConfidence):
    def __init__(self, threshold_lims=np.linspace(0.000, 1.000, 1001)):
        super().__init__()
        self.threshold_lims = threshold_lims
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        
        # Vectorized soft dice function
        def vectorized_opt_quadratic(y_pred, threshold_lim):
            y_true = y_pred > threshold_lim
            value = mape(y_pred,y_true, soft=True)
            return -value
        
        # Vectorize the soft dice function
        vectorized_func = np.vectorize(vectorized_opt_quadratic, excluded=['y_pred'], signature='(m,n),()->()')
        
        # Compute dice scores for each threshold using vectorized function
        vec_values = vectorized_func(y_pred, self.threshold_lims)
        
        #dice_values = [vectorized_soft_dice(y_pred, thresh, eps=self.eps) for thresh in self.threshold_lims]
        return np.max(vec_values), self.threshold_lims[np.argmax(vec_values)]   

    
class SoftDice_PLL(SegmentationConfidence):
    def __init__(self, threshold_lim=.5,eps=0, alpha =0.0):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.eps = eps
        self.alpha = alpha
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        dice = soft_dice(y_pred,y_true,eps=self.eps) 
        pll = np.sum(y_true)/len(y_true.flatten())
        #soft_dice_pll = dice*(1+self.alpha*pll)
        soft_dice_pll = dice+self.alpha*pll
        return soft_dice_pll   
    
class Median_min_max(SegmentationConfidence):
    def metric(self, probs: np.array, epsilon=1e-9) -> float:  
        # y_pred = np.sum(probs[1:], axis=0) 
        #median =  np.median(y_pred)
        #max_ = np.max(y_pred)
        #min_ = np.min(y_pred)
        #return (median+min_)/max_
        log_probs = np.log(probs + epsilon)
        unmap = -np.sum(probs * log_probs, axis=0)
        median =  np.median(unmap)
        max_ = np.max(unmap)
        min_ = np.min(unmap)
        return -(median+min_)/max_

class Median_min_max1(SegmentationConfidence):
    def metric(self, probs: np.array, epsilon=1e-9) -> float:  
        # y_pred = np.sum(probs[1:], axis=0) 
        #median =  np.median(y_pred)
        #max_ = np.max(y_pred)
        #min_ = np.min(y_pred)
        #return (median+min_)/max_
        msp = confidence = np.max(probs, axis=0)
        unmap = 1-msp
        median =  np.median(unmap)
        max_ = np.max(unmap)
        min_ = np.min(unmap)
        return -(median+min_)/max_
    
class Median_min_max2(SegmentationConfidence):
    def metric(self, probs: np.array, epsilon=1e-9) -> float:  
        y_pred = np.sum(probs[1:], axis=0) 
        median =  np.median(y_pred)
        max_ = np.max(y_pred)
        min_ = np.min(y_pred)
        return (median+min_)/max_

     
class Quantile_thresh(SegmentationConfidence):
    def __init__(self, threshold_alpha=.5,):
        super().__init__()
        self.q = 1 - threshold_alpha
    def metric(self, probs: np.array, epsilon=1e-9) -> float:
        log_probs = np.log(probs + epsilon)
        unmap = -np.sum(probs * log_probs, axis=0)
        threshold = np.quantile(unmap.flatten(),self.q)
        u = np.mean(unmap*(unmap>threshold))
        return -u

class Patch_Level_Aggregation(SegmentationConfidence):
    def __init__(self, patch_size=10, mean=False):
        super().__init__()
        self.patch_size = patch_size
        self.mean = mean

    def _effective_patch_size(self, probs_shape_spatial):
        """
        probs_shape_spatial: tuple of spatial dims (excluding channel dim).
        If self.patch_size is int, broadcast and cap each dim with min(k, d).
        If it's a sequence, zip-cap elementwise and validate length.
        """
        if isinstance(self.patch_size, int):
            k = self.patch_size
            return tuple(min(k, d) for d in probs_shape_spatial)
        elif isinstance(self.patch_size, (list, tuple)):
            if len(self.patch_size) != len(probs_shape_spatial):
                raise ValueError(
                    f"patch_size length {len(self.patch_size)} "
                    f"does not match image dims {len(probs_shape_spatial)}"
                )
            return tuple(min(k, d) for k, d in zip(self.patch_size, probs_shape_spatial))
        else:
            raise TypeError("patch_size must be int, list, or tuple.")

    def metric(self, probs: np.ndarray, epsilon=1e-9) -> float:
        """
        probs shape: (C, *spatial)  e.g., (C, H, W) or (C, D1, D2, D3, ...)
        Uses 'probs' to infer spatial dimensions and computes an entropy-like
        uncertainty map over channels, then performs valid N-D convolution
        with a ones kernel (sliding sum/mean over patches).

        Example for 2D (H, W):
          if given k (int), effective kernel is (min(k, H), min(k, W)).
        """
        # Build uncertainty map over channels (spatial-only array)
        log_probs = np.log(probs + epsilon)
        unmap = -np.sum(probs * log_probs, axis=0)  # shape == probs.shape[1:]

        # Determine effective patch size from probs' spatial shape
        spatial_shape = probs.shape[1:]
        eff_patch = self._effective_patch_size(spatial_shape)

        # N-D convolution kernel of ones with capped patch size
        kernel = np.ones(eff_patch, dtype=unmap.dtype)

        # Valid convolution gives patch-aggregated scores
        patch_aggregated = convolve(unmap, kernel, mode="valid")

        if self.mean:
            patch_aggregated = patch_aggregated / np.prod(eff_patch)

        # Locate one (the first) max location and the corresponding slice in original coords
        max_indices = np.where(np.isclose(patch_aggregated, patch_aggregated.max()))
        start_idx = tuple(int(ax_indices[0]) for ax_indices in max_indices)
        max_indices_slice = tuple((s, s + p) for s, p in zip(start_idx, eff_patch))
        # (max_indices_slice gives the spatial window in the original image)

        # Keep return behavior consistent with original code
        return -float(patch_aggregated.max())