from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import auc, roc_auc_score, average_precision_score

from src.metrics import accuracy, dice_coef, dice_norm_metric, hd95,soft_dice_norm_metric, rc_curve, balanced_accuracy, mae, sme, mape
from src.confidence import (BinaryCrossEntropy, new_form_gSoftDice, new_form_wl_gSoftDice, new_form_geometric_gSoftDice, SoftgDice, gDSCIntegralOverThreshold, bDSCIntegralOverThreshold,SoftnDice,SoftbDice, nDSCIntegralOverThreshold, bDSCIntegralOverThreshold, Lesion_Load,ExpectedEntropy, ExpectedEntropy_With_If,
                            MeanMaxConfidence, MeanMaxConfidence_With_If,
                            PredChangeDSC,
                            HD95IntegralOverThreshold,
                            DSCIntegralOverThreshold, SoftDice, DSC_Cov_based_confidence, DSC_Cov_based_confidence_diff_mag,nDSC_Cov_based_confidence, HD95_Cov_based_confidence, FocalLoss, BinaryCrossEntropy_with_fp_fn, IoULossConfidence, LovaszSoftmaxConfidence, HausdorffDTConfidence, SSLossConfidence,
AsymLossConfidence, TverskyLossConfidence, GDiceLossConfidence, DistBinaryDiceLossConfidence, BDLossConfidence, WeightedCrossEntropyLossConfidence, TopKLossConfidence, CrossentropyNDConfidence, HausdorffERLossConfidence, Linear_soft_dice, Linear_soft_dice_based_in_lesionload, CrossEntropyWithL1Confidence, BoundaryLossConfidence, HDOneSidedLossConfidence, GenSurfLossConfidence, Softmae, Optimumquadratic,  SoftDice_Opt_thresh, Optimumquadratic_Opt_thresh,SoftDice_New_Opt_thresh, Softmape, Softmape_Opt_thresh, SoftDice_PLL,SoftLesion_Load, Quantile_thresh, Median_min_max, Patch_Level_Aggregation)
from src.utils import plot_baselines, plot_rc_curve, plot_rc_curves, plot_aurc_curves, write_aurc_curves, rc_curve
from scipy.special import softmax, logit, expit
from collections import defaultdict
from matplotlib.ticker import FuncFormatter
import scienceplots

plt.style.use(['science'])
plt.rcParams['image.cmap'] = 'bwr'
#plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = (3.0, 2.625)
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 6
plt.rcParams.update({'font.size': 9})
colors = plt.colormaps['tab20']

def format_func(value, _):
    return f"{value:.1f}"

def auroc_func(y,y_hat):
    try:
        return roc_auc_score(y[i].flatten(), y_hat[i].flatten())
    except:
        return 0.0
    
def make_fig(y, y_hat,dir_path_save, name,dpi=300, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, metric='dice'):
    fig, ax = plt.subplots()
    if metric == 'dice':
        dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'mape':
        dice_errors = np.stack([mape(y_hat[i], y[i], threshold=threshold) for i in range(len(y))])
        
    threshold_alpha = thresh_alpha(y_hat,threshold = threshold)             
    confidence_metrics = {
        'MSP': MeanMaxConfidence(),
        'Neg.Entropy': ExpectedEntropy(),
        'PLL':Lesion_Load(threshold),
        'SDC': SoftDice(threshold),
        #'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        #'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        #'BCE': BinaryCrossEntropy(),
        #'Softmae':Softmae(threshold_lim= soft_thres),
        #'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        'MMMC':Median_min_max(),
        'NTLA':Quantile_thresh(threshold_alpha),
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(y_hat_i) for y_hat_i in y_hat]
                   for name, confidence_metric in confidence_metrics.items()}

    ax = plot_rc_curves(confidences, dice_errors, ax)

    ax.set_ylim(0,np.mean(dice_errors)*1.2)

    ax.set_title('%s'%name)
    ax.legend(loc='lower right')
    ax.set_ylabel('1-Dice')
    ax.set_xlabel('Coverage')
    fig.savefig('../%s/%s.pdf'%(dir_path_save,name), dpi=dpi)
    fig.show()

    
def make_fig_aurc(y, y_hat,dir_path_save, name,dpi=300, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, metric='dice'):
    fig, ax = plt.subplots()

    if metric == 'dice':
        dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(),y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'mape':
        dice_errors = np.stack([mape(y_hat[i], y[i], threshold=threshold) for i in range(len(y))])
    threshold_alpha = thresh_alpha(y_hat,threshold = threshold)                     
    confidence_metrics = {
        'aMSP': MeanMaxConfidence(),
        'aNE': ExpectedEntropy(),
        #'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        #'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        #'BCE': BinaryCrossEntropy(),
        #'Softmae':Softmae(threshold_lim= soft_thres),
        #'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        'MMMC':Median_min_max(),
        'NTLA':Quantile_thresh(threshold_alpha),
        'PLL':Lesion_Load(threshold),
        'SDC': SoftDice(threshold),
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(y_hat_i) for y_hat_i in y_hat]
                   for name, confidence_metric in confidence_metrics.items()}
    #confidences['Avg_HDT_Lession_Load'] = (np.array(confidences['HDT_loss'])+np.array(confidences['Lesion Load']))/2
    #confidences['Avg_HDT_Lession_Load'] = confidences['Avg_HDT_Lession_Load'].tolist()
    ax = plot_aurc_curves(confidences, dice_errors, ax)

    ax.set_ylim(0,np.mean(dice_errors)*1.2)

    #ax.set_title('%s'%name)
    #ax.legend(loc='lower right')
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    ax.set_ylabel('Risk')
    ax.set_xlabel('Coverage')
    #fig.set_size_inches(4.5,3.25)
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    fig.savefig('../%s/%s.pdf'%(dir_path_save,name), dpi=dpi)
    fig.show()

def make_row_aurc(y, p_hat, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, metric='dice'):

    if metric == 'dice':
        dice_scores = np.stack([dice_coef(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(p_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(p_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(),p_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), p_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'mape':
        dice_errors = np.stack([mape(p_hat[i], y[i], threshold=threshold) for i in range(len(y))])
        
    threshold_alpha = thresh_alpha(p_hat,threshold = threshold)            
    confidence_metrics = {
        'aMSP': MeanMaxConfidence(),
        'aNE': ExpectedEntropy(),
        #'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        #'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        #'BCE': BinaryCrossEntropy(),
        #'Softmae':Softmae(threshold_lim= soft_thres),
        #'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        'MMMC':Median_min_max(),
        'TLA':Quantile_thresh(threshold_alpha),
        'PLA':Patch_Level_Aggregation(),
        #'PLL':Lesion_Load(threshold),
        'SDC': SoftDice(threshold),
        #'MSP': MeanMaxConfidence(),
        #'Neg.Entropy': ExpectedEntropy(),
        #'Lesion Load':Lesion_Load(soft_thres),
        #'SDC': SoftDice(soft_thres),
        #'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        #'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        #'BCE': BinaryCrossEntropy(threshold_lim= soft_thres),
        #'FocalLoss':FocalLoss(threshold_lim= soft_thres), 
        #'BCE_with_fp_fn':BinaryCrossEntropy_with_fp_fn(threshold_lim= soft_thres), 
        #'IOU_loss':IoULossConfidence(threshold= soft_thres), 
        #'Lovasz_loss':LovaszSoftmaxConfidence(threshold = soft_thres), 
        #'HDT_loss':HausdorffDTConfidence(threshold = soft_thres),
        #'SS_loss':SSLossConfidence(threshold = soft_thres),
        #'Tversky_loss':TverskyLossConfidence(threshold = soft_thres),
        #'Asym_loss':AsymLossConfidence(threshold = soft_thres),
        #'GDice_loss':GDiceLossConfidence(threshold = soft_thres),
        #'FocalTversky_loss':FocalTversky_lossConfidence(threshold = soft_thres),
        #'PenaltyGDice_loss':PenaltyGDiceLossConfidence(threshold = soft_thres),
        #'ExpLog_loss':ExpLog_lossConfidence(soft_dice_kwargs, wce_kwargs, gamma=0.3, threshold = soft_thres),
        #'BD_loss':BDLossConfidence(threshold = soft_thres),
        #'DistBinaryDice_loss':DistBinaryDiceLossConfidence(threshold = soft_thres),
        #'WeightedCE_loss':WeightedCrossEntropyLossConfidence(threshold = soft_thres),
        #'TopK_loss':TopKLossConfidence(threshold = soft_thres),
        #'HER_loss':HausdorffERLossConfidence(threshold = soft_thres),
        #'RCE(l1)': CrossEntropyWithL1Confidence(threshold = soft_thres),
        #'Linear_soft_dice_a_b=1': Linear_soft_dice(threshold = soft_thres),
        #'Linear_soft_dice_based_in_lesionload':  Linear_soft_dice_based_in_lesionload(threshold = soft_thres),
        #'BoundaryLossConfidence': BoundaryLossConfidence(classes=2, threshold = soft_thres),
        #'Softmae':Softmae(threshold_lim= soft_thres),
        #'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        #'Softmape':Softmape(threshold_lim= soft_thres),
        #'HDOneSidedLossConfidence': HDOneSidedLossConfidence(threshold = soft_thres),
        #'GenSurfLossConfidence': GenSurfLossConfidence(class_weights=None, threshold=soft_thres)
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(p_hat_i) for p_hat_i in p_hat]
                   for name, confidence_metric in confidence_metrics.items()}
    
    #confidences['Avg_HDT_Lession_Load'] = (np.array(confidences['HDT_loss'])+np.array(confidences['Lesion Load']))/2
    #confidences['Avg_HDT_Lession_Load'] = confidences['Avg_HDT_Lession_Load'].tolist()
    aurcs = write_aurc_curves(confidences, dice_errors)
    
    return aurcs

def corruption_level(p_hat, y, error_base=0, threshold = 0.5, n_dice_r=0.7, metric='dice'):
    if metric == 'dice':
        dice_scores = np.stack([dice_coef(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(p_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(p_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(),p_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), p_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(p_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'mape':
        dice_errors = np.stack([mape(p_hat[i], y[i], threshold=threshold) for i in range(len(y))])   
        
    corruption = {}
    threshold_alpha = thresh_alpha(p_hat,threshold = threshold)            
    confidence_metrics = {
        'aMSP': MeanMaxConfidence(),
        'aNE': ExpectedEntropy(),
        #'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        #'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        #'BCE': BinaryCrossEntropy(),
        #'Softmae':Softmae(threshold_lim= soft_thres),
        #'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        'MMMC':Median_min_max(),
        'TLA':Quantile_thresh(threshold_alpha),
        'PLA':Patch_Level_Aggregation(),
        #'PLL':Lesion_Load(threshold),
        'SDC': SoftDice(threshold),
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(p_hat_i) for p_hat_i in p_hat]
                   for name, confidence_metric in confidence_metrics.items()}
    for name, confidence in confidences.items():
        coverage, risk, _ = rc_curve(confidence, dice_errors)
        coverage, risk = np.array(coverage), np.array(risk)
        try:
            corruption[name] = round(coverage[np.where(risk<=error_base)][-1]*100,3)
        except:
            corruption[name] = '-'
    coverage_ideal, risk_ideal, _ = rc_curve(-dice_errors, dice_errors, ideal=True)
    coverage_ideal, risk_ideal = np.array(coverage_ideal), np.array(risk_ideal)
    corruption['Ideal'] = round(coverage_ideal[np.where(risk_ideal<=error_base)][-1]*100,1)
    return corruption

def write_text_table_AURC(all_aurcs,name_file):
    with open("%s.txt"%name_file, "w") as f:
        for name in all_aurcs:
            _, model, dataset = name.split(' - ')
            f.write('\hline \n')
            f.write('%s&%s&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f\\\ \n' %(model.split(' model')[0], dataset.replace(' ', ''), all_aurcs[name]['Random'], all_aurcs[name]['MSP'], all_aurcs[name]['Neg.Entropy'],  all_aurcs[name]['Lesion Load'], all_aurcs[name]['SDC'],  all_aurcs[name]['Ideal']))
            
def write_text_table_corruption(all_aurcs,name_file):
    with open("%s.txt"%name_file, "w") as f:
        for name in all_aurcs:
            _, model, dataset = name.split(' - ')
            f.write('\hline \n')
            f.write('%s&%s&%s&%s&%s&%s&%s\\\ \n' %(model.split(' model')[0], dataset.replace(' ', ''), all_aurcs[name]['MSP'], all_aurcs[name]['Neg.Entropy'], all_aurcs[name]['Lesion Load'], all_aurcs[name]['SDC'], all_aurcs[name]['Ideal']))

def thresh_alpha(y_hat,threshold = 0.5):
    threshold_alpha = []
    for y_hat_i in y_hat:
        threshold_alpha.append(np.sum(y_hat_i>threshold)/len(y_hat_i.flatten()))
    return np.mean(threshold_alpha)

def make_fig_aurc_with_if(y, y_hat,dir_path_save, name,dpi=300, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, metric='dice'):
    fig, ax = plt.subplots()

    if metric == 'dice':
        dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(),y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    thresh_alpha = thresh_alpha(y_hat,threshold = threshold)            
    confidence_metrics = {
        'MSP': MeanMaxConfidence(),
        'Neg.Entropy': ExpectedEntropy(),
        'PLL':Lesion_Load(threshold),
        'SDC': SoftDice(threshold),
        #'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        #'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        #'BCE': BinaryCrossEntropy(),
        #'Softmae':Softmae(threshold_lim= soft_thres),
        #'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        'Median_min_max':Median_min_max(),
        'Quantile_thresh':Quantile_thresh(thresh_alpha),
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(y_hat_i) for y_hat_i in y_hat]
                   for name, confidence_metric in confidence_metrics.items()}

    ax = plot_aurc_curves(confidences, dice_errors, ax)

    ax.set_ylim(0,np.mean(dice_errors)*1.2)

    #ax.set_title('%s'%name)
    ax.legend(loc='lower right')
    ax.set_ylabel('Risk')
    ax.set_xlabel('Coverage')
    fig.savefig('../%s/%s.pdf'%(dir_path_save,name), dpi=dpi)
    fig.show()

def make_row_aurc_with_if(y, y_hat, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, metric='dice'):

    if metric == 'dice':
        dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(),y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores    
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
                
    confidence_metrics = {
        'MSP': MeanMaxConfidence(),
        'Neg.Entropy': ExpectedEntropy(),
        'Lesion Load':Lesion_Load(soft_thres),
        'SDC': SoftDice(soft_thres),
        'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        'BCE': BinaryCrossEntropy(),
        'Softmae':Softmae(threshold_lim= soft_thres),
        'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(y_hat_i) for y_hat_i in y_hat]
                   for name, confidence_metric in confidence_metrics.items()}

    aurcs = write_aurc_curves(confidences, dice_errors)
    
    return aurcs

def corruption_level_with_if(y_hat, y, error_base=0, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, metric='dice'):
    if metric == 'dice':
        dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y_hat[i].flatten()>threshold, y[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])     
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
                
    corruption = {}
    confidence_metrics = {
        'MSP': MeanMaxConfidence(),
        'Neg.Entropy': ExpectedEntropy(),
        'Lesion Load':Lesion_Load(soft_thres),
        'SDC': SoftDice(soft_thres),
        'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        'BCE': BinaryCrossEntropy(),
        'Softmae':Softmae(threshold_lim= soft_thres),
        'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(y_hat_i) for y_hat_i in y_hat]
                   for name, confidence_metric in confidence_metrics.items()}
    for name, confidence in confidences.items():
        coverage, risk, _ = rc_curve(confidence, dice_errors)
        coverage, risk = np.array(coverage), np.array(risk)
        try:
            corruption[name] = round(coverage[np.where(risk<=error_base)][-1]*100,3)
        except:
            corruption[name] = '-'
        coverage_ideal, risk_ideal, _ = rc_curve(-dice_errors, dice_errors, ideal=True)
        coverage_ideal, risk_ideal = np.array(coverage_ideal), np.array(risk_ideal)
        corruption['Ideal'] = round(coverage_ideal[np.where(risk_ideal<=error_base)][-1]*100,3)
    return corruption

def make_dataframe(y, y_hat,file_name, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7):
    
    dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    dice_errors = 1 - dice_scores
  
    ndice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
    ndice_errors = 1 - ndice_scores
    
    
    hd95_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    
    acc_scores = np.stack([accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    acc_errors = 1 - acc_scores

    
    bacc_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    bacc_errors = 1 - bacc_scores
    
    auroc_scores = np.stack([auroc_func(y[i].flatten(),y_hat[i].flatten()) for i in range(len(y))])
    auroc_errors = 1 - auroc_scores
    
    aupr_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
    aupr_errors = 1 - aupr_scores
    
    mae_errors = np.stack([mae(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
     
    sme_errors = np.stack([sme(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    
    mape_errors = np.stack([mape(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])  
    
    confidence_metrics = {
        'MSP': MeanMaxConfidence(),
        'Neg.Entropy': ExpectedEntropy(),
        'Lesion Load':Lesion_Load(soft_thres),
        'SDC': SoftDice(soft_thres),
        'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        'BCE': BinaryCrossEntropy(threshold_lim= soft_thres),
        'FocalLoss':FocalLoss(threshold_lim= soft_thres), 
        'BCE_with_fp_fn':BinaryCrossEntropy_with_fp_fn(threshold_lim= soft_thres), 
        'IOU_loss':IoULossConfidence(threshold= soft_thres), 
        'Lovasz_loss':LovaszSoftmaxConfidence(threshold = soft_thres), 
        'HDT_loss':HausdorffDTConfidence(threshold = soft_thres),
        'SS_loss':SSLossConfidence(threshold = soft_thres),
        'Tversky_loss':TverskyLossConfidence(threshold = soft_thres),
        'Asym_loss':AsymLossConfidence(threshold = soft_thres),
        'GDice_loss':GDiceLossConfidence(threshold = soft_thres),
        #'FocalTversky_loss':FocalTversky_lossConfidence(threshold = soft_thres),
        #'PenaltyGDice_loss':PenaltyGDiceLossConfidence(threshold = soft_thres),
        #'ExpLog_loss':ExpLog_lossConfidence(soft_dice_kwargs, wce_kwargs, gamma=0.3, threshold = soft_thres),
        #'BD_loss':BDLossConfidence(threshold = soft_thres),
        'DistBinaryDice_loss':DistBinaryDiceLossConfidence(threshold = soft_thres),
        #'WeightedCE_loss':WeightedCrossEntropyLossConfidence(threshold = soft_thres),
        #'TopK_loss':TopKLossConfidence(threshold = soft_thres),
        'HER_loss':HausdorffERLossConfidence(threshold = soft_thres),
        'RCE(l1)': CrossEntropyWithL1Confidence(threshold = soft_thres),
        'Linear_soft_dice_a_b=1': Linear_soft_dice(threshold = soft_thres),
        'Linear_soft_dice_based_in_lesionload':  Linear_soft_dice_based_in_lesionload(threshold = soft_thres),
        'BoundaryLossConfidence': BoundaryLossConfidence(classes=2, threshold = soft_thres),
        'Softmae':Softmae(threshold_lim= soft_thres),
        'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        'Softmape':Softmape(threshold_lim= soft_thres),
        #'HDOneSidedLossConfidence': HDOneSidedLossConfidence(threshold = soft_thres),
        #'GenSurfLossConfidence': GenSurfLossConfidence(class_weights=None, threshold=soft_thres)
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    
    all_dict = {name: [confidence_metric(y_hat_i) for y_hat_i in y_hat]
                   for name, confidence_metric in confidence_metrics.items()}
    all_dict['Avg_HDT_Lession_Load'] = (np.array(all_dict['HDT_loss'])+np.array(all_dict['Lesion Load']))/2
    all_dict['Avg_HDT_Lession_Load'] =  all_dict['Avg_HDT_Lession_Load'].tolist()
    all_dict['dice risk'] = dice_errors
    all_dict['ndice risk'] = ndice_errors
    all_dict['hd95 risk'] = hd95_errors
    all_dict['acc risk'] = acc_errors
    all_dict['bacc risk'] = bacc_errors
    all_dict['auroc risk'] = auroc_errors
    all_dict['aupr risk'] = aupr_errors
    all_dict['mae risk'] = mae_errors
    all_dict['sme risk'] = sme_errors
    
    pd.DataFrame.from_dict(all_dict).to_pickle(f"../dataframes/{file_name}.pkl")  
    

def make_fig_2(y, y_hat,dir_path_save, name,dpi=300, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, metric='dice', ref = 'SDC_Opt_thresh'):
    fig, ax = plt.subplots()
    
        
    confidence_metrics = {
        'MSP': MeanMaxConfidence(),
        'Neg.Entropy': ExpectedEntropy(),
        'Lesion Load':Lesion_Load(soft_thres),
        'SDC': SoftDice(soft_thres),
        'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        'BCE': BinaryCrossEntropy(),
        'HDT_loss':HausdorffDTConfidence(threshold = soft_thres),
        'Softmae':Softmae(threshold_lim= soft_thres),
        'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        'Softmape':Softmape(threshold_lim= soft_thres),
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
        #'new_form_gSoftDice': new_form_gSoftDice(alpha=best_alpha,gamma1=best_g1, gamma2=best_g2),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = defaultdict(list)
    optimum_threshs = defaultdict(list)
    for name, confidence_metric in confidence_metrics.items():
        conf, thresh = confidence_metric(y_hat_i)
        confidences[name].append(conf)
        optimum_threshs[name].append(thresh)
        
    
    if metric == 'dice':
        dice_scores = np.stack([new_dice_coef(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=optimum_threshs[ref][i],r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
    elif metric == 'mape':
        dice_errors = np.stack([mape(y_hat[i], y[i], threshold=threshold) for i in range(len(y))])   
        
    ax = plot_rc_curves(confidences, dice_errors, ax)

    ax.set_ylim(0,np.mean(dice_errors)*1.2)

    ax.set_title('%s'%name)
    ax.legend(loc='lower right')
    ax.set_ylabel('1-Dice')
    ax.set_xlabel('Coverage')
    fig.savefig('../%s/%s.pdf'%(dir_path_save,name), dpi=dpi)
    fig.show()
    
def make_row_aurc_2(y, y_hat, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, metric='dice', ref='SoftDice_Opt_thresh'):
    confidence_metrics = {
        'SoftDice_Opt_thresh': SoftDice_Opt_thresh(),
        'Optimumquadratic_Opt_thresh': Optimumquadratic_Opt_thresh(),
        'SoftDice_New_Opt_thresh': SoftDice_New_Opt_thresh(),
        'Softmape_Opt_thresh':Softmape_Opt_thresh(threshold_lim= soft_thres),
        
    }
    confidences = defaultdict(list)
    optimum_threshs = defaultdict(list)
    for name, confidence_metric in confidence_metrics.items():
        for y_hat_i in y_hat:
            conf, thresh = confidence_metric(y_hat_i)
            confidences[name].append(conf)
            optimum_threshs[name].append(thresh)
        
    
    if metric == 'dice':
        dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=optimum_threshs[ref][i],r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=optimum_threshs[ref][i]) for i in range(len(y))])                           
    elif metric == 'mape':
        dice_errors = np.stack([mape(y_hat[i], y[i], threshold=threshold) for i in range(len(y))])   
    aurcs = write_aurc_curves(confidences, dice_errors)
    
    return aurcs

def make_fig_aurc_with_random(y, y_hat,dir_path_save, name,random,dpi=300, threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, alpha=0, alpha_g_dice=1, threshold_alpha=0.5,metric='dice'):
    fig, ax = plt.subplots()

    if metric == 'dice':
        dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(),y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'mape':
        dice_errors = np.stack([mape(y_hat[i], y[i], threshold=threshold) for i in range(len(y))])
                        
    confidence_metrics = {
        'MSP': MeanMaxConfidence(),
        'Neg.Entropy': ExpectedEntropy(),
        'Median_min_max':Median_min_max(),
        'Quantile_thresh':Quantile_thresh(threshold_alpha),
        'PLL':Lesion_Load(soft_thres),
        #'SLL':SoftLesion_Load(),
        'SDC': SoftDice(soft_thres),
        #'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        #'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        #'BCE': BinaryCrossEntropy(threshold_lim= soft_thres),
        #'FocalLoss':FocalLoss(threshold_lim= soft_thres), 
        #'BCE_with_fp_fn':BinaryCrossEntropy_with_fp_fn(threshold_lim= soft_thres), 
        #'IOU_loss':IoULossConfidence(threshold= soft_thres), 
        #'Lovasz_loss':LovaszSoftmaxConfidence(threshold = soft_thres), 
        #'HDT_loss':HausdorffDTConfidence(threshold = soft_thres),
        #'SS_loss':SSLossConfidence(threshold = soft_thres),
        #'Tversky_loss':TverskyLossConfidence(threshold = soft_thres),
        #'Asym_loss':AsymLossConfidence(threshold = soft_thres),
        #'GDice_loss':GDiceLossConfidence(threshold = soft_thres),
        #'FocalTversky_loss':FocalTversky_lossConfidence(threshold = soft_thres),
        #'PenaltyGDice_loss':PenaltyGDiceLossConfidence(threshold = soft_thres),
        #'ExpLog_loss':ExpLog_lossConfidence(soft_dice_kwargs, wce_kwargs, gamma=0.3, threshold = soft_thres),
        #'BD_loss':BDLossConfidence(threshold = soft_thres),
        #'DistBinaryDice_loss':DistBinaryDiceLossConfidence(threshold = soft_thres),
        #'WeightedCE_loss':WeightedCrossEntropyLossConfidence(threshold = soft_thres),
        #'TopK_loss':TopKLossConfidence(threshold = soft_thres),
        #'HER_loss':HausdorffERLossConfidence(threshold = soft_thres),
        #'RCE(l1)': CrossEntropyWithL1Confidence(threshold = soft_thres),
        #'Linear_soft_dice_a_b=1': Linear_soft_dice(threshold = soft_thres),
        #'Linear_soft_dice_based_in_lesionload':  Linear_soft_dice_based_in_lesionload(threshold = soft_thres),
        #'BoundaryLossConfidence': BoundaryLossConfidence(classes=2, threshold = soft_thres),
        #'Softmae':Softmae(threshold_lim= soft_thres),
        #'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        #'Softmape':Softmape(threshold_lim= soft_thres),
        'SoftDice_PLL(alpha=%.4f)'%(alpha):SoftDice_PLL(threshold_lim= soft_thres, alpha=alpha),
        #'HDOneSidedLossConfidence': HDOneSidedLossConfidence(threshold = soft_thres),
        #'GenSurfLossConfidence': GenSurfLossConfidence(class_weights=None, threshold=soft_thres)
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
    #    'gSoftDice': new_form_gSoftDice(alpha=alpha_g_dice,gamma1=1, gamma2=1),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(y_hat_i) for y_hat_i in y_hat]
                   for name, confidence_metric in confidence_metrics.items()}
    
    #confidences['Avg_HDT_Lession_Load'] = (np.array(confidences['HDT_loss'])+np.array(confidences['PLL']))/2
    #confidences['Avg_HDT_Lession_Load'] = confidences['Avg_HDT_Lession_Load'].tolist()
    #confidences['TLL'] = [confidence_metrics['PLL'](y_i) for y_i in y]
    confidences['Random_Forest'] = random.tolist()
    ax = plot_aurc_curves(confidences, dice_errors, ax)

    ax.set_ylim(0,np.mean(dice_errors)*1.2)

    #ax.set_title('%s'%name)
    ax.legend(loc='lower right')
    ax.set_ylabel('Risk')
    ax.set_xlabel('Coverage')
    #fig.set_size_inches(4.5,3.25)
    fig.savefig('../%s/%s.pdf'%(dir_path_save,name), dpi=dpi)
    fig.show()
    
def make_row_aurc_with_random(y, y_hat,random,threshold = 0.5, soft_thres=0.5, n_dice_r=0.7, alpha=0, alpha_g_dice=1,threshold_alpha=0.5,metric='dice'):

    if metric == 'dice':
        dice_scores = np.stack([dice_coef(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric=='ndice':
        dice_scores = np.stack([dice_norm_metric(y_hat[i], y[i],threshold=threshold,r=n_dice_r) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric =='hd95':
        dice_errors = np.stack([hd95(y_hat[i], y[i]) for i in range(len(y))])
    elif metric == 'acc':
        dice_scores = np.stack([accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'bacc':
        dice_scores = np.stack([balanced_accuracy(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'auroc':
        dice_scores = np.stack([auroc_func(y[i].flatten(),y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'aupr':
        dice_scores = np.stack([average_precision_score(y[i].flatten(), y_hat[i].flatten()) for i in range(len(y))])
        dice_errors = 1 - dice_scores
    elif metric == 'mae':
        dice_errors = np.stack([mae(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'sme':
        dice_errors = np.stack([sme(y_hat[i], y[i],threshold=threshold) for i in range(len(y))])
    elif metric == 'mape':
        dice_errors = np.stack([mape(y_hat[i], y[i], threshold=threshold) for i in range(len(y))])
        
    confidence_metrics = {
        'MSP': MeanMaxConfidence(),
        'Neg.Entropy': ExpectedEntropy(),
        'Median_min_max':Median_min_max(),
        'Quantile_thresh':Quantile_thresh(threshold_alpha),
        'PLL':Lesion_Load(soft_thres),
        #'SLL':SoftLesion_Load(),
        'SDC': SoftDice(soft_thres),
        #'SoftnDice': SoftnDice(r= n_dice_r, threshold_lim= soft_thres),
        #'HD95IOT': HD95IntegralOverThreshold(threshold_lim= soft_thres),
        #'BCE': BinaryCrossEntropy(threshold_lim= soft_thres),
        #'FocalLoss':FocalLoss(threshold_lim= soft_thres), 
        #'BCE_with_fp_fn':BinaryCrossEntropy_with_fp_fn(threshold_lim= soft_thres), 
        #'IOU_loss':IoULossConfidence(threshold= soft_thres), 
        #'Lovasz_loss':LovaszSoftmaxConfidence(threshold = soft_thres), 
        #'HDT_loss':HausdorffDTConfidence(threshold = soft_thres),
        #'SS_loss':SSLossConfidence(threshold = soft_thres),
        #'Tversky_loss':TverskyLossConfidence(threshold = soft_thres),
        #'Asym_loss':AsymLossConfidence(threshold = soft_thres),
        #'GDice_loss':GDiceLossConfidence(threshold = soft_thres),
        #'FocalTversky_loss':FocalTversky_lossConfidence(threshold = soft_thres),
        #'PenaltyGDice_loss':PenaltyGDiceLossConfidence(threshold = soft_thres),
        #'ExpLog_loss':ExpLog_lossConfidence(soft_dice_kwargs, wce_kwargs, gamma=0.3, threshold = soft_thres),
        #'BD_loss':BDLossConfidence(threshold = soft_thres),
        #'DistBinaryDice_loss':DistBinaryDiceLossConfidence(threshold = soft_thres),
        #'WeightedCE_loss':WeightedCrossEntropyLossConfidence(threshold = soft_thres),
        #'TopK_loss':TopKLossConfidence(threshold = soft_thres),
        #'HER_loss':HausdorffERLossConfidence(threshold = soft_thres),
        #'RCE(l1)': CrossEntropyWithL1Confidence(threshold = soft_thres),
        #'Linear_soft_dice_a_b=1': Linear_soft_dice(threshold = soft_thres),
        #'Linear_soft_dice_based_in_lesionload':  Linear_soft_dice_based_in_lesionload(threshold = soft_thres),
        #'BoundaryLossConfidence': BoundaryLossConfidence(classes=2, threshold = soft_thres),
        #'Softmae':Softmae(threshold_lim= soft_thres),
        #'Optimum_quadratic': Optimumquadratic(threshold_lim= soft_thres),
        #'Softmape':Softmape(threshold_lim= soft_thres),
        'SoftDice_PLL(alpha=%.4f)'%(alpha):SoftDice_PLL(threshold_lim= soft_thres, alpha=alpha),
        #'HDOneSidedLossConfidence': HDOneSidedLossConfidence(threshold = soft_thres),
        #'GenSurfLossConfidence': GenSurfLossConfidence(class_weights=None, threshold=soft_thres)
        #'DSCIntegralOverThreshold': DSCIntegralOverThreshold(),
        #'bDSCIntegralOverThreshold': bDSCIntegralOverThreshold(r=0.08644042231819847),
    #    'gSoftDice': new_form_gSoftDice(alpha=alpha_g_dice,gamma1=1, gamma2=1),
    #    'SoftLesion_Load':SoftLesion_Load(),
    #    'DSC_Cov_based_confidence_Min_Max':DSC_Cov_based_confidence(eta),
    #    'DSC_Cov_based_confidence_0.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha1),
    #    'DSC_Cov_based_confidence_1.5':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha2),
    #    'DSC_Cov_based_confidence_2.0':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha3),
    #    'DSC_Cov_based_confidence_Multi':DSC_Cov_based_confidence_diff_mag(eta,alpha=alpha4),
    }
    confidences = {name: [confidence_metric(y_hat_i) for y_hat_i in y_hat]
                   for name, confidence_metric in confidence_metrics.items()}
    
    #confidences['Avg_HDT_Lession_Load'] = (np.array(confidences['HDT_loss'])+np.array(confidences['PLL']))/2
    #confidences['Avg_HDT_Lession_Load'] = confidences['Avg_HDT_Lession_Load'].tolist()
    #confidences['TLL'] = [confidence_metrics['PLL'](y_i) for y_i in y]
    confidences['Random_Forest'] = random.tolist()
    aurcs = write_aurc_curves(confidences, dice_errors)
    
    return aurcs

def coverage_at_specific_error(confidences,dice_errors, error_base=0, threshold = 0.5):
    corruption = {}
    for name, confidence in confidences.items():
        coverage, risk, _ = rc_curve(confidence, dice_errors)
        coverage, risk = np.array(coverage), np.array(risk)
        try:
            corruption[name] = round(coverage[np.where(risk<=error_base)][-1]*100,3)
        except:
            corruption[name] = 0
    coverage_ideal, risk_ideal, _ = rc_curve(-dice_errors, dice_errors, ideal=True)
    coverage_ideal, risk_ideal = np.array(coverage_ideal), np.array(risk_ideal)
    corruption['Ideal'] = round(coverage_ideal[np.where(risk_ideal<=error_base)][-1]*100,1)
    return corruption
