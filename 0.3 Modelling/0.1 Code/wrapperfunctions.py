# General modules & loading data
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from IPython.display import display
from copy import deepcopy

# Splitting & Oversampling Modules
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


# Pipeline Building Modules
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
# from skopt import BayesSearchCV



# Performance Metrics Modules
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, fbeta_score, auc, roc_curve, precision_recall_curve
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, confusion_matrix

import kaplanmeier as km


def train_test_split(image_dataframe,n_splits=2, encode_targets = True, apply_SMOTE = True, random_state = 42):
    skf = StratifiedKFold(n_splits)
    
    ifeatures = image_dataframe.iloc[:, 1:-2]
    feature_names = ifeatures.columns
    ifeatures = ifeatures.to_numpy()
    itargets = image_dataframe.iloc[:, -2].to_numpy()
    pids = image_dataframe.iloc[:, 0].to_numpy()
    kaplan_times = image_dataframe.iloc[:, -1].to_numpy()



    for train_index, test_index in skf.split(ifeatures, itargets):
        train_feats, test_feats = ifeatures[train_index], ifeatures[test_index]
        train_targets, test_targets = itargets[train_index], itargets[test_index]
        train_pids, test_pids = pids[train_index], pids[test_index]
        train_kap_times, test_kap_times = kaplan_times[train_index], kaplan_times[test_index]
    
    if encode_targets:
        le = preprocessing.LabelEncoder()
        le.fit(train_targets)
        le.classes_ = le.classes_[::-1]
        train_targets_encoded=le.transform(train_targets)
        test_targets_encoded=le.transform(test_targets)
     
    if apply_SMOTE:
        oversample = SMOTE(k_neighbors=4, random_state=random_state)
        train_feats, train_targets_encoded = oversample.fit_resample(train_feats, train_targets_encoded)
        train_targets = le.inverse_transform(train_targets_encoded) 

    packaged_result = {'feature_names': feature_names,
                       'pre_split_feature_values': ifeatures,
                       'pre_split_targets': itargets,
                       'train_feats': train_feats,
                       'test_feats': test_feats,
                       'train_targets': train_targets,
                       'test_targets': test_targets,
                       'train_targets_encoded': train_targets_encoded,
                       'test_targets_encoded': test_targets_encoded,
                       'train_pids': train_pids,
                       'test_pids': test_pids,
                       'train_kap_times': train_kap_times,
                       'test_kap_times': test_kap_times,
                        }
   
    return  packaged_result



def apply_single_clf(clf, split_data, timepoint,save_location,repeat = 1, apply_feature_selection = False, bagging=False, silent=False):
    
    split_data_timepoint = split_data[timepoint]
    
    feature_names = split_data_timepoint['feature_names'].values.tolist()
    train_feats = split_data_timepoint['train_feats']
    test_feats = split_data_timepoint['test_feats']
    train_targets= split_data_timepoint['train_targets_encoded']
    test_targets = split_data_timepoint['test_targets_encoded']
    
    clf_name = type(clf).__name__
    clf_original = clf_name
    mode = 'Base'
    
    if bagging:
        clf_name = clf_name + ' + Bagging'
        clf = BaggingClassifier(base_estimator=clf, n_estimators=20, n_jobs=20)
        mode = 'bagging'

    elif apply_feature_selection:
        clf_name = clf_name + ' + Feature Selection'
        clf = Pipeline([('feature_selection', SelectFromModel(DecisionTreeClassifier(criterion='entropy'))),
                        ('classification', clf)
                        ])
        mode = 'feature selection'
        
    subfolder = os.path.join(save_location,timepoint)
    
    subfolder = os.path.join(subfolder,clf_original)
    subfolder = os.path.join(subfolder,clf_name)
    
    try:
        os.makedirs(subfolder)
    except:
        pass   

    accuracy_history = []
    auc_history = []
    pr_score_history = []
    f1_score_history = []
    fb_score_history = []
    mcc_history = []
    tpr_history = []
    precision_history = []
    cm_history =[]
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100) 
    
    
    plt.style.use('classic')
    plt.tight_layout()
    figure_sizes = (5,5)
    cmap = plt.cm.Blues

    fig,ax = plt.subplots(figsize =figure_sizes)
    fig_pr,ax_pr = plt.subplots(figsize =figure_sizes)
    
    for i in trange(repeat, desc = clf_name): 
        clf_fit =  clf.fit(train_feats, train_targets)
       
        test_y_pred = clf_fit.predict(test_feats)
        test_y_probas = clf_fit.predict_proba(test_feats)
        test_accuracy = clf_fit.score(test_feats,test_targets)
        test_auc = roc_auc_score(test_targets, test_y_probas[:,1])
        test_pr_score = average_precision_score(test_targets, test_y_probas[:,1])
        test_f1_score = f1_score(test_targets, test_y_pred)
        test_fb_score = fbeta_score(test_targets, test_y_pred, beta=2)
        test_mcc = matthews_corrcoef(test_targets, test_y_pred)
     
        test_cm = confusion_matrix(test_targets, test_y_pred,normalize='true')

        accuracy_history.append(test_accuracy)
        auc_history.append(test_auc)
        pr_score_history.append(test_pr_score)
        f1_score_history.append(test_f1_score)
        fb_score_history.append(test_fb_score)
        mcc_history.append(test_mcc)
        cm_history.append(test_cm)
        
        viz_roc= plot_roc_curve(clf, test_feats,test_targets,alpha=0.1, lw=1, ax=ax,label='_nolegend_')
        viz_pr = plot_precision_recall_curve(clf, test_feats,test_targets,alpha=0.1, lw=1, ax=ax_pr,label='_nolegend_')
        
        interp_tpr = np.interp(mean_fpr, viz_roc.fpr, viz_roc.tpr)
        interp_tpr[0] = 0.0
               
        interp_precision = np.interp(mean_recall, np.flip(viz_pr.recall),np.flip(viz_pr.precision))
      
        tpr_history.append(interp_tpr)
        precision_history.append(interp_precision)
        
    mean_tpr = np.mean(tpr_history,axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    std_auc = np.std(auc_history)
    std_tpr = np.std(tpr_history, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
     
    mean_precision = np.mean(precision_history,axis=0)
    mean_pr_score = np.mean(pr_score_history)
    std_pr_score = np.std(pr_score_history)
    std_precision = np.std(precision_history, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0) 
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b', 
            label=f'Mean ROC (AUC = {mean_auc:.2f} ${{\pm}}$ {std_auc:.2f})',
            lw=2, alpha=.8)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax_pr.set_xlabel('False Positive Rate')
    ax_pr.set_ylabel('True Positive Rate')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.set_title(clf_name)
    fig.savefig(os.path.join(subfolder,clf_name)+'_roc.png',bbox_inches = "tight")
    plt.close(fig)
    
    ax_pr.plot([0, 1], [0, 0], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    ax_pr.plot(mean_recall, mean_precision, color='b', 
               label=f'Mean PR Curve \n(Average Precision = {mean_pr_score:.2f} ${{\pm}}$ {std_pr_score:.2f})',
               lw=2, alpha=.8)
    ax_pr.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')

    ax_pr.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax_pr.set_title(clf_name)
    fig_pr.savefig(os.path.join(subfolder,clf_name)+'_pr.png',bbox_inches = "tight")
    plt.close(fig_pr)    
        
    # Plot Confusion Matrix
    fig,ax = plt.subplots(figsize =figure_sizes)
    labels = ('NED','ED')
    mean_cm = np.mean(cm_history,axis=0)
    std_cm = np.std(cm_history,axis=0)
    ned_labels = [ f'{mean_cm.flatten()[i]:.2f} ${{\pm}}$ {std_cm.flatten()[i]:.2f}\n({mean_cm.flatten()[i]*(len(test_targets)-sum(test_targets)):.2f} ${{\pm}}$ {std_cm.flatten()[i]*(len(test_targets)-sum(test_targets)):.2f})' for i in range(2)]
    ed_labels = [ f'{mean_cm.flatten()[i]:.2f} ${{\pm}}$ {std_cm.flatten()[i]:.2f}\n({mean_cm.flatten()[i]*sum(test_targets):.2f} ${{\pm}}$ {std_cm.flatten()[i]*sum(test_targets):.2f})' for i in range(2,4)]
    value_labels = ned_labels + ed_labels
    value_labels = np.asarray(value_labels).reshape(2,2)
    sns.heatmap(mean_cm,annot = value_labels,fmt='',cmap=cmap,cbar=False,ax=ax,xticklabels=labels, yticklabels=labels)
    ax.set_title(clf_name)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.savefig(os.path.join(subfolder,clf_name)+'_cm.png',bbox_inches = "tight")
    plt.close(fig)
     
    average_accuracy = sum(accuracy_history)/repeat
    average_auc = sum(auc_history)/repeat
    average_pr_score = sum(pr_score_history)/repeat
    average_f1_score =  sum(f1_score_history)/repeat
    average_fb_score = sum(fb_score_history)/repeat

    average_mcc = sum(mcc_history)/repeat

    
    res = {'model': clf_name,
           'mode': mode,
           'timepoint': timepoint,
           'accuracy': average_accuracy,
           'AUC': average_auc,
           'pr_score': average_pr_score,
           'f1_score': average_f1_score,
           'fb_score': average_fb_score,
           'MCC_Score': average_mcc,
          }
    
    results = pd.DataFrame(res,index=[0])
       
    packaged_result = {'fit_model': clf_fit,
                    'results_df': results,
                    'model':clf_name,
                    'test_result':results
                    }    
            
    return packaged_result




def apply_multi_clf(clf, split_data, save_location,repeat = 1, apply_feature_selection = False, bagging=False, random_seed =42, silent=False):
    timepoints = list(split_data.keys())
    clf_name = type(clf).__name__
    clf_original = clf_name
    
    if bagging:
        clf_name = clf_name + ' + Bagging'
        clf = BaggingClassifier(base_estimator=clf, n_estimators=20, n_jobs=20)
        mode = 'bagging'
        
    elif apply_feature_selection:
        clf_name = clf_name + ' + Feature Selection'
        clf = Pipeline([('feature_selection', SelectFromModel(DecisionTreeClassifier(criterion='entropy'))),
                        ('classification', clf)
                        ])
        mode = 'feature selection'
        
    else:
        mode = 'base'
        
    subfolder = os.path.join(save_location,'Multi_Models')
    subfolder = os.path.join(subfolder,clf_original)
    subfolder = os.path.join(subfolder,clf_name)
    
    try:
        os.makedirs(subfolder)
    except:
        pass     
        
    train_targets= split_data[timepoints[0]]['train_targets_encoded']
    test_targets = split_data[timepoints[0]]['test_targets_encoded']
    # test_targ_weights = test_targets + 1  # This makes the positive class weighted twice as important as the negative
    
    accuracy_history = []
    auc_history = []
    pr_score_history = []
    f1_score_history = []
    fb_score_history = []
    mcc_history = []
    tpr_history = []
    precision_history = []
    cm_history =[]
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    
    plt.style.use('classic')
    plt.tight_layout()
    figure_sizes = (5,5)
    cmap = plt.cm.Blues
    
    fig,ax = plt.subplots(figsize =figure_sizes)
    fig_pr,ax_pr = plt.subplots(figsize =figure_sizes)
    fit_clf ={'t1':deepcopy(clf),
              't2':deepcopy(clf)}
    
    joining_clf = GaussianNB()

    for i in trange(repeat, desc = clf_name):
        for t in timepoints:
            fit_clf[t] = fit_clf[t].fit(split_data[t]['train_feats'],split_data[t]['train_targets_encoded'])
    
        layer1_train_features = np.concatenate((fit_clf['t1'].predict_proba(split_data['t1']['train_feats']),fit_clf['t2'].predict_proba(split_data['t2']['train_feats'])),axis=1)
        layer1_test_features = np.concatenate((fit_clf['t1'].predict_proba(split_data['t1']['test_feats']),fit_clf['t2'].predict_proba(split_data['t2']['test_feats'])),axis=1)

        layer2_fit =  joining_clf.fit(layer1_train_features, train_targets)
         
        layer2_y_pred = layer2_fit.predict(layer1_test_features)
 
        layer2_y_probas = layer2_fit.predict_proba(layer1_test_features)
        layer2_accuracy = layer2_fit.score(layer1_test_features,test_targets)
        layer2_auc = roc_auc_score(test_targets, layer2_y_probas[:,1])
        layer2_pr_score = average_precision_score(test_targets, layer2_y_probas[:,1])
        layer2_f1_score = f1_score(test_targets, layer2_y_pred)
        layer2_fb_score = fbeta_score(test_targets, layer2_y_pred,beta=2)
        layer2_mcc = matthews_corrcoef(test_targets, layer2_y_pred)
        layer2_cm = confusion_matrix(test_targets, layer2_y_pred, normalize='true')
        
        accuracy_history.append(layer2_accuracy)
        auc_history.append(layer2_auc)
        pr_score_history.append(layer2_pr_score)
        f1_score_history.append(layer2_f1_score)
        fb_score_history.append(layer2_fb_score)
        mcc_history.append(layer2_mcc)
        cm_history.append(layer2_cm)
        
        viz_roc= plot_roc_curve(joining_clf, layer1_test_features,test_targets,alpha=0.3, lw=1, ax=ax,label='_nolegend_')
        viz_pr = plot_precision_recall_curve(joining_clf, layer1_test_features,test_targets,alpha=0.3, lw=1, ax=ax_pr,label='_nolegend_')
        
        interp_tpr = np.interp(mean_fpr, viz_roc.fpr, viz_roc.tpr)
        interp_tpr[0] = 0.0
        
        interp_precision = np.interp(mean_recall, np.flip(viz_pr.recall),np.flip(viz_pr.precision))
      
        tpr_history.append(interp_tpr)
        precision_history.append(interp_precision)
    
    mean_tpr = np.mean(tpr_history,axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    
    std_auc = np.std(auc_history)
 
    std_tpr = np.std(tpr_history, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
     
    mean_precision = np.mean(precision_history,axis=0)
    mean_pr_score = np.mean(pr_score_history)
    std_pr_score = np.std(pr_score_history)
    std_precision = np.std(precision_history, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    
    
    # r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
    # f'Mean ROC (AUC = {mean_auc:.2f} ${{\pm}}$ {std_auc:.2f}'
    # f'Mean PR Curve \n(Average Precision = {mean_pr_score:.2f} ${{\pm}}$ {std_pr_score:.2f}'
    
    # label=r'Mean ROC (Average Precision Score = %0.2f $\pm$ %0.2f)' % (mean_pr_score, std_pr_score),

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label = f'Mean ROC (AUC = {mean_auc:.2f} ${{\pm}}$ {std_auc:.2f})',
            lw=2, alpha=.8)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.set_title(clf_name)
    fig.savefig(os.path.join(subfolder,clf_name)+'_roc.png',bbox_inches = "tight")
    plt.close(fig)
    
    ax_pr.plot([0, 1], [0, 0], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    ax_pr.plot(mean_recall, mean_precision, color='b',
        label=f'Mean PR Curve \n(Average Precision = {mean_pr_score:.2f} ${{\pm}}$ {std_pr_score:.2f})',
        lw=2, alpha=.8)
    ax_pr.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

    ax_pr.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax_pr.set_title(clf_name)
    fig_pr.savefig(os.path.join(subfolder,clf_name)+'_pr.png',bbox_inches = "tight")
    plt.close(fig_pr)
    
    # Plot Confusion Matrix
    fig,ax = plt.subplots(figsize =figure_sizes)
    labels = ('NED','ED')
    mean_cm = np.mean(cm_history,axis=0)
    std_cm = np.std(cm_history,axis=0)
    ned_labels = [ f'{mean_cm.flatten()[i]:.2f} ${{\pm}}$ {std_cm.flatten()[i]:.2f}\n({mean_cm.flatten()[i]*(len(test_targets)-sum(test_targets)):.2f} ${{\pm}}$ {std_cm.flatten()[i]*(len(test_targets)-sum(test_targets)):.2f})' for i in range(2)]
    ed_labels = [ f'{mean_cm.flatten()[i]:.2f} ${{\pm}}$ {std_cm.flatten()[i]:.2f}\n({mean_cm.flatten()[i]*sum(test_targets):.2f} ${{\pm}}$ {std_cm.flatten()[i]*sum(test_targets):.2f})' for i in range(2,4)]
    value_labels = ned_labels + ed_labels
    value_labels = np.asarray(value_labels).reshape(2,2)
    sns.heatmap(mean_cm,annot = value_labels,fmt='',cmap=cmap,cbar=False,ax=ax,xticklabels=labels, yticklabels=labels)
    ax.set_title(clf_name)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.savefig(os.path.join(subfolder,clf_name)+'_cm.png',bbox_inches = "tight")
    plt.close(fig)
   
    average_accuracy = sum(accuracy_history)/repeat
    average_auc = sum(auc_history)/repeat
    average_pr_score = sum(pr_score_history)/repeat
    average_f1_score =  sum(f1_score_history)/repeat
    average_fb_score = sum(fb_score_history)/repeat
    average_mcc = sum(mcc_history)/repeat
    
    res = {'model': clf_name,
           'mode': mode,
           'accuracy': average_accuracy,
           'AUC': average_auc,
           'PR_score': average_pr_score,
           'f1_score': average_f1_score,
           'fb_score': average_fb_score,
           'MCC_Score': average_mcc,
          }
    
    results = pd.DataFrame(res,index=[0])
    
    packaged_result = {'fit_model': layer2_fit,
                    'accuracy': layer2_accuracy,
                    'AUC': layer2_auc,
                    'MCC_Score': layer2_mcc,
                    'PR_score': layer2_pr_score,
                    'f1_score': layer2_f1_score,
                    'bi_class_probas': layer2_y_probas,
                    'predictions': layer2_y_pred,
                    'results_df': results,
                    'model':clf_name,
                    }
    
    return packaged_result


def generate_plots(clf, test_feats, test_targets, clf_name, save_location):
    
    packaged_plots ={}

    plt.style.use('classic')
    plt.tight_layout()
    figure_sizes = (5,5)
    cmap = plt.cm.Blues
    
    fig, axs = plt.subplots(1,1, figsize = figure_sizes)
    cm = plot_confusion_matrix(clf, test_feats,test_targets,display_labels=('NED','ED'),normalize='true',cmap =cmap, ax=axs,colorbar=False)
    axs.set_title(clf_name)
    packaged_plots['confusion_matrix']=cm
    plt.savefig(os.path.join(save_location,clf_name)+'_cm.png',bbox_inches = "tight")
    plt.close(fig)
    
    fig, axs = plt.subplots(1,1, figsize = figure_sizes)
    roc = plot_roc_curve(clf, test_feats,test_targets, ax=axs )
    axs.set_title(clf_name)
    packaged_plots['roc']=roc
    plt.savefig(os.path.join(save_location,clf_name)+'_roc.png',bbox_inches = "tight")
    plt.close(fig)
    
    fig, axs = plt.subplots(1,1, figsize = figure_sizes)
    pr = plot_precision_recall_curve(clf, test_feats,test_targets, ax=axs)
    axs.set_title(clf_name)
    packaged_plots['pr_curve']=pr
    plt.savefig(os.path.join(save_location,clf_name)+'_pr_curve.png',bbox_inches = "tight")
    plt.close(fig)

    return packaged_plots



def plot_km(model_output,split_data,folder,save_path,multi=False):
     subfolder = os.path.join(save_path,folder)
     try:
          os.makedirs(subfolder)
     except:
          pass  
     
     if multi:
         d = {'time' : split_data['t1']['test_kap_times'],
               'progressed' : split_data['t1']['test_targets_encoded'],
               'groups' : [ 'Remission' if x ==0 else 'Progression' for x in model_output['predictions']],
               }
         model_name = model_output['model']          
     else:
          d = {'time' : split_data['t1']['test_kap_times'],
               'progressed' : split_data['t1']['test_targets_encoded'],
               'groups' : [ 'Remission' if x ==0 else 'Progression' for x in model_output['test_result']['predictions']],
               }
          model_name = model_output['timepoint'] + '_'+model_output['model']
     
     savepath = os.path.join(subfolder,model_name)
     
     df = pd.DataFrame(d)
     out=km.fit(df['time'],df['progressed'],df['groups'])
     
     km.plot(out,savepath=savepath,title=model_name, full_ylim=True)
     
     plt.close()
     
     
def clinical_train_test_split(image_dataframe,n_splits=2, encode_targets = True, apply_SMOTE = True, random_state = 42):
    skf = StratifiedKFold(n_splits)
    
    ifeatures = image_dataframe.iloc[:, 1:-2]
    feature_names = ifeatures.columns
    ifeatures = ifeatures.to_numpy()
    itargets = image_dataframe.iloc[:, -1].to_numpy()
    pids = image_dataframe.iloc[:, 0].to_numpy()



    for train_index, test_index in skf.split(ifeatures, itargets):
        train_feats, test_feats = ifeatures[train_index], ifeatures[test_index]
        train_targets, test_targets = itargets[train_index], itargets[test_index]
        train_pids, test_pids = pids[train_index], pids[test_index]
    
    if encode_targets:
        le = preprocessing.LabelEncoder()
        le.fit(train_targets)
        le.classes_ = le.classes_[::-1]
        train_targets_encoded=le.transform(train_targets)
        test_targets_encoded=le.transform(test_targets)
     
    if apply_SMOTE:
        oversample = SMOTE(k_neighbors=4, random_state=random_state)
        train_feats, train_targets_encoded = oversample.fit_resample(train_feats, train_targets_encoded)
        train_targets = le.inverse_transform(train_targets_encoded) 

    packaged_result = {'feature_names': feature_names,
                       'pre_split_feature_values': ifeatures,
                       'pre_split_targets': itargets,
                       'train_feats': train_feats,
                       'test_feats': test_feats,
                       'train_targets': train_targets,
                       'test_targets': test_targets,
                       'train_targets_encoded': train_targets_encoded,
                       'test_targets_encoded': test_targets_encoded,
                       'train_pids': train_pids,
                       'test_pids': test_pids,
                        }
   
    return  packaged_result



def save_stats_tests(chi_tests, numeric_analysis, save_location):
    
    writer = pd.ExcelWriter(os.path.join(save_location,'split_evaluation.xlsx'), engine='xlsxwriter')

    labels = ['Expected','Observed','Stats']


    for key in chi_tests.keys():
        row = 1
        
        for i in range(len(chi_tests[key])):
            chi_tests[key][i].to_excel(writer, sheet_name=key,startrow=row , startcol=0)
            worksheet = writer.sheets[key]
            worksheet.write_string(row-1, 0, labels[i])

            row = row + len(chi_tests[key][i].index)  + 3

    row =1       
    for i in list(numeric_analysis.keys()):
        numeric_analysis[i].to_excel(writer, sheet_name='Numeric Analysis',startrow=row , startcol=0)
        worksheet = writer.sheets['Numeric Analysis']
        worksheet.write_string(row-1, 0, i)
        row = row + len(numeric_analysis[i].index)  + 3
            
    writer.save()
    writer.close()





def get_best_mcc(trues, probas, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0,1,1000)
    
    mcc_history = []
    
    for t in thresholds:
        predicts = [0 if probas[i]<t else 1 for i in range(len(probas))]
        tn,tp,fn,fp = 0,0,0,0

        for x in range(len(trues)):
            tr=trues[x]
            pr=predicts[x]
            if tr==pr:
                if tr==0:
                    tn+=1
                else:
                    tp+=1
            else:
                if pr==0:
                    fn+=1
                else:
                    fp+=1
                    
        mcc = compute_mcc(tn,tp,fn,fp)
        
        mcc_history.append(mcc)
    
    ix = np.argmax(mcc_history)
    best_mcc = mcc_history[ix]
    best_threshold = thresholds[ix]
    
    res = {'mcc':best_mcc,
           'thres':best_threshold,        
    }    
  
    return res


def compute_mcc(tn,tp,fn,fp):
    # predicts = [0 if probas[i]<threshold else 1 for i in range(len(probas))]
    # tn,tp,fn,fp = 0,0,0,0

    # for x in range(len(trues)):
    #     tr=trues[x]
    #     pr=predicts[x]
    #     if tr==pr:
    #         if tr==0:
    #             tn+=1
    #         else:
    #             tp+=1
    #     else:
    #         if pr==0:
    #             fn+=1
    #         else:
    #             fp+=1
    
    mcc = np.nan_to_num(((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),copy=True,nan=0)      
                    
    
    return mcc
        