# General modules & loading data
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from IPython.display import display

# Splitting & Oversampling Modules
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


# Pipeline Building Modules
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from skopt import BayesSearchCV



# Performance Metrics Modules
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, fbeta_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
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



def apply_single_clf(clf, split_data, timepoint,save_path='', apply_feature_selection = False, bagging=False, opt=False,opt_params={}, random_seed =42, silent=False):
    
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
        clf = BaggingClassifier(base_estimator=clf, n_estimators=20, n_jobs=20, random_state=random_seed)
        mode = 'bagging'

    elif apply_feature_selection:
        clf_name = clf_name + ' + Feature Selection'
        clf = Pipeline([('feature_selection', SelectFromModel(DecisionTreeClassifier(criterion='entropy', random_state=random_seed))),
                        ('classification', clf)
                        ])
        mode = 'feature selection'
    # elif opt:
    #     clf_name = clf_name + 'optimised'
    #     clf = BayesSearchCV(clf,
    #                         opt_params,
    #                         n_iter=32,
    #                         n_jobs=18,
    #                         cv=0)
        
    if type(clf).__name__ =='XGBClassifier':
        clf_fit =  clf.fit(train_feats, train_targets, eval_metric='aucpr')
        
    else:
        clf_fit =  clf.fit(train_feats, train_targets)
    
    train_y_pred = clf_fit.predict(train_feats)
    train_y_probas = clf_fit.predict_proba(train_feats)
    train_accuracy = clf_fit.score(train_feats,train_targets)
    train_auc = roc_auc_score(train_targets, train_y_probas[:,1])
    
    train_results = {'accuracy': train_accuracy,
                    'AUC': train_auc,
                    'bi_class_probas': train_y_probas,
                    'predictions': train_y_pred
                    }
       
    test_y_pred = clf_fit.predict(test_feats)
    test_y_probas = clf_fit.predict_proba(test_feats)
    test_accuracy = clf_fit.score(test_feats,test_targets)
    test_auc = roc_auc_score(test_targets, test_y_probas[:,1])
    test_pr_score = average_precision_score(test_targets, test_y_probas[:,1])
    test_f1_score = f1_score(test_targets, test_y_pred)
    test_mcc = matthews_corrcoef(test_targets, test_y_pred)
    test_fb_score = fbeta_score(test_targets, test_y_pred, beta=2)
    
    res = {'model': clf_name,
           'timepoint': timepoint,
           'mode': mode,
           'accuracy': test_accuracy,
           'AUC': test_auc,
           'pr_score': test_pr_score,
           'f1_score': test_f1_score,
           'fb_score': test_fb_score,
           'MCC_Score': test_mcc,
             }   
    results = pd.DataFrame(res,index=[0])
    
    test_results = {'accuracy': test_accuracy,
                    'AUC': test_auc,
                    'pr_score': test_pr_score,
                    'f1_score': test_f1_score,
                    'fb_score': test_fb_score,
                    'MCC_Score': test_mcc,
                    'bi_class_probas': test_y_probas,
                    'predictions': test_y_pred,
                    'results_df': results
                    } 
    
    packaged_result = {'model':clf_name,
                       'train_result': train_results,
                       'test_result': test_results,
                       'timepoint': timepoint
                       }
    
    if apply_feature_selection:
        selected_features = [split_data_timepoint['feature_names'][i] for i in clf_fit['feature_selection'].get_support(indices=True)]
        packaged_result['selected_features']=selected_features

    if not silent:
        print(f'Classifier: {clf_name}')
        print(f'Feature Selection Applied = {apply_feature_selection}')
        if apply_feature_selection:
            print(f'Features Selected:{selected_features}')
        print(f"Number of mislabeled points out of a total {len(test_feats)} points : {(test_targets != test_y_pred).sum()}")

        display(results)

        plt.style.use('classic')
        fig, axs = plt.subplots(1,3, figsize = (22.5,5))
        
        cm = plot_confusion_matrix(clf, test_feats,test_targets, ax=axs[0])
        roc = plot_roc_curve(clf, test_feats,test_targets, ax=axs[1] )
        pr = plot_precision_recall_curve(clf, test_feats,test_targets, ax=axs[2])
        
        fig.suptitle(clf_name)
        packaged_result['confusion_matrix']=cm
        packaged_result['roc']=roc
        
        subfolder = os.path.join(save_path,clf_original)
        subfolder = os.path.join(subfolder,timepoint)
        try:
            os.makedirs(subfolder)
        except:
            pass  
        
        plt.savefig(os.path.join(subfolder,clf_name)+'.png')
        plt.close(fig)
        
    if type(clf).__name__ =='XGBClassifier':    
        clf_fit.get_booster().feature_names = feature_names
        
    packaged_result['fit_model'] = clf_fit
    
    return packaged_result




def apply_multi_clf(clf, split_data,repeat = 1, apply_feature_selection = False, bagging=False, random_seed =42, silent=False):
    timepoints = list(split_data.keys())
    clf_name = type(clf).__name__
    
    if bagging:
        mode='bagging'
    elif apply_feature_selection:
        mode = 'feature selection'
    else:
        mode = 'base'
    
    
    accuracy_history = []
    
    auc_history = []
    pr_score_history = []
    f1_score_history = []
    fb_score_history = []
    mcc_history = []
    
    train_targets= split_data[timepoints[0]]['train_targets_encoded']
    test_targets = split_data[timepoints[0]]['test_targets_encoded']

    for i in trange(repeat, desc = clf_name):
        layer1_outputs = {}
        for t in timepoints:
            layer1_outputs[t] = apply_single_clf(clf,split_data,timepoint = t, silent=True,apply_feature_selection=apply_feature_selection, bagging=bagging)
        
        layer1_train_features = np.concatenate((layer1_outputs['t1']['train_result']['bi_class_probas'],layer1_outputs['t2']['train_result']['bi_class_probas']),axis=1)
        layer1_test_features  = np.concatenate((layer1_outputs['t1']['test_result']['bi_class_probas'],layer1_outputs['t2']['test_result']['bi_class_probas']),axis=1)

        layer2_fit =  clf.fit(layer1_train_features, train_targets)
        
        layer2_y_pred = layer2_fit.predict(layer1_test_features)
        layer2_y_probas = layer2_fit.predict_proba(layer1_test_features)
        layer2_accuracy = layer2_fit.score(layer1_test_features,test_targets)
        layer2_auc = roc_auc_score(test_targets, layer2_y_probas[:,1])
        layer2_pr_score = average_precision_score(test_targets, layer2_y_probas[:,1])
        layer2_f1_score = f1_score(test_targets, layer2_y_pred)
        layer2_fb_score = fbeta_score(test_targets, layer2_y_pred,beta=2)
        layer2_mcc = matthews_corrcoef(test_targets, layer2_y_pred)
    
        accuracy_history.append(layer2_accuracy)
        auc_history.append(layer2_auc)
        pr_score_history.append(layer2_pr_score)
        f1_score_history.append(layer2_f1_score)
        fb_score_history.append(layer2_fb_score)
        mcc_history.append(layer2_mcc)
    
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
    fig, axs = plt.subplots(1,3, figsize = (22.5,5))
    cm = plot_confusion_matrix(clf, layer1_test_features,test_targets,ax=axs[0])
    roc = plot_roc_curve(clf, layer1_test_features,test_targets, ax=axs[1] )
    prc = plot_precision_recall_curve(clf, layer1_test_features,test_targets, ax=axs[2])
    fig.suptitle(clf_name)
    
    packaged_result = {'fit_model': layer2_fit,
                    'accuracy': layer2_accuracy,
                    'AUC': layer2_auc,
                    'MCC_Score': layer2_mcc,
                    'PR_score': layer2_pr_score,
                    'f1_score': layer2_f1_score,
                    'bi_class_probas': layer2_y_probas,
                    'predictions': layer2_y_pred,
                    'results_df': results,
                    'confusion_matrix': cm,
                    'ROC':roc,
                    'prc':prc,
                    'model':clf_name,
                    }
    plt.close(fig)

    return packaged_result




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




# def single_timepoint_wrapper(data, clf,timepoints,save_path='',clinical_model=False, silent=True):

#     times = timepoints
    
#     results = {'basic': [apply_single_clf(clf, data,save_path = save_path, timepoint=t, apply_feature_selection=False, bagging=False, silent=silent) for t in times],
#                 'feature_selection': [apply_single_clf(clf, data,save_path = save_path, timepoint=t, apply_feature_selection=True, bagging=False, silent=silent) for t in times],
#                 'bagging':[apply_single_clf(clf, data,save_path = save_path, timepoint=t, apply_feature_selection=False, bagging=True, silent=silent) for t in times],
#                 }

#     df = pd.DataFrame()

#     for i in list(results.keys()):
#         for j in range(len(times)):
#             df=df.append(results[i][j]['test_result']['results_df']).reset_index(drop=True)
#             if not clinical_model:
#                 plot_km(results[i][j],data,folder='Naive Bayes',save_path = kaplan_saves)
            
#     df.sort_values(by='timepoint',inplace=True)
    
#     return df