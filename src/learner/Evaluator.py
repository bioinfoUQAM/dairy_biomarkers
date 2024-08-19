from sklearn.metrics import precision_recall_fscore_support as prfs, roc_auc_score
import numpy as np
from src.cutils import UtilMethods
import shap
import pandas as pd
from os import path

class Evaluator:
    
    def __init__(self,data_instances, performance_path, merge_performance, filter_names):
        self.data_instances = data_instances
        self.performance_path = performance_path
        self.metrics = ['P', 'R', 'F1']
        self.header = 'classifier\ttransformation\ttask\tlabel\tsplit\tbalance\ttAcc\tfilters\tfeatCount\t'
        self.performance_per_ids = ''
        self.merge_performance = merge_performance
        self.filter_names = filter_names
        self.shap_importance = pd.DataFrame(columns=['feat', 'shap_value','task','split_type','label_type','feat_len','filters'])
        self.output = ''
        
        
    def output_metrics(self, header, output, tags, task, classifier_type):        
        print(header, '\n', output[:-1])
        result = header + '\n' + output[:-1]  
        task = task + '_'+ classifier_type if 'regular' not in classifier_type else task
        ext = '.merged.metrics' if self.merge_performance else '.metrics'
        result_path = self.performance_path + task+ '_' + tags['split_type'] + '_' + tags['split_balance'] + '_' + tags['label_type'] + '_' + tags['feat_len'] + tags['filters'].replace(',','-') 
        UtilMethods.write_file(result_path + ext, result)
        UtilMethods.write_file(result_path + ext + '.IDs', self.performance_per_ids[:-1])
        
        
    
    def shap_features(self, model, X_test, tags, task, features):        
        result_path = self.performance_path + '/shap/'
        result_path = result_path + task+ '_' + tags['classifier'] + '_' + tags['transformation'] + '_' + tags['split_type'] + '_' + tags['split_balance'] + '_' + tags['label_type'] + '_' + tags['feat_len'] + '_' + tags['filters'].replace(',','-') 
        result_path = result_path + '.shap'
        
        if (not path.isfile(result_path)):
            print('Computing shap for', result_path)
            idx = np.random.choice(len(X_test), 120, replace=False)
            sample_data = np.array([X_test[i] for i in idx])
            explainer = shap.KernelExplainer(model, sample_data)
            shap_values = explainer(sample_data)
            ext = '.shap'
            values = np.abs(shap_values.values).mean(0)
            importance = pd.DataFrame(list(zip(features,values)), columns=['feat','shap_value'])
            importance['classifier'] = tags['classifier']
            importance['transformation'] = tags['transformation']
            importance['task'] = task
            importance['split_type'] = tags['split_type']
            importance['label_type'] = tags['label_type']
            importance['feat_len'] = tags['feat_len']
            importance['filters'] = tags['filters'].replace(',','-')
            #self.shap_importance = pd.concat([self.shap_importance, importance]) if not self.shap_importance.empty else importance
            
            importance.to_csv(result_path,sep='\t',index=False)
        #shap.summary_plot(shap_values.data, sample_data, feature_names = features)        
        #return shap_values
        
    
    def get_metrics_per_id(self, tags, task, these_labels, probabilites, predictions, true_labels):
        
        if(self.merge_performance or 'predict' in task):
            self.performance_per_ids += 'ATQ\tclassifier\ttransformation\ttask\tlabel_type\tfilters\tfeat_len\treal\tpredicted\tscore_real\t' + 'score_pred\n' 
            true_labels, predictions, probabilites = self.merge_predictions(task, probabilites, predictions, true_labels)            
            ids = list(true_labels.keys())
            for id in ids:                
                context = id + '\t' + tags['classifier'] +'\t' + tags['transformation']+ '\t' + task +'\t' + tags['label_type'] +'\t' + tags['filters'].replace('_','-') +'\t' + tags['feat_len']
                predicted = [these_labels[idx] for idx, i in enumerate(predictions[id]) if i ==1] if 'multiclass' in tags['label_type'] else these_labels[predictions[id].item()]
                real = [these_labels[idx] for idx, i in enumerate(true_labels[id]) if i == 1] if 'multiclass' in tags['label_type'] else these_labels[true_labels[id].item()]
                score_real = (predictions[id] * true_labels[id]).sum() / true_labels[id].sum() if 'multiclass' in tags['label_type'] else true_labels[id].item()
                score_pred = (predictions[id] * true_labels[id]).sum() / predictions[id].sum() if 'multiclass' in tags['label_type'] else predictions[id].item()
                real = ','.join(real) if 'multiclass' in tags['label_type'] else real
                predicted = ','.join(predicted) if 'multiclass' in tags['label_type'] else predicted
                self.performance_per_ids += context + '\t'  + real + '\t' + predicted + '\t' + str(score_real) + '\t' + str(score_pred) + '\n'
            
            probabilites = list(probabilites.values())
            predictions = list(predictions.values())
            true_labels = list(true_labels.values())
        
        return probabilites, predictions, true_labels
    
    
    def get_classification_metrics(self, tags, task, acc, probabilities, predictions, true_labels):   
        
        metrics_dict = {'acc': '{0:.3f}'.format(acc)}
        header, content = '',''
        prfscore = prfs(true_labels, predictions)   
        classifier = tags['classifier'] 
        if('predict' not in task):            
            content = classifier + '\t'+ tags['transformation'] + '\t' + task + '\t' + tags['label_type'] + '\t' + tags['split_type'] + '\t' + tags['split_balance']  + '\t' + '{0:.3f}'.format(acc) + '\t' + tags['filters'].replace('_','-') + '\t' + str(tags['feat_len'] ) +'\t'            
            roc = roc_auc_score(true_labels, predictions) if 'multi' not in tags['label_type']  else roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro') 
            
            header = self.header + 'roc_auc' + '_' + tags['label_type']  + '\t'
            content += '{0:.3f}'.format(roc) + '\t'
            metrics_dict['roc_auc'] =  '{0:.3f}'.format(roc)
            label_type = tags['label_type'].split('_')[0]
            label_idx = 0            
            for label, encoded_label in self.data_instances[task+'_Y_'+label_type+'Mapping'].items():
                label_idx = encoded_label if 'binary' in label_type else label_idx
                for idx, metric in enumerate(self.metrics):
                    header += metric + '_' + label + '\t'                    
                    content += '{0:.3f}'.format(prfscore[idx][label_idx]) + '\t'                    
                    metrics_dict[metric + '_' + label] =  '{0:.3f}'.format(prfscore[idx][label_idx])
                label_idx += 1
        else:
            metrics_dict['prfs'] = prfscore
        
        return header[:-1], content[:-1], metrics_dict
    
    
    
    def merge_predictions(self, task, probabilites, predictions, true_labels):
        labels_per_ATQ, predictions_per_ATQ, probabilites_per_ATQ = {}, {}, {}
        for idx, instance in  enumerate(self.data_instances[task+'_ids']):
            atq = instance.split('_')[0]
            these_preds = predictions[idx]
            these_probs = probabilites[idx]
            these_labels = true_labels[idx]
            if(atq not in predictions_per_ATQ):
                predictions_per_ATQ[atq] = [these_preds]
                probabilites_per_ATQ[atq] = [these_probs]
                labels_per_ATQ[atq] = [these_labels]
            else:
                predictions_per_ATQ[atq].append(these_preds)
                probabilites_per_ATQ[atq].append(these_probs)
                labels_per_ATQ[atq].append(these_labels)
        
        probabilites_per_ATQ = dict(sorted({atq: np.average(np.vstack(values),axis=0) for atq, values in probabilites_per_ATQ.items()}.items()))
        predictions_per_ATQ = dict(sorted({atq: np.max(np.vstack(values),axis=0) for atq, values in predictions_per_ATQ.items()}.items()))
        labels_per_ATQ = dict(sorted({atq: np.max(np.vstack(values),axis=0) for atq, values in labels_per_ATQ.items()}.items()))
        return labels_per_ATQ, predictions_per_ATQ, probabilites_per_ATQ
    