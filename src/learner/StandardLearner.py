from src.loader.DatasetLoader import DatasetLoader
from src.learner.Evaluator import Evaluator
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import ExtraTreesClassifier as ETsC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.tree import ExtraTreeClassifier as ETC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore') 

import joblib
import os
import numpy as np



class StandardLearner:
    
    def __init__(self):
        
        self.dataset_loader = DatasetLoader()
        self.encoder = self.dataset_loader.encoder_transformer
        self.data_instances = self.dataset_loader.data_instances    
        self.models_path = self.dataset_loader.config.get('vectorizer', 'models_path')
        self.performance_path = self.dataset_loader.config.get('vectorizer', 'performance_path')
        self.multiclass_classifiers = ['mlp','logit', 'lsvc', 'randomforest', 'etc', 'etsc']   # ['randomforest'] # 
        self.one_vs_rest_classifiers = ['sgd', 'logit_ovr', 'gbc', 'lsvc_ovr'] 
        self.task = self.dataset_loader.config.get('vectorizer', 'task')
        self.target_label = self.dataset_loader.target_label
        self.split_type = self.dataset_loader.split_type
        self.split_balance = self.dataset_loader.split_balance
        self.classifier_type = self.dataset_loader.config.get('vectorizer', 'classifier_type')
        self.merge_performance = self.dataset_loader.config.getboolean('vectorizer','merge_performance')
        self.evaluator = Evaluator(self.data_instances, self.performance_path, self.merge_performance, self.dataset_loader.filter_names)

        
    def main(self):        
        header, output,tags = '','',''
        self.dataset_loader.get_data_instances(self.task, True, '')        
        print('Filters:', self.dataset_loader.filter_names)
        classifiers = self.one_vs_rest_classifiers if 'ovr' in self.classifier_type else self.multiclass_classifiers
        
        for classifer_name in classifiers:
            print('Setting up', classifer_name)
            for transformation in self.encoder.transformation_inputs:
                feat_len = 'feat'+ str(len(self.data_instances['train_X_feats']))
                tags = {'classifier': classifer_name,
                        'classifier_type': self.classifier_type if 'cv' in self.classifier_type else '',
                        'transformation': transformation,
                        'split_type': self.split_type,
                        'split_balance': self.split_balance,
                        'feat_len': feat_len,
                        'label_type': self.target_label,
                        'filters': self.dataset_loader.filter_names     
                        }                
                model_file = self.set_context(tags)
                classifier = self.train(tags, model_file)                    
                header, content, classification_metrics = self.predict(classifier, tags, '', '')
                output += content + '\n'
            print('Done.')
        
        self.evaluator.output_metrics(header, output, tags, self.task, self.classifier_type)
        
    
    def set_context(self, tags):        
        joined_tags = ('_').join([i for i in tags.values() if i])
        model_file = self.models_path + joined_tags
        model_file += '.model.pkl' 
        return model_file
    
    
    def train(self, tags, model_file):        
        classifier, params = self.set_up_classifier_parameters(tags['classifier'], self.target_label)    
        classifier = MultiOutputClassifier(classifier, n_jobs=6) if 'multi' in self.target_label else classifier
        # merge train and validation if cross valid
        train_data = self.data_instances['train_X'] 
        train_labels = self.data_instances['train_Y_'+ self.target_label] 
        
        transformed_data = self.encoder.transform_data(train_data, tags['transformation'], '')
        if(not os.path.isfile(model_file) or 'random' in self.split_type):    
            classifier.fit(transformed_data, train_labels)
            if('random' not in self.split_type):                
                joblib.dump(classifier, model_file)                    
        else:
            classifier = joblib.load(model_file)
                    
        return classifier


    def predict(self, classifier, tags, task, target_label):
        if(not task):
            task = self.task
        else:
            self.dataset_loader.get_data_instances(task, False, target_label)
        label_type = tags['label_type']#.split('_')[0]
        transformed_data = self.encoder.transform_data(self.data_instances[task+'_X'], tags['transformation'], '')
        true_labels = np.array(self.data_instances[task+'_Y_'+label_type])
        predictions = classifier.predict(transformed_data)
        probabilites = classifier.predict_proba(transformed_data)
        acc = classifier.score(transformed_data, true_labels)
        these_labels = list(self.data_instances[task+'_Y_'+label_type+'Mapping'].keys())
        if('multi' in label_type):
             probabilites = np.transpose([label[:, 1] for label in probabilites])
        
        probabilites, predictions, true_labels = self.evaluator.get_metrics_per_id(tags, task, these_labels, probabilites, 
                                                              predictions, true_labels)
            
        header, content, classification_metrics = self.evaluator.get_classification_metrics(tags, task, acc, probabilites, 
                                                                             predictions, true_labels) 
        
        feature_list = self.dataset_loader.data_instances['train_X_feats']
        #shap_rank = self.evaluator.shap_features(classifier.predict, transformed_data, tags, task, feature_list)
        
        if('predict' in task):
            return predictions, probabilites, these_labels
        else:
            return header, content, classification_metrics
    
    
    def set_up_classifier_parameters(self,classifier_name, label_type):
        params = {}
        if('lsvc' in classifier_name):
            params = {'C': 0.01, 'loss': 'squared_hinge', 'penalty': 'l2'}
            if('multi' in label_type):
                params['multi_class'] = 'crammer_singer'
            if('ovr' in classifier_name):
                params['multi_class'] = 'ovr'
            if (not 'grid' in self.task):
                model_svc = svm.LinearSVC(**params)
                return CalibratedClassifierCV(model_svc), params
            else:
                return svm.LinearSVC(probability=True, dual='auto'), params

        # in case positive perc is low, NuSVC nu param has to be adjusted
        elif('nusvc' in classifier_name):            
            if(not 'grid' in self.task):
                params = {'nu': 0.5, 'coef0': 0.01, 'gamma': 0.01, 'kernel': 'sigmoid'}
                return svm.NuSVC(**params, probability=True), params
            else:
                params = {'nu': 0.5,'probability':True}
                return svm.NuSVC(**params), params

        elif(classifier_name in 'svc'):
            # pass 'probability=True' if confidence values must be computed
            if (not 'grid' in self.task):
                params= {'C':100, 'gamma':0.001, 'kernel':'rbf', 'probability':True}
                return svm.SVC(**params), params
            else:
                return svm.SVC(probability=True), params
        elif('decisiontree' in classifier_name):
            return DT()

        elif ('svr' in classifier_name):
            return svm.SVR(), params
        
        elif('etc' in classifier_name):
            return ETC(), params
        
        elif('etsc' in classifier_name):
            return ETsC(), params
        
        elif('sgd' in classifier_name):
            model_sgd = SGD()
            return CalibratedClassifierCV(model_sgd), params

        elif('gbc' in classifier_name):
            return GBC(), params
        
        elif('gaussian' in classifier_name):
            return GPC(multi_class="one_vs_rest"), params

        elif ('mlp' in classifier_name):
            if (not 'grid' in self.task):
                params = {'max_iter':2000, 'activation':'relu', 'batch_size':256, 'hidden_layer_sizes':256, 'learning_rate':'adaptive', 'solver':'adam'}
                return MLP(**params), params
            else:
                params['max_iter']=2000
                return MLP(**params), params

        elif('randomforest' in classifier_name):
            if (not 'grid' in self.task):
                params = {'bootstrap':False, 'criterion':'entropy', 'max_features':'log2', 'n_estimators':1000}
                return RF(**params), params
            else:
                return RF(), params

        elif('logit' in classifier_name):
            params = {'max_iter': 5000, 'penalty': 'l1', 'C':10, 'solver':'saga'}
            if('multi' in label_type):
                params['multi_class'] = 'multinomial'
            elif('ovr' in classifier_name):
                params['multi_class'] = 'ovr'
            if (not 'grid' in self.task):
                return Logit(**params), params
            else:
                params['max_iter'] = 2000
                return Logit(**params), params            

if __name__ == '__main__':
    StandardLearner().main()