from src.cutils import UtilMethods
import os
from src.vectorizer.EncoderTransformer import EncoderTransformer
import pandas as pd
from typing import List
from src.extractor import DSA
import numpy as np
import ast

class DatasetLoader:
    
    def __init__(self) -> None:
        
        self.config = UtilMethods.load_config()
        self.data_path = self.config.get('default', 'data_dir')
        self.dataset_path = self.config.get('loader','data_instances')
        self.labels_file = self.data_path + '/dataset_labels'
        self.feature_file = self.config.get('loader', 'feat_dir') + '/features'
        self.train_perc = self.config.get('loader', 'train_perc')
        self.valid_perc = self.config.get('loader', 'valid_perc')
        self.test_perc = self.config.get('loader', 'test_perc')
        self.label_codes = self.config.get('extractor', 'health_labels')
        self.analysis_path = self.config.get('vectorizer', 'analysis_path')
        self.models_path = self.config.get('vectorizer', 'models_path')
        self.feature_list =  UtilMethods.read_file_lines(self.feature_file)
        self.dataset_perc = { }
        self.feat_filters = self.config.get('loader', 'feat_filters').lower()
        self.filter_names = ''      
        self.features = []
        self.dataset = pd.DataFrame()
        self.dataset_labels = pd.DataFrame()
        self.health_cd_map = self.map_health_codes()
        self.label_map = {}
        self.instances_per_split = {}
        self.instances_all = []
        self.split_type = self.config.get('loader', 'split_type')
        self.split_balance = self.config.get('loader', 'split_balance')
        self.target_label = self.config.get('loader', 'target_label')
        self.data_instances = {'split_type' : self.split_type}
        self.encoder_transformer = EncoderTransformer(self.config)
        self.dsa_extractor = DSA.DSA(self.config)
       
        
    
    def set_filter_names(self):
        filter_names = ''       
        if('rare' in self.feat_filters and 'rare' not in filter_names.lower()):
            filter_names += '_noRare'
        if ('health' in self.feat_filters and 'health' not in filter_names.lower()):
            filter_names += '_noHealth'
        if ('farm' in self.feat_filters and 'farm' not in filter_names.lower()):
            filter_names += '_noFarm'    
        if ('dsa' in self.feat_filters and 'dsa' not in filter_names.lower()):
            filter_names += '_noDSA'            
        if ('nmr' in self.feat_filters and 'nmr' not in filter_names.lower()):
            filter_names += '_noNMR'
        if ('metab' in self.feat_filters and 'metab' not in filter_names.lower()):
            filter_names += '_noMetab'
        
        return filter_names[1:]      
    
    
    def set_context(self, task, export_complete):
        self.dsa_extractor.set_context(task)
        self.metab_feats = [i.lower() for i in self.dsa_extractor.metabolite_mapping['AnalyseSource'].to_list()]
        self.nmr_feats = [i for i in self.metab_feats if '-10' in i]
        #self.dsa_feats = [i for i in ]
        
        self.filter_names = self.set_filter_names()
        self.dataset_perc = {'train': self.train_perc, 'valid': self.valid_perc, 'test': self.test_perc}        
        if('predict' not in task):
            self.split(task)          
        if(self.dataset.empty):
            self.load(export_complete)         
        self.map_dataset_labels(task) 
    
    
    def get_data_instances(self, task, export_complete, target_label):      
        
        if(not target_label):
            target_label = self.target_label
            
        self.set_context(task, export_complete)              
        
        for split, instances in self.instances_per_split.items():            
            instance_ids = self.dataset.index.values.tolist() if 'predict' in task else [UtilMethods.get_id_from_filename(i) for i in instances]
            self.data_instances[split+'_ids'] = instance_ids
    
            # select instance IDs from datasets
            task_dataset = self.dataset[self.dataset.index.isin(instance_ids)]  
            self.normalize_features(split, task_dataset.columns.values.tolist())
                                  
            dense_encode = True if('rank' in task) else split                
            task_dataset, processed_features = self.pre_process(split, dense_encode, task_dataset)
    
            self.data_instances[split+'_X'] = task_dataset
            self.data_instances[split+'_X_feats'] = processed_features        
    
            for this_type, mapping in self.label_map.items():
                if(this_type not in self.encoder_transformer.label_encoders):
                    self.encoder_transformer.label_encoders[this_type] = ''
                encode_type = this_type 
                if('disease' in self.split_balance):
                    encode_type += '-disease'
                # encode labels
                encoded_labels, mapping = self.encoder_transformer.label_encode(instance_ids, mapping, encode_type, target_label)
                self.data_instances[split+'_Y_'+this_type] = encoded_labels    
                self.data_instances[split+'_Y_'+this_type+'Mapping'] = mapping
            
            print(split, 'instances:', self.data_instances[split+'_X'].shape[0], 'labels:', len(encoded_labels), 'features:', len(self.data_instances[split+'_X_feats']))
        
    
    def normalize_features(self, task, task_dataset_cols):
        task_dataset_cols = [i.lower() for i in task_dataset_cols]  
        rare_feats = ['trueprotein', 'peakmilkyld', 'cumulmilkvalue']        
        task_features_file = self.feature_file+'_'+ self.split_type +'_'+ self.split_balance +'_'+self.filter_names
        if(not os.path.isfile(task_features_file)):
            self.features = task_dataset_cols   
            if('train' in task):
                if('rare' in self.feat_filters):                    
                    self.features = [ i for i in task_dataset_cols if i not in rare_feats]                    
                if('farm' in self.feat_filters):
                    self.features.remove('farmid')                    
                if('healthcd' in self.feat_filters):
                    self.features.remove('healthcd')
                if('dsa' in self.feat_filters):
                    self.features = [ i for i in self.features if i in self.metab_feats]                   
                if('nmr' in self.feat_filters):
                    self.features = [ i for i in self.features if i not in self.nmr_feats]
                if('metab' in self.feat_filters):                    
                    self.features = [ i for i in self.features if i not in self.metab_feats]                   
                            
                UtilMethods.write_file(task_features_file, '\n'.join(self.features))
            else:
                raise Exception('Feature file not found. Please train model before proceeding.')
        else:
            self.features = UtilMethods.read_file_lines(task_features_file)
                
        

    def load(self, 
             export_complete: bool) -> None:  
        self.dataset, self.dataset_labels = UtilMethods.dataset_to_dataframe(self.instances_per_split, self.get_categorical_attributes())        
        if(export_complete):
            complete_dataset_name = 'completeDataset' + self.filter_names
            complete_dataset_Path = self.analysis_path + complete_dataset_name
            if not os.path.isfile(complete_dataset_Path):
                self.dataset.to_csv(complete_dataset_Path, sep='\t')
                self.dataset.describe().to_csv(self.analysis_path + complete_dataset_name + '_stats', sep='\t')

        print('Done loaded dataset.')

    
    # gets list of categorical attributes
    def get_categorical_attributes(self):
        categ_cols = [i for i in self.features if ('Cd' in i) ]
        categ_cols = categ_cols + ['farmId'] if 'farmid' not in self.feat_filters else categ_cols
        categ_cols = set(categ_cols + ['HealthCd', 'DrugId', 'Code', 'IsPositive', 'ServiceType', 'Description1'])        
        categ_cols = [i.lower() for i in categ_cols]
        return categ_cols
    
    
    # generates map between healthCd 
    # and disease names
    def map_health_codes(self):
        label_codes = pd.read_csv(self.label_codes, sep='\t', keep_default_na=False, dtype=object)
        return dict(zip(label_codes['code'],label_codes['maladie'])) 

    # assigns binary or multiclass label 
    # to dataset instances
    def map_dataset_labels(self, task):        
        
        if('predict' in task):
            self.dataset_labels[self.target_label] = 'nondisease' if 'health' in self.target_label else 'normale' if 'reprod' in self.target_label else '0'
        else:
            self.label_map[self.target_label] = self.dataset_labels['target-' + self.target_label].to_dict()
            if('multi' in self.target_label):
                self.label_map[self.target_label] = self.dataset_labels['target-' + self.target_label].to_dict() 
        
                for k,v in self.label_map[self.target_label].items():
                    if (v and 'non' not in v and 'health' in self.target_label):
                        health_codes = ast.literal_eval(v)
                        health_code_names = set([self.health_cd_map[i] for i in health_codes if i in self.health_cd_map.keys()])
                        self.label_map[self.target_label][k] = health_code_names
                    else:
                        self.label_map[self.target_label][k] = set([v])
        
        
    # outputs a list of IDs that should be used for train, valid, test
    # if task ID list already exists, returns existing list
    def split(self, task):
        split_path, dataDir = self.load_splits(task)              
        dataset_ext = 'disease.vet' if 'disease' in self.split_type else 'vet'
        if(split_path):    
            for task, perc in self.dataset_perc.items():      
                # if split exists, just read list
                one_task_path = self.models_path + '/' + task + '_' + self.split_type +'_'+ self.split_balance + '_' + self.target_label + perc + '.split'
                if (os.path.exists(one_task_path)):
                    print(task, 'split already done. Loading dataset...')
                    split_files = UtilMethods.read_file_lines(one_task_path)                      
                    self.instances_per_split[task] = split_files
                                   
        # if split doesnt exist or if random split
        else:
            #data_instances = UtilMethods.list_files_per_ext(dataDir, "/*." + dataset_ext)
            #splitframe = pd.DataFrame(data_instances, columns=['filename'])  
            splitframe = pd.read_csv(self.labels_file, sep='\t', header=0, converters={'target-health-multiclass': ast.literal_eval})
            splitframe['diag_id'] = splitframe['instance'].str.split('.').str[0]
            splitframe['diag_id'] = splitframe['diag_id'].str.split('/').str[-1]
            splitframe[['ATQ', 'farmId', 'date', 'healthCd', 'velap', 'days', 'prod', 'reprod']] = splitframe['diag_id'].str.split('_', expand=True)
                                
            # define the classification target: health, production or reproduction  
            target_label = ''                              
            if('health' in self.target_label):
                target_label = 'target-health-binary' if 'binary' in self.target_label else 'target-health-multiclass-map'
                splitframe['target-health-multiclass-map'] = splitframe['target-health-multiclass'].explode().map(self.health_cd_map).groupby(level=0).agg(list)
                splitframe['target-health-multiclass-map'] = splitframe['target-health-multiclass-map'].apply(lambda y: ['nondisease'] if y==[np.nan] else y)
                if('disease' in self.split_balance):
                    splitframe = splitframe[~splitframe['target-health-binary'].str.contains("nondisease")]                
            else:
                target_label = 'target-reprod-multiclass' if 'reprod' in self.target_label else 'target-prod-multiclass' if 'prod' in self.target_label else ''
                
                if('reprod' in self.target_label):
                    splitframe = splitframe[~splitframe[target_label].str.contains("unknown")]            
            
            # diag_id doesnt consider velap days so that same profile wont be repeated in splits
            splitframe['diag_id'] = splitframe['ATQ'] + '_' + splitframe['farmId'] + '_' + splitframe['date'] 
            # keeping one entry per day for prod and reprod
            splitframe.drop(columns=['velap', 'days'], inplace=True)
            if('health' not in self.target_label):
                splitframe = splitframe.drop_duplicates(subset='diag_id', keep="first")
            
            split_field = 'ATQ' if 'animal' in self.split_type else 'farmId' if 'farm' in self.split_type else 'diag_id'
            # select unique split fields per label
            groups = splitframe.explode(target_label).groupby(target_label)[split_field].agg(['unique'])
            # sort groups per lenght of unique fields (make sure less common labels have representative instances)
            this_list_sorted = sorted(groups['unique'].to_list(), key=lambda x: len(x))
            # get min size among all groups
            smallest_group = len(min(this_list_sorted, key=len))
            train, valid, test = set(), set(), set()
            prior = np.empty([0,])
            
            for item in this_list_sorted:                        
                this_item = np.setdiff1d(item, prior)
                # if balanced, random select TRAIN only ATQs based on smallest_group size
                sample_size = len(this_item) if 'balance' not in self.split_balance else smallest_group
                ids = np.random.choice(this_item, int(sample_size * (int(self.train_perc) / 100)), replace=False)
                train.update(ids.tolist())
                remain = np.setdiff1d(this_item, ids)                        
                ids = np.random.choice(remain, int(len(remain) * 0.5), replace=False)
                valid.update(ids.tolist())
                remain = np.setdiff1d(remain, ids)
                ids = np.random.choice(remain, int(len(remain)), replace=False)
                test.update(ids.tolist())
                #prior = np.append(prior,item)
                prior = np.append(prior,this_item)
                 
            self.instances_per_split['train'] = splitframe[splitframe[split_field].isin(train)]['instance'].values.tolist()
            self.instances_per_split['valid'] = splitframe[splitframe[split_field].isin(valid)]['instance'].values.tolist()
            self.instances_per_split['test'] = splitframe[splitframe[split_field].isin(test)]['instance'].values.tolist()
            
            for task, values in self.instances_per_split.items():
                task_path = self.models_path + '/' + task + '_' + self.split_type +'_'+ self.split_balance + '_' + self.target_label + self.dataset_perc[task] + '.split'
                UtilMethods.write_file(task_path, '\n'.join(values))
                
        return self.instances_per_split.values()
                    
    
    def load_splits(self, task):
        data_dir = self.dataset_path
        task = 'train' if 'rank' in task else task
        this_ext = "/" + task + '_' + self.split_type + '_'+ self.split_balance + '_' + self.target_label +"*.split"
        split_path = UtilMethods.list_files_per_ext(self.models_path, this_ext) 
        return split_path, data_dir
    
    
    # compute stats on the dataset
    def get_stats(self):    
        stats_df = pd.DataFrame()  
        stats_dict = {}
        total = len(self.dataset.index)
        categ_cols = self.get_categorical_attributes()
        for col in self.dataset.columns:   
            if(col not in categ_cols):
                missing = self.dataset[col].isnull().sum()        
                col_stats = self.dataset[col].astype(float).describe().astype(float)
                col_stats['missing'] = missing 
                col_stats['missing%'] = (missing * 100) / total
                stats_dict[col] = col_stats
        
        temp = stats_dict.values()
        stats_df = pd.concat(temp, axis=1, ignore_index=True)
        stats_df.columns = stats_dict.keys()
        
        return stats_df

        
        
    # pre-process, one-hot encode category attributes       
    def pre_process(self, task, dense_encode, task_dataset):  
        pd.options.mode.chained_assignment = None  # default='warn'
        task_columns = task_dataset.columns.values.tolist()
        missing = [i for i in self.features if i not in task_columns]
        if(missing):
            for i in missing:
                task_dataset[i] = np.nan
        task_dataset = task_dataset[self.features]
                
        # use only categ cols that were present in train
        categ_cols = self.get_categorical_attributes()
        categ_cols = [i for i in categ_cols if i in self.features]
        
        # normalize categ columns to string type
        # before transforming to one hot vecs
        for categ_col in categ_cols:
            task_dataset[categ_col] = task_dataset[categ_col].fillna(0)
            try:
                # farmId needs to be converted to float so all values are uniform
                # and str convertion will be equally applied to all values            
                task_dataset[categ_col] = task_dataset[categ_col].astype(int)    
            except:
                pass
            task_dataset[categ_col] = task_dataset[categ_col].astype(str)    
        
        # transform remaining columns to float
        floats = list(set(self.features) - set(categ_cols)) 
        task_dataset[floats] = task_dataset[floats].astype(float)                
        encoded_dataset, processed_features = self.encoder_transformer.onehot_encode(dense_encode, task_dataset, categ_cols, self.features, self.filter_names)
        processed_features = [i.split('__')[1] for i in processed_features]
        
        return encoded_dataset, processed_features 
            
    
if __name__ == '__main__':
    DatasetLoader().get_data_instances(task='train', export_complete=False, target_label='')
    
    
