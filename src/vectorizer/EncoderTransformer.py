from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import *
import numpy as np
import joblib
import os

class EncoderTransformer:
    
    def __init__(self, config):
        self.encoder_path = config.get('vectorizer', 'encoders_path')
        self.onehot_encoder_name = 'onehot_encoder'
        self.encoder_ext = '.pkl'
        self.onehot_encoder = ''
        self.label_encoders = {}
        self.split_type = config.get('loader', 'split_type')
        self.transformation_inputs = ['NaN0', 'NaN0-StdScaled', 'NaN0-Normalized']

        self.imputters = {}    
    
    
    def load_encoder(self, encoder_name):
        encoder_path = self.encoder_path + encoder_name + self.encoder_ext
        encoder_type = encoder_name.split('_')[0]
        if(os.path.exists(encoder_path)):
            if('onehot' in encoder_name):
                self.onehot_encoder = joblib.load(encoder_path)
            else:
                self.label_encoders[encoder_type] = joblib.load(encoder_path)
            return True
        else:
            return False
        
    
    def label_encode(self, dataset_ids, all_labels, encoder_type, target_label):
        encoder = ''
        encoder_name = encoder_type
        label_with_names = [all_labels[i] for i in dataset_ids]
        if(self.load_encoder(encoder_name)):
            encoder = self.label_encoders[encoder_type]
        else:
            encoder = LabelEncoder() if 'binary' in encoder_type else MultiLabelBinarizer()    
            encoder = encoder.fit(label_with_names)   
            self.label_encoders[encoder_type] = encoder   
            joblib.dump(self.label_encoders[encoder_type], self.encoder_path  + encoder_name + self.encoder_ext, compress=9)  
                    
        encoded_labels = encoder.transform(label_with_names).tolist()
        # for multilabel, input has to be in format:
        # [{}]
        encoder_classes = encoder.classes_ if 'binary' in encoder_type else [set([i]) for i in encoder.classes_]
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder_classes)))  
       
        return encoded_labels, mapping
    
    
    
    def onehot_encode(self, dense_encode, task_dataset, categ_cols, features, filterNames):
        encoded_dataset, processed_features = '',''    
        featLen = 'feat'+str(len(features))
        if(dense_encode):
            self.onehot_encoder_name = self.onehot_encoder_name if 'dense' in self.onehot_encoder_name else self.onehot_encoder_name + '_dense_' 
        self.onehot_encoder_name = self.onehot_encoder_name if 'split' in self.onehot_encoder_name else self.onehot_encoder_name + 'split'+ self.split_type + '_'     
        self.onehot_encoder_name = self.onehot_encoder_name if featLen in self.onehot_encoder_name else self.onehot_encoder_name + featLen      
        self.onehot_encoder_name = self.onehot_encoder_name if filterNames in self.onehot_encoder_name else self.onehot_encoder_name + filterNames
        if(self.load_encoder(self.onehot_encoder_name)):
            encoded_dataset = self.onehot_encoder.transform(task_dataset)
        else:
            this_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) if dense_encode else OneHotEncoder(handle_unknown='ignore') 
            self.onehot_encoder = make_column_transformer((this_encoder, categ_cols,), remainder='passthrough')
            encoded_dataset = self.onehot_encoder.fit_transform(task_dataset)
            joblib.dump(self.onehot_encoder, self.encoder_path + self.onehot_encoder_name + self.encoder_ext, compress=9)
        
        processed_features = self.onehot_encoder.get_feature_names_out().tolist()

        return encoded_dataset, processed_features
    
    
    # Transformation variations: constant (0) or mean values to replace NaNs, standard scale, or normalization
    def get_imputter_obj(self, transformation, format):
        # input column mean to NaN values
        if('NaNMean' in transformation):            
            imputter = SimpleImputer(missing_values=np.nan, strategy='mean',fill_value=0)

        # input zeros to NaN values
        if('NaN0' in transformation):
            imputter = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

        # transform values to standard scale (based on mean)
        if('StdScaled' in transformation):
            # with_mean for sparse matrices
            imputter = StandardScaler(with_mean=False)        

        # transform values to normalized
        if('Normalized' in transformation):
            imputter = Normalizer()

        if('pandas' in format):
            imputter.set_output(transform="pandas")
        
         
        return imputter



    def transform_data(self, this_dataset, transformation, format):
        imputter, result  = '', np.array([])
        #imputter, result  = '', ''
        for transf in transformation.split('-'):
            isnp = isinstance(result,np.ndarray)
            #if(isnp):
            # not just numpy.empty because it can also be scipy csc sparse
            result = result if result.shape[0] > 0 else this_dataset
            #else:                
                #result = result if not result.empty else this_dataset
            if(transf not in self.imputters):
                imputter = self.get_imputter_obj(transf, format)
                self.imputters[transf] = imputter.fit(result)            
            
            imputter = self.imputters[transf]        
            result = imputter.transform(result)        
            
        return result
        