import configparser
import os
import pandas as pd
import glob
import numpy as np
import json


def load_config():
    path = os.path.dirname(__file__)
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(path + '/../config.ini')
    return config


def list_files_per_ext(dir, extension):
    print(dir+extension)
    return glob.glob(dir + extension, recursive=True)


def get_id_from_filename(filename):
    temp = filename.split('/')[-1]
    temp = temp.split('.')[0]
    return temp


def remove_key(thisdict, key):
    thisdict.pop(key)
    return thisdict


def load_file_input(input_type, table_names, filename):       
        
    if('predict' in input_type):
        json_dataset,json_content = {}, {}
        data = {}
        with open(filename,'r') as f:
            json_content = json.loads(f.read())
        for key, value in json_content.items():
            json_dataset[key] = pd.json_normalize(value)      
        return json_dataset
        
    elif('dsa' in input_type):
        return pd.read_excel(filename, keep_default_na=False, sheet_name=table_names, dtype=object)
    else:
        return  pd.read_csv(filename, sep='\t', keep_default_na=False, dtype=object)



def read_file_lines(file):
    if(os.path.exists(file)):
        with open(file) as thisfile:
            return [line.rstrip('\n') for line in thisfile]
    else:
        print('File does not exist: ' + file)
        return ""


def dataset_to_dataframe(instances_per_split, categorical):
    dfs = []
    output = ''

    for values in instances_per_split.values():
        for value in values:
            instance = pd.DataFrame()                        
            try:
                temp = pd.read_csv(value, sep='\t',header=0)
                instance = temp.set_index('diag_id').T
            except:
                # array items are decomposed into multiple lines when doing df.from_dict
                value['target-health-multiclass'] = str(value['target-health-multiclass'])
                value.pop('AnimalId')
                idx = value.pop('diag_id')
                instance = pd.DataFrame.from_records(value, index=[idx]).replace('', np.nan)
            dfs.append(instance)  
            
    output = pd.concat(dfs)  
    output.columns = map(str.lower, output.columns)
    target_cols = ['target-lactationno','target-health-binary', 'target-health-multiclass', 'target-prod-multiclass', 'target-reprod-multiclass']
    labels = output[target_cols]
    
    output = output.drop(target_cols, axis=1)
    numerical = [i for i in list(output.columns) if i not in categorical]
    # handle potential missing tables from predict
    for i in categorical:
        if i not in output:
            output[i] = np.nan
    output[numerical] = output[numerical].astype('float')
    output[categorical] = output[categorical].astype('str')
    for i in categorical:
        output[i] = output[i].str.replace('.0','')
        
    return output, labels


def write_file(path, content):
    with open(path, 'w+') as file:
        # write content in a file, and return cursor to file position 0
        file.write(str(content))
        file.seek(0)
        file.close()
        





