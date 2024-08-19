import pandas as pd
from src.cutils import UtilMethods

class DSA():    
    
    def __init__(self, config) -> None:
        self.dsa_file = config.get('extractor', 'dsa_db')
        self.attribute_file = config.get('extractor', 'db_attributes')
        self.label_codes_file = config.get('extractor', 'health_labels')
        self.metabolite_map_file = config.get('extractor', 'metabolite_mapping')
        self.metabolite_mapping = []
        self.target_animals_file = config.get('extractor', 'animal_list')
        self.input_files = [self.dsa_file, self.attribute_file, self.label_codes_file, self.target_animals_file]
        self.attributes_dsa = []
        self.tables_dsa = []
        self.att_list = {}
        self.dsa_data = []
        self.animals = []
        self.ATQ_list = []
        self.target_animal_ATQs = [] 
        self.target_animal_dsa_ids, self.ATQ_dsa_mapping = [],[]
        self.disease_animals, self.nondisease_animals = [], []
        self.prediction_history_attributes = {'MilkWeights': ['HerdMilkWeight','HerdAverageMilkWeight','HerdMaxMilkWeight'],
                                       'AnimalMilkCompositions': ['HerdFat', 'HerdAverageFat', 'HerdCrudeProtein', 'HerdAverageCrudeProtein'],
                                       'HistoryAttributes': ['LactationStartDate', 'LactationNo', 'LactationEndDate', 'HealthsDuringLactation',
                                                             'MilkWeightsDuringLactation','AnimalMilkCompositionsDuringLactation']}
        self.label_codes = []
    
    def extract(self, task, input_data):        
        self.set_context(task)    
        self.load_data(task, input_data)
        self.map_ATQ_dsa_ids(task)
        self.merge_drug_info()
        self.prep_lab_results()
            
        self.add_lactation_ids()
        self.set_labels()
        self.disease_animals = self.select_per_label('disease')
        self.nondisease_animals = self.select_per_label('nondisease')
        self.get_dsa_attributes(task)
               
            
    def prep_lab_results(self):        
        # keeps only rows for which targeted metabolite results were found
        lab_results = pd.merge(self.dsa_data['LabResults'],self.metabolite_mapping,on='ComponentId', how='left')
        lab_results = lab_results[pd.to_numeric(lab_results.Result, errors='coerce').notnull()]
        lab_results['Result'] = lab_results['Result'].astype(float)        
        lab_results = pd.pivot_table(lab_results, values = 'Result', index=['ResultDate','AnimalId','SourceId'], columns = 'AnalyseSource').reset_index()        
        # add missing metabolite columns if not found
        lab_results = lab_results.reindex(columns=self.att_list['LabResults'])
        self.dsa_data['LabResults'] = lab_results
        
        
    # defines attribute list to be used
    def set_context(self, task) -> None:
        self.metabolite_mapping = pd.read_csv(self.metabolite_map_file, sep='\t')
        self.metabolite_mapping['AnalyseSource'] = self.metabolite_mapping['Analyse'] + '-' + self.metabolite_mapping['SourceId'].apply(str)
        self.metabolite_mapping = self.metabolite_mapping.drop('SourceId', axis=1) 
        attributes = UtilMethods.load_file_input('attributes', self.tables_dsa, self.attribute_file)
        # update `label` as an attribute to be used
        attributes.loc[len(attributes.index)] = ['label', 'DSA', 'Healths']
        attributes.loc[len(attributes.index)] = ['diag_id', 'DSA', 'Healths']
        attributes.loc[len(attributes.index)] = ['l_id', 'DSA', 'StartLactations']
        self.attributes_dsa = attributes.loc[attributes['OriginDB'] == 'DSA'] 
        tables_dsa = self.attributes_dsa['OriginTable'].unique()
        self.tables_dsa = tables_dsa.tolist()
        #Listing the attributes to be used from each DSA table
        for table in self.tables_dsa:    
            table_attributes = self.attributes_dsa.loc[self.attributes_dsa['OriginTable'] == table]['Attribute'].tolist()            
            self.att_list[table] = table_attributes
        self.att_list['SailliesCalculees'] = [item.replace('Date','SailliesDate') for item in self.att_list['SailliesCalculees']]
        self.att_list['LabResults'] = ['AnimalId', 'ResultDate', 'SourceId'] + self.metabolite_mapping['AnalyseSource'].values.tolist()
        # adding history attributes retrieved for prediction
        if('predict' in task):
            for key in self.prediction_history_attributes.keys():
                if(key not in self.tables_dsa):
                    self.tables_dsa.append(key)
                    self.att_list[key] = []
                self.att_list[key] += self.prediction_history_attributes[key]
        
    
    
    def load_data(self, task, input_data):
        input_data = input_data if input_data else self.dsa_file
        #input_type = 'json' if 'predict' in task else 'dsa'
        input_type = 'dsa-predict' if 'predict' in task else 'dsa'
        self.dsa_data = UtilMethods.load_file_input(input_type,self.tables_dsa,input_data)
        # normalize columns
        self.dsa_data['AnimalMilkCompositions'] = self.dsa_data['AnimalMilkCompositions'].replace([' ','  '],'')
        self.dsa_data['LactationYields'] = self.dsa_data['LactationYields'].replace([' ','  '],'') 
        self.dsa_data['MilkWeights'] = self.dsa_data['MilkWeights'].replace([' ','  '],'')
        self.dsa_data['SailliesCalculees'] = self.dsa_data['SailliesCalculees'].rename(columns={'Date': 'SailliesDate'})
    
    
    def map_ATQ_dsa_ids(self, task):
        self.animals = self.dsa_data['Animals']
        self.animals = self.animals.rename(columns={'PermId': 'DSAId'})
        self.animals['DSAId'] = self.animals['DSAId'].astype(str)
        self.animals = self.animals[self.animals['DSAId'].str.strip().astype(bool)]
        target = [ int(i) for i in self.animals['DSAId'].to_list() ] if 'predict' in task else ''   
        
        target_animals_df = self.get_target_animals_list(target)   
        target_animals_df = target_animals_df[['DSAId','ATQ', 'farmId']].astype(str)
        # add project start date for animals that have one
        self.animals = pd.merge(self.animals,target_animals_df,on='DSAId', how='left') 
        self.animals['ATQ'] = self.animals['ATQ'].astype(object)
        self.ATQ_dsa_mapping = self.animals[['ATQ', 'Id', 'farmId']]
        self.ATQ_dsa_mapping = self.ATQ_dsa_mapping.rename(columns={"Id": "AnimalId"})
        
        mask_ATQ = self.ATQ_dsa_mapping['ATQ'].isin(self.target_animal_ATQs)
        self.ATQ_dsa_mapping = self.ATQ_dsa_mapping.loc[mask_ATQ]
        self.ATQ_dsa_mapping['ATQ'] = self.ATQ_dsa_mapping['ATQ'].astype(str)
        self.ATQ_list = self.animals['ATQ'].to_list()
        
    
    def get_target_animals_list(self, target):
        target_animals_df = pd.read_csv(self.target_animals_file, sep='\t')
        target_animals_df = target_animals_df.rename(columns={'DSA': 'DSAId','LBIO': 'ATQ'})
        # select IDs if task is 'predict'
        if(target):
            mask_target = target_animals_df['ATQ'].isin(target)
            target_animals_df = target_animals_df.loc[mask_target]
            # if ATQ is not found in list of project animals            
            if(target_animals_df.empty):
                temp = {'DSAId': target, 'ATQ': target, 'farmId': ['unk' for i in target]}
                target_animals_df = pd.DataFrame.from_dict(temp, dtype=str)            
        self.target_animal_ATQs = target_animals_df['ATQ'].to_list()
        return target_animals_df
    
    
    def merge_drug_info(self):
        dsa_drugs = self.dsa_data['Drugs']
        dsa_drugs = dsa_drugs[['Id','Description1']]
        dsa_drugs = dsa_drugs.rename(columns={'Id': 'DrugId'})
        self.dsa_data['Treatments'] = pd.merge(self.dsa_data['Treatments'], dsa_drugs, on = 'DrugId', how = 'left')
        self.att_list['Treatments'].append('Description1')
        #self.dsa_data['Treatments'].drop('DrugId',axis=1,inplace=True)
        cleanup = ['Drugs'] #,'MilkLabResults']
        for table in cleanup:
            self.dsa_data.pop(table, None)    
            self.att_list.pop(table,None)
            self.tables_dsa.remove(table) if table in self.tables_dsa else self.tables_dsa
            
    
    # Handle cases for which animal was reformed before the project lactation.
    # In such cases, synthetic StartLactation and LactationEndDate dates for history data were created based on:
    # - LactationEndDate: equivalent to AnimalExits | LeftHerdDate
    # - StartLactation: -30 days before metSampleDate (therefore covering drying period - tarissement)
    def set_labels(self):
        # select the 'Healths' table from troupeau file
        dsa_healths = self.dsa_data['Healths'].copy(deep=True)
        dsa_healths['ObservationDate'] = pd.to_datetime(dsa_healths['ObservationDate'])
        dsa_healths['HealthCd'] = dsa_healths['HealthCd'].astype(str)
        dsa_healths.insert(loc=0, column='label', value='')        
    
        self.label_codes =  UtilMethods.load_file_input('label',self.tables_dsa,self.label_codes_file)
        # assign disease label to events matching disease codes
        dsa_healths.loc[dsa_healths['HealthCd'].isin(self.label_codes.code), ['label']] = 'disease'
        # assign non-disease label to events not matching disease codes
        dsa_healths.loc[~dsa_healths['HealthCd'].isin(self.label_codes.code), ['label']] = 'nondisease'

        # add ATQ to Healths to create a diag_id
        dsa_healths = pd.merge(dsa_healths,self.ATQ_dsa_mapping[['AnimalId', 'ATQ']], on='AnimalId',how='left')

        # create a diagnosis id to make diagnosis unique
        dsa_healths['diag_id'] = dsa_healths['ATQ'] + '_' + dsa_healths['ObservationDate'].dt.strftime('%Y%m%d') + '_' + dsa_healths['HealthCd']

        # update `label` as a new attribute for `Healths`
        self.dsa_data['Healths'] = dsa_healths.copy(deep=True) 
    
    
    def select_per_label(self, label):
        dsa_healths = self.dsa_data['Healths']
        # find animals associated or not with health codes of relevant diseases
        dsa_ids_per_label = dsa_healths.loc[dsa_healths['label'] == label]      
        dsa_ids_per_label = dsa_ids_per_label[['AnimalId','ObservationDate','HealthCd']]
        # map DSA AnimalId and ATQs
        ATQ_per_label = dsa_ids_per_label.merge(self.ATQ_dsa_mapping, on='AnimalId', how='inner')

        ATQ_list = [str(i) for i in self.ATQ_list]
        # filter animals that belong to the project
        mask_ids = ATQ_per_label['ATQ'].isin(ATQ_list)
        animals_per_label = ATQ_per_label.loc[mask_ids]
        return animals_per_label
        
        
    def add_lactation_ids(self):
        dsa_lactations = self.dsa_data['StartLactations'].copy(deep=True)
        dsa_lactations = pd.merge(dsa_lactations,self.ATQ_dsa_mapping[['AnimalId', 'ATQ']], on='AnimalId',how='left')
        dsa_lactations['StartDate'] = pd.to_datetime(dsa_lactations['StartDate'])
        dsa_lactations['l_id'] = dsa_lactations['ATQ'] + '_' + dsa_lactations['StartDate'].dt.strftime('%Y%m%d') + '_' + dsa_lactations['LactationNo'].astype(str)
        self.dsa_data['StartLactations'] = dsa_lactations.copy(deep=True) 

    
    def get_dsa_attributes(self,task):
        self.dsa_data['Animals'] = self.dsa_data['Animals'].rename(columns={'Id': 'AnimalId'})
        self.att_list['Animals'] = ['AnimalId' if item == 'Id' else item for item in self.att_list['Animals']]
        dsa_list = []
        if('predict' in task):
            dsa_list = self.dsa_data['Animals']['AnimalId'].unique().tolist()
        else:
            dsa_list = self.ATQ_dsa_mapping['AnimalId'].unique().tolist()

        # iterate over tables and attributes to be used
        for table, attribute in self.att_list.items():      
            this_table = pd.DataFrame()
            try:                
                this_table = self.dsa_data[table][attribute]  # prompts non-existing column                
                mask_dsa = this_table['AnimalId'].isin(dsa_list)
                this_table = this_table.loc[mask_dsa]                  
            except:
                print('Attribute missing in DSA|' + table)
            # handle empty tables (potentially from predict samples) 
            if(this_table.empty):
                this_table = pd.DataFrame(columns = attribute)
                this_table['AnimalId'] = dsa_list
            self.dsa_data[table] = this_table            
            del this_table      
    
    
    
        