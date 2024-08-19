
import pandas as pd

def normalize_data(dsa_data):
    animals = pd.concat([dsa_data.disease_animals,dsa_data.nondisease_animals], ignore_index=True)
    animals['ATQ'] = animals['ATQ'].astype(str)        
    animals = animals.to_dict('records')        
    animals = prepare_animals_records(animals, "dsa")        
    return animals


# prepare animals with only minimal shared fields
def prepare_animals_records(animals, db):
    result = []
    # iterate over each animal
    for animal in animals:                      
        # get corresponding IDs        
        temp = {}
        atq = str(animal['ATQ'])        
        temp['_id'] = atq
        temp['ATQ'] = atq
        temp['AnimalId'] = str(animal['AnimalId']) if "dsa" in db else ''      
        temp['farmId'] = animal['farmId']
        result.append(temp) if temp not in result  else ''   

    return result



# prepare subdocuments (tables) to match an animal entry in the collection
# df : respective dataframe obtained from loaders 
def prepare_subdocs(animals, data, collection_name):
    events = []
    for animal in animals:
        id_field = 'AnimalId' if 'dsa' in collection_name else 'ATQ'        
        ATQ = str(animal['ATQ'])        
        id = str(animal['AnimalId']) if 'dsa' in collection_name else ATQ        
        animal_subdoc = {}
            
        for table_name in data:            
            table = data[table_name]               
            # filter table with records for specific animal ID  
            table = table[(table[id_field] == id)]         
            # drop IDs to avoid redundancy
            table = table.drop(['AnimalId'], axis=1, errors='ignore')
            table = table.drop(['ATQ'], axis=1, errors='ignore')
            table = table.drop(['farmId'], axis=1, errors='ignore')
            
            rows = table.to_dict('records')
            if('Animals' in table_name):
                # adding further non-changeable attributes
                animal_subdoc['BreedCd']= rows[0]['BreedCd']
                animal_subdoc['BirthDate'] = rows[0]['BirthDate']                
            else:
                # add info to table if it already exists
                if(table_name in animal_subdoc.keys()):                  
                    animal_subdoc[table_name].update(rows)
                else:
                    animal_subdoc[table_name] = rows                    
        
        events.append([ATQ, animal_subdoc])
    return events