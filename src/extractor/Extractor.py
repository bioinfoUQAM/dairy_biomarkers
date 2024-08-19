import warnings;   warnings.filterwarnings("ignore")
from src.cutils import UtilMethods
from src.extractor import DSA

class Extractor:
    
    def __init__(self) -> None:
        self.config = UtilMethods.load_config()
        self.dsa_extractor = DSA.DSA(self.config)        
        
    
    def main(self):
        self.extract(task='', input_data='')
    
    
    def extract(self, task, input_data):
        self.dsa_extractor.extract(task, input_data)
   
            
        

if __name__ == '__main__':
    Extractor().main()