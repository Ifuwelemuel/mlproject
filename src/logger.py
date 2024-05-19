import logging
import os 
from datetime import datetime
#creating log file
LOG_FILE= f"{datetime.now().strftime('%m_%d%_%Y_%H_%S')}.log" 
#with respect to loggs path 
#basically adding logs path
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
#keep appending logs even though there is a file
os.makedirs(logs_path,exist_ok=True)


LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    
  
)

if __name__=="__main__":
    logging.info("logging has started")