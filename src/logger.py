import logging   #with the help of this package we can keep track of the log messages that are generated when the code is rum
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%s')}.log"   #! we are creating the last folder where our log file will be present

logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)   #! creating the path which include cwd-> "current working directory" inside a floder called logs then inside a folder LOG_FILE content that is made earlier

os.makedirs(logs_path, exist_ok=True)   #! making directory at created path and if it already exisits then don't throw any error

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)    #! first argument >> path, second argument >> log file is given

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    level=logging.INFO
)