from constants import *

import os
import io
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CustomLogger:
    file_names: list[str]
    log_file: io.TextIOWrapper

    def __init__(self):
        self.file_names = []

        # log file
        if not os.path.exists(LOG_DIR_PATH):
            # Create the directory if it is not present
            os.makedirs(LOG_DIR_PATH)
        
        log_file_path = os.path.join(
            LOG_DIR_PATH, 
            # log_2023_12_31_23_59_59.txt
            "log_" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + ".txt"
        )
        self.log_file = open(log_file_path, "w")
    
    def __del__(self):
        """
        Destructor to close the log file
        """
        self.log_file.close()
    
    def log(self, print_str):
        """
        Print to the console and log file.
        """
        self.log_file.write(
            str(datetime.now()) + ":\t" + print_str + "\n"
        )
        print(str(datetime.now()) + ":\t" + print_str)

LOGGER = CustomLogger()
