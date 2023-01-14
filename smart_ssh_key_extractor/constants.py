import os

ROOT_DIR_PATH = "../Smart-VMI/data/new" # TODO: ??? remove
VALIDATION_DIR_PATH = os.environ["HOME"] + "/Documents/code/phdtrack/phdtrack_data/Validation"

MODEL_DIR_PATH = "./models"
LOG_DIR_PATH = "./logs"
RESULTS_PATH = "./results"
TYPES = ["client-side", "dropbear", "OpenSSH", "port-forwarding", "scp", "normal-shell"]

LENGTHS = [16, 24, 32, 64]
WINDOW_SIZE = 128
KEY_SIZE = 64
