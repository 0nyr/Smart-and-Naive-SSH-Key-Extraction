"""
File that contains all commonly used functions.
"""

import numpy as np
import os
import io
import json
from datetime import datetime
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
from enum import Enum

ROOT_DIR_PATH = "../Smart-VMI/data/new" # TODO: ??? remove
VALIDATION_DIR_PATH = os.environ["HOME"] + "/Documents/code/phdtrack/phdtrack_data/Validation"

MODEL_DIR_PATH = "./models"
LOG_DIR_PATH = "./logs"
RESULTS_PATH = "./results"
TYPES = ["client-side", "dropbear", "OpenSSH", "port-forwarding", "scp", "normal-shell"]

LENGTHS = [16, 24, 32, 64]
WINDOW_SIZE = 128
KEY_SIZE = 64

@dataclass
class ConfigParameter:
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

class WrappedClassifier:
    def __init__(self, 
        resampled_classifier: RandomForestClassifier, 
        classifier: RandomForestClassifier, 
        final_stage_classifier: RandomForestClassifier
    ):
        self.resampled_classifier = resampled_classifier
        self.classifier = classifier
        self.final_stage = final_stage_classifier

    def predict(self, x):
        test_rc = self.resampled_classifier.predict_proba(x)
        test_clf = self.classifier.predict_proba(x)

        transformed_input = np.hstack((test_rc, test_clf))
        results = self.final_stage.predict(transformed_input)
        return results

    def return_classifiers(self):
        return self.classifier, self.resampled_classifier, self.final_stage


PARAMS = ConfigParameter()

def log(print_str):
    """
    Print to the console and log file.
    """
    PARAMS.log_file.write(
        str(datetime.now()) + ":\t" + print_str + "\n"
    )
    print(str(datetime.now()) + ":\t" + print_str)


def read_keys(path):
    """
    This function reads the keys from the key file.

    Structure of the key file is as follows:
    KEY 0:
    00000000: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    00000010: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    00000020: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    KEY 1:
    00000000: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    00000010: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    00000020: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    ...
    """

    with open(path, "r") as file:
        data: list[str] = file.readlines()

    # Extract upto 6 keys
    keys: list[bytearray] = []
    temp_key = bytearray()
    for row in data:
        curr_row = row.strip()
        if len(curr_row) > 0: # not empty row
            if 'KEY' in curr_row:
                # key start
                if len(temp_key) > 0:
                    keys.append(temp_key)
                temp_key = bytearray() # reset when changing key
            else:
                curr_row = curr_row[23:].strip()
                temp_key = temp_key + bytearray.fromhex(curr_row)

    if len(temp_key) > 0:
        keys.append(temp_key)
    return keys

class BlockType(Enum):
    INVALID = 0 # More than one key
    VALID = 1 # One and only one key
    EMPTY = 2 # No key

@dataclass
class BlockData:
    dataset: bytearray
    label: int # if VALID then 1 else EMPTY 0
    offset: int
    length: int

def get_block_data_from_keys_in_dataset(
    keys: list[bytearray], 
    dataset: bytearray,
):
    """
    This function checks if any of the keys are present in the dataset.
    Return the block data which contains the dataset, label and offset.
    It also specifies the type of the block depending on the number of keys in the block.
    """

    block_data = BlockData()
    block_data.dataset = dataset
    block_data_type = BlockType.VALID

    # Check if any of the keys are present in the window
    found: list[int] = []
    for list_index in range(len(keys)):
        if keys[list_index] in dataset:
            found.append(list_index)
        else:
            found.append(0)

    if len(found) > 0:
        block_data.label = 1

        # Find the offset of the key in the window
        # NOTE: There can be multiple keys in the same window
        all_offset_indexes = []
        element = None
        for element_index in found:
            if element_index != 0:
                element = keys[element_index]
                current_offset = dataset.find(element)
                all_offset_indexes.append(current_offset)

        # If there are multiple keys in the same window 
        # then we will ignore the sample
        if len(all_offset_indexes) > 1:
            print(
                "WARN: Multiple keys detected in same "
                "window. Ignoring sample."
            )
            block_data_type = BlockType.INVALID
            block_data.label = 0
            block_data.offset = 0
            block_data.length = 0
        else:
            block_data_type = BlockType.VALID
            block_data.offset = all_offset_indexes[0]
            block_data.length = len(element)
    else:
        block_data_type = BlockType.EMPTY
        block_data.label = 0
        block_data.offset = 0
        block_data.length = 0
    
    return block_data_type, block_data

def get_data_from_keys_in_heap_dump(
    heap_dump_file_path: str,
    keys: list[bytearray]
):
    """
    This function reads the heap dump file.
    It creates 16 byte blocks and checks if 
    any of the keys are present in the block.
    NOTE: Only one key can be present in a block.

    Returns: The block data.
    """

    block_datas :list[BlockData] = []
    i = 0

    with open(heap_dump_file_path, "rb") as file:
        data = bytearray(file.read())
        # We create 16 byte blocks

        # iterate over key lengths
        while i + WINDOW_SIZE <= len(data):
            window_sum = sum(data[i:i+WINDOW_SIZE])

            if window_sum != 0:
                block_data_type, block_data = get_block_data_from_keys_in_dataset(
                    keys, data[i:i+WINDOW_SIZE]
                )
                if block_data_type is not BlockType.INVALID:
                    block_datas.append(block_data)
            
            i += KEY_SIZE
        
        # Check if there is any data left
        window_sum = sum(data[-WINDOW_SIZE:])
        # take into account the last bytes of the file
        if i < len(data) and window_sum > 0:
            block_data_type, block_data = get_block_data_from_keys_in_dataset(
                keys, data[-WINDOW_SIZE:]
            )
            if block_data_type is not BlockType.INVALID:
                block_datas.append(block_data)

    return block_data

def create_dataset(heap_dump_dir_path, keys_path):
    """
    The aim is to split the raw file into multiple blocks of 128 bytes.
    If the key for the file is present in the block then that page will be labelled True else False.
    This will be an unbalanced dataset.

    :param folder_path: path of the heap dump files
    :param keys_path: path to the folder containing the corresponding keys of dumps
    :return:
    """
    file_paths = os.listdir(heap_dump_dir_path)
    all_block_datas: list[BlockData] = []

    for file_path in file_paths:

        if file_path in PARAMS.file_names:
            print('WARNING: VALIDATION FILE OVERLAPS WITH TRAINING DATASET. \n %s' % file_path)
            continue

        PARAMS.file_names.append(file_path)

        heap_dump_file_path = os.path.join(heap_dump_dir_path, file_path)
        key_path = os.path.join(keys_path, file_path[:-9] + '-key.log')
        keys_in_file = read_keys(key_path)
        
        file_block_datas = get_data_from_keys_in_heap_dump(
            heap_dump_file_path, keys_in_file
        )
        all_block_datas.extend(file_block_datas)

    return all_block_datas


def read_files(paths, key_paths, model=None, window_size=128, key_size=64, root_dir=None, oversample=False):
    """
    Reads a list of files and their corresponding keys
    :param paths: File paths as a list
    :param key_paths: Path of corresponding key files
    :param model: Doc2Vec model for concatenating the model to the block
    :param window_size: Size of the block which the binary file is to sliced
    :param key_size: Length of the largest key in bytes
    :param root_dir: Root of the directory if it is not the default ROOT
    :param oversample: Increase the number of positive samples by shifting the key by 8 bytes
    :return: Matrix of bytes of shape Nx128, labels, offsets
    """
    dataset = []
    labels = []
    offsets = []

    if root_dir is None:
        base_path_length = len(ROOT_DIR_PATH) + 1
    else:
        base_path_length = len(root_dir) + 1

    for path, key_path in zip(paths, key_paths):

        assert (key_path[:-5] in path)
        curr_keys = read_key_files(key_path)
        with open(path, "rb") as fp:
            data = fp.read()
            data = bytearray(data)
            idx = 0

            # Get the heap representation by Doc2Vec
            feature_vector = []
            if model is not None:
                feature_vector = model.infer_vector(list(map(str, data)))
                feature_vector = feature_vector.tolist()

            for curr_key in curr_keys:
                if curr_key not in data:
                    print(str(curr_key) + " not found in data file:" + path)

            enc_alg_info = path[base_path_length:].split(sep="/")[:-1]
            assert (enc_alg_info[0] in TYPES)
            assert (int(enc_alg_info[2]) in LENGTHS)

            # We create 16 byte blocks
            while idx + window_size <= len(data):
                window_sum = sum(data[idx:idx + window_size])
                if window_sum == 0:
                    idx += key_size
                    continue

                if model is None:
                    dataset.append(data[idx:idx + window_size])
                else:
                    temp = list(map(float, data[idx:idx + window_size])) + feature_vector
                    dataset.append(temp)

                found = [l_idx for l_idx in range(len(curr_keys)) if curr_keys[l_idx] in data[idx:idx + window_size]]
                if len(found) > 0:
                    labels.append(1)
                    offset = [data[idx:idx + window_size].find(curr_keys[element]) for element in found]
                    if len(offset) > 1:
                        offsets.append(2)  # Setting it as 2 so that we know that there are multiple keys in there
                    else:
                        offsets.append(offset[0])
                else:
                    labels.append(0)
                    offsets.append(0)

                if len(labels) != len(offsets):
                    print("Hello")
                idx += key_size

        window_sum = sum(data[-window_size:])
        if idx < len(data) and window_sum > 0:

            if model is None:
                dataset.append(data[-window_size:])
            else:
                temp = list(map(float, data[-window_size:])) + feature_vector
                dataset.append(temp)

            found = [l_idx if curr_keys[l_idx] in data[idx:idx + window_size]
                     else 0 for l_idx in range(len(curr_keys))]

            if any(found) is True:
                labels.append(1)
                offset = [data[idx:idx + window_size].find(curr_keys[element]) for element in found if element != 0]
                offsets.append(offset[0])

                assert (len(offsets) == 1)

            else:
                labels.append(0)
                offsets.append(0)

    return dataset, labels, offsets


def read_key_files(path):
    """
    Reads the present keys with a JSON key file
    :param path: path of the key file
    :return: keys in a list
    """
    key_names = ['KEY_A', 'KEY_B', 'KEY_C', 'KEY_D', 'KEY_E', 'KEY_F']
    keys = []
    with open(path, "r") as fp:
        data = json.loads(fp.read())

    for key in key_names:

        key_value = data.get(key, None)

        # If the key is not found or key is empty
        if key_value is None or len(key_value) == 0:
            continue

        keys.append(bytearray.fromhex(key_value))

    return keys


def get_dataset_file_paths(path, deploy=False):
    """
    Gets the file paths of the dataset. 
    If deploy is true, it will return all the files.
    :param path: Path of the dataset
    :param deploy: If true, it will return all the files
    :return: List of file paths
    """

    import glob
    paths = []

    file_paths = []
    key_paths = []

    sub_dir = os.walk(path)
    for potential_dir in sub_dir:
        # check if it is a directory
        if os.path.isdir(potential_dir[0]):
            paths.append(potential_dir[0])

    paths = set(paths)
    for path in paths:
        # print(os.listdir(path))
        files = glob.glob(os.path.join(path, '*.raw'), recursive=False)

        if len(files) == 0:
            continue

        for file in files:
            key_file = file.replace("-heap.raw", ".json")
            if os.path.exists(key_file) and deploy is False:
                file_paths.append(file)
                key_paths.append(key_file)

            elif deploy is True:
                file_paths.append(file)

            else:
                log("Corresponding Key file does not exist for :%s" % file)

    return file_paths, key_paths


def get_metrics(y_true, y_pred, return_cm=False):
    """
    Obtain metrics from the true and predicted labels.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    pr = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    tp = cm[1][1]
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]

    if return_cm is False:
        return acc, pr, recall, f1, tp, tn, fp, fn
    else:
        return acc, pr, recall, f1, cm


def test(clf, file_paths, key_paths, window_size=128, model=None, root_dir=None):
    """

    :param clf: model to be tested
    :param file_paths:
    :param key_paths:
    :param window_size: Size of the block of data to be extracted from heap at a time
    :param model: Doc2Vec model for generating representations of the heap
    :return: truth, predicted values and data frame with metrics on each group of test
    """
    import pandas as pd

    idx = 0
    y_true = []
    y_pred = []

    if root_dir is None:
        base_path_length = len(ROOT_DIR_PATH) + 1
    else:
        base_path_length = len(root_dir) + 1

    df = pd.DataFrame(columns=['Algorithm', 'Version', 'Key Length', 'Total Instances', 'Positive Instances',
                               'Negative Instances', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'True Negatives',
                               'True Positives', 'False Positives', 'False Negatives'])

    while idx < len(key_paths):

        enc_alg_info = file_paths[idx][base_path_length:].split(sep="/")[:-1]

        assert (enc_alg_info[0] in TYPES)
        assert (int(enc_alg_info[2]) in LENGTHS)

        path_idx = file_paths[idx].rfind("/")
        limit = 1
        while idx + limit < len(key_paths) and file_paths[idx][:path_idx] in file_paths[idx + limit]:
            limit += 1

        # print((enc_alg_info[0], enc_alg_info[1], enc_alg_info[2], idx, limit, len(key_paths)))

        x_test, curr_labels, _ = read_files(paths=file_paths[idx:idx + limit], key_paths=key_paths[idx:idx + limit],
                                            model=model, window_size=window_size, root_dir=root_dir)

        x_test = np.array(x_test).astype(int)
        curr_pred = clf.predict(x_test)

        y_true = y_true + curr_labels
        y_pred = y_pred + curr_pred.tolist()

        acc, pr, recall, f1, tp, tn, fp, fn = get_metrics(y_true=curr_labels, y_pred=curr_pred)

        total = tp + tn + fp + fn
        total_neg = tn + fp
        total_pos = tp + fn

        df.loc[len(df.index)] = enc_alg_info[0], enc_alg_info[1], enc_alg_info[2], total, total_pos, total_neg, acc, \
                                pr, recall, f1, tn, tp, fp, fn

        idx += limit

    return y_true, y_pred, df


def print_metrics(y_test, y_pred):
    """
    Print the metrics for the given test and predicted values.
    Metrics are Accuracy, Precision, Recall, F1-Measure and Confusion Matrix.
    """
    acc, pr, recall, f1, cm = get_metrics(y_true=y_test, y_pred=y_pred, return_cm=True)
    log("Accuracy: %f" % acc)
    log("Precision: %f" % pr)
    log("Recall: %f" % recall)
    log("F1-Measure: %f" % f1)
    log("\nConfusion Matrix:\n" + str(cm))


def get_splits(path, val_per=0.15, test_per=0.15, random_state=42):
    """
    Get the data splits for the dataset path.
    """
    from sklearn.model_selection import train_test_split
    import time

    start = timer()
    file_paths, key_paths = get_dataset_file_paths(path)
    end = timer()
    log('Time taken for finding all files: %f' % (end - start))

    start = timer()
    train_files, val_files, train_keys, val_keys, = \
        train_test_split(file_paths, key_paths, test_size=test_per, random_state=random_state)
    end = timer()
    log('Time taken for splitting: %f' % (end - start))

    start = timer()
    train_files, test_files, train_keys, test_keys, = \
        train_test_split(train_files, train_keys, test_size=val_per, random_state=random_state)
    end = timer()
    log('Time taken for secondary splitting: %f' % (end - start))

    return train_files, train_keys, test_files, test_keys, val_files, val_keys


def train_classifier(
    dataset, 
    labels, 
    test_paths=[], 
    test_keys=[], 
    retrain_rf=False, 
    retrain_resampled=False,
    retrain_final=False
):
    """
    Trains the classifier and returns the wrapped classifier.
    Can also be used to retrain the classifier.
    Can also reload the classifier from disk.

    :param dataset: Paths of files to be trained
    :param labels:  The corresponding keys
    :param test_paths: Paths of heaps to be tested
    :param test_keys: Paths of corresponding key files
    :param retrain_rf: Whether to retrain the random forest or load it from disk
    :param retrain_resampled: Whether to retrain the resampled data classifier or load it from disk
    :param retrain_final: Whether to retrain the final classifier or not
    :return: Wrapped classifier
    """

    import time
    import pickle

    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier

    path = os.path.join(MODEL_DIR_PATH, 'rf.pkl')

    # retrain classifier if needed (asked or not present)
    if retrain_rf is True or not os.path.exists(path):
        start = time.time()
        rf = RandomForestClassifier(n_estimators=5)
        rf.fit(X=dataset, y=labels)
        end = time.time()
        log('Time taken for training the classifier: %f' % (end - start))

        with open(path, 'wb') as fp:
            pickle.dump(rf, fp)

    else:
        with open(path, 'rb') as fp:
            rf = pickle.load(fp)

    path = os.path.join(MODEL_DIR_PATH, 'resampled_clf.pkl')
    if retrain_resampled is True or not os.path.exists(path):
        # Use SMOTE oversampling
        start = time.time()
        sm = SMOTE()
        x_train, y_train = sm.fit_resample(dataset, labels)
        end = time.time()
        log('Time taken for resampling: %f' % (end - start))

        #  clf.partial_fit(X=np.array(dataset).astype(int), y=labels, classes=classes)
        start = time.time()
        resampled_clf = RandomForestClassifier(n_estimators=5)
        resampled_clf.fit(X=np.array(x_train), y=y_train)
        end = time.time()
        log('Time taken for training the classifier on resampled data: %f' % (end - start))

        # Clear memory of x_train and y_train
        x_train = None
        y_train = None

        with open(path, 'wb') as fp:
            pickle.dump(resampled_clf, fp)

    else:
        with open(path, 'rb') as fp:
            resampled_clf = pickle.load(fp)

    path = os.path.join(MODEL_DIR_PATH, 'secondary_clf.pkl')
    if retrain_final is True or not os.path.exists(path=path):

        # Predict probabilities to generate modified training vectors
        start = time.time()
        resampled_predicted = resampled_clf.predict_proba(dataset)
        end = time.time()
        log('Time taken for predicting on the resampled classifier: %f' % (end - start))

        start = time.time()
        non_resampled_predicted = rf.predict_proba(dataset)
        end = time.time()
        log('Time taken for predicting on the random forest classifier: %f' % (end - start))

        # Stack the probabilities together
        combined_dataset = np.hstack((resampled_predicted, non_resampled_predicted))

        # Train Random Forest classifier on the modified input data
        start = time.time()
        final_clf = RandomForestClassifier(n_estimators=3)
        final_clf.fit(X=np.array(combined_dataset), y=labels)
        end = time.time()
        log('Time taken for training the final classifier: %f' % (end - start))

        # Save the model to the disk
        path = os.path.join(MODEL_DIR_PATH, 'secondary_clf.pkl')
        with open(path, 'wb') as fp:
            pickle.dump(final_clf, fp)

    else:
        with open(path, 'rb') as fp:
            final_clf = pickle.load(fp)

    clf = WrappedClassifier(resampled_classifier=resampled_clf, classifier=rf, final_stage_classifier=final_clf)

    if len(test_paths) == 0:
        return clf

    print('Testing Dataset')
    start = time.time()
    y_test, y_pred, df = test(clf=clf, file_paths=test_paths, key_paths=test_keys)
    end = time.time()
    log('Time taken for reading and testing: %f' % (end - start))

    path = os.path.join(RESULTS_PATH, "test_results_" + str(datetime.now()) + ".csv")
    df.to_csv(path)

    start = time.time()
    log('METRICS OF TEST SET')
    print_metrics(y_test=y_test, y_pred=y_pred)
    end = time.time()
    log('Time taken for computing metrics: %f' % (end - start))

    return clf


def load_models(load_high_recall_only=False):
    """
    Load the models from the disk
    :param load_high_recall_only: Whether to load only the high recall classifier or not.
    """

    import pickle

    def __load_model_data(model_data_filename: str):
        """
        Load the model data from the disk.
        :return: Model data
        """
        start = timer()
        path = os.path.join(MODEL_DIR_PATH, model_data_filename)

        # check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError('Model file %s not found' % path)

        with open(path, 'rb') as file:
            model_data = pickle.load(file)
        end = timer()
        return model_data, end - start

    
    # Load the model data
    resampled_clf, load_time = __load_model_data('resampled_clf_entropy.pkl')
    log('Time taken for loading high recall classifier: %f' % load_time)

    assert isinstance(resampled_clf, RandomForestClassifier), 'resampled_clf is not a RandomForestClassifier'

    if load_high_recall_only is True:
        return resampled_clf
    
    rf, load_time = __load_model_data('rf_entropy.pkl')
    log('Time taken for loading high precision classifier: %f' % load_time)

    secondary_clf, load_time = __load_model_data('secondary_clf_entropy.pkl')

    clf = WrappedClassifier(
        resampled_classifier=resampled_clf, 
        classifier=rf, 
        final_stage_classifier=secondary_clf
    )
    return clf


def check_path_exists(path: str):
    """
    Check if the path exists. Return True if it exists, False otherwise.
    """
    if not os.path.exists(path):
        print('WARNING: Path does not exist: %s' % path)
        return False
    return True

