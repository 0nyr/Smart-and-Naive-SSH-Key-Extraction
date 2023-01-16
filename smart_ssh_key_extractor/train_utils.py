"""
Utility functions for training and data loading
(dataset creation, model creation, training, etc.)
"""
from constants import *
from classes import *

import os
import json
from log_custom import * 



def read_keys_from_json(json_filepath):
    """
    Reads the present keys with a JSON key file
    :param json_filepath: path of the key file
    :return: keys in a list
    """
    key_names = ['KEY_A', 'KEY_B', 'KEY_C', 'KEY_D', 'KEY_E', 'KEY_F']
    keys: list[bytearray] = []
    with open(json_filepath, "r") as fp:
        data = json.loads(fp.read())

    for key in key_names:

        key_value = data.get(key, None)

        # If the key is not found or key is empty
        if key_value is None or len(key_value) == 0:
            continue

        keys.append(bytearray.fromhex(key_value))

    return keys


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

    if len(found) > 0:
        block_data.label = BlockType.VALID.value

        # Find the offset of the key in the window
        # NOTE: There can be multiple keys in the same window
        all_offset_indexes = []
        current_key = None
        for element_index in found:
            current_key = keys[element_index]
            current_offset = dataset.find(current_key)
            all_offset_indexes.append(current_offset)

        # If there are multiple keys in the same window 
        # then we will ignore the sample
        if len(all_offset_indexes) > 1:
            block_data_type = BlockType.INVALID
            block_data.label = BlockType.INVALID.value
            block_data.offset = 0
            block_data.length = 0
        else:
            block_data_type = BlockType.VALID
            block_data.offset = all_offset_indexes[0]
            block_data.length = len(current_key)
    else:
        block_data_type = BlockType.EMPTY
        block_data.label = BlockType.EMPTY.value
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

    :return: all_block_data_in_file, list of block data in the heap dump file
    :return: nb_invalid_blocks, number of invalid blocks in the heap dump file
    """
    nb_invalid_blocks = 0
    all_block_data_in_file: list[BlockData] = []
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
                    all_block_data_in_file.append(block_data)
                else:
                    nb_invalid_blocks += 1
            
            i += KEY_SIZE
        
        # Check if there is any data left
        window_sum = sum(data[-WINDOW_SIZE:])
        # take into account the last bytes of the file
        if i < len(data) and window_sum > 0:
            block_data_type, block_data = get_block_data_from_keys_in_dataset(
                keys, data[-WINDOW_SIZE:]
            )
            if block_data_type is not BlockType.INVALID:
                all_block_data_in_file.append(block_data)
            else:
                nb_invalid_blocks += 1

    return all_block_data_in_file, nb_invalid_blocks


def get_dataset_file_paths(path, deploy=False):
    """
    Gets the file paths of the dataset. 
    If deploy is false, it will also return all the key files.
    :param path: Path of the dataset
    :param deploy: If false, it will also return the key files.
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
                LOGGER.log("Corresponding Key file does not exist for :%s" % file)

    return file_paths, key_paths


def create_dataset(heap_dump_dir_path):
    """
    The aim is to split the raw file into multiple blocks of 128 bytes.
    If the key for the file is present in the block then that page will be labelled True else False.
    This will be an imbalanced dataset.

    :param heap_dump_dir_path: dir path of the heap dump (raw) files
    :return: A big list of block data from all the files. Few of them will be labelled True and the rest False.
    :return: Total number of invalid blocks. These blocks have been ignored.
    """
    # get all the .raw files from subdirectories
    file_paths, _ = get_dataset_file_paths(heap_dump_dir_path)
    file_paths = set(file_paths) # remove duplicates
    print("Number of .raw files to load: %d" % len(file_paths))

    # generate the list of block data
    all_block_datas: list[BlockData] = []
    nb_all_invalid_blocks = 0

    for file_path in file_paths:

        if file_path in LOGGER.file_names:
            print('WARNING: VALIDATION FILE OVERLAPS WITH TRAINING DATASET. \n %s' % file_path)
            continue
    
        assert(file_path.endswith("-heap.raw"))

        LOGGER.file_names.append(file_path)

        json_key_file_path = file_path.replace("-heap.raw", ".json")
        keys_in_file = read_keys_from_json(json_key_file_path)
        
        file_block_datas, nb_invalid_blocks = get_data_from_keys_in_heap_dump(
            file_path, keys_in_file
        )
        all_block_datas.extend(file_block_datas)
        nb_all_invalid_blocks += nb_invalid_blocks

    return all_block_datas, nb_all_invalid_blocks