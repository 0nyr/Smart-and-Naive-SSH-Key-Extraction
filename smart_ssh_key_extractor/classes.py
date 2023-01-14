from enum import Enum
from constants import *

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass


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


class BlockType(Enum):
    VALID = 1 # One and only one key
    EMPTY = 0 # No key
    INVALID = 2 # More than one key


@dataclass
class BlockData:
    dataset: bytearray = None # dataset of the block, contains bytes
    label: int = None # if VALID then 1 else EMPTY 0
    offset: int = None # offset of the key in the dataset byte array
    length: int = None # length of the key

