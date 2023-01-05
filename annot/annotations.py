"""
Involves operations related to fetching and reading annotations for MPII Dataset
"""

import json


class Data:
    def __init__(self):
        pass

    def get_train_data(self):
        TRAIN_JSON_DIR = "C:\\Users\\kocam\\Projects\\Datascience\\Projects\\ml_pose_estimation\\annot\\train.json"

        file = open(TRAIN_JSON_DIR)
        return json.load(file)

    def get_validation_data(self):
        VALIDATION_JSON_DIR = "C:\\Users\\kocam\\Projects\\Datascience\\Projects\\ml_pose_estimation\\annot\\trainval.json"

        file = open(VALIDATION_JSON_DIR)
        return json.load(file)

    def get_test_data(self):
        TEST_JSON_DIR = "C:\\Users\\kocam\\Projects\\Datascience\\Projects\\ml_pose_estimation\\annot\\test.json"

        file = open(TEST_JSON_DIR)
        return json.load(file)

    def get_valid(self):
        VALID_JSON_DIR = "C:\\Users\\kocam\\Projects\\Datascience\\Projects\\ml_pose_estimation\\annot\\valid.json"

        file = open(VALID_JSON_DIR)
        return json.load(file)
