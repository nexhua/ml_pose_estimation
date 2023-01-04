"""
Config file for the entire Machine Learning Project
"""


class Configuration:
    def __init__(self):
        self.IMAGES_BASE_DIR = "C:\\Users\\kocam\\Projects\\Datascience\\Projects\\ml_pose_estimation\\images"
        self.ORIGINAL_DATASET_BASE_DIR = "C:\\Users\\kocam\\Projects\\Datascience\\Projects\\ml_pose_estimation\\images\\all\\images"
        self.KEYPOINT_COUNT = 16
        self.DIM = self.KEYPOINT_COUNT * 2 + 2
        self.INPUT_DIMS = (224, 224, 3)
