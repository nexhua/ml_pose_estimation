"""
Involves operations for the preprocessing operations.
Generates y_train list(Sorted, resized, normalized)
"""

import os
import numpy as np
from annot import annotations
import configurations

class Label:
    def __init__(self):
        self.data_loader = annotations.Data()
        self.config = configurations.Configuration()

    def get_y_train(self):
        sorted_fnames = sorted(os.listdir(
            os.path.join(self.config.IMAGES_BASE_DIR, "train")))

        train_labels = self.data_loader.get_train_data()

        y_train = np.ndarray(
            shape=(len(sorted_fnames), self.config.DIM), dtype=int)

        for i, name in enumerate(sorted_fnames):
            found = next((x for x in train_labels if x["image"] == name), None)
            if found is not None:
                y_train[i] = np.concatenate(
                    [np.array(found["joints"]).ravel(), np.array(found["center"])])
        return y_train
