import numpy as np

class HeartDataSet:
    def __init__(self, attributeName, value, x=None, y=np.ndarray, exampleName=np.ndarray, splits=[], result = None) -> None:
        self.attributeName = attributeName
        self.value = value
        self.x = x
        self.y = y
        self.exampleName = exampleName
        self.splits = splits
        self.result = result