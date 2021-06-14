import pandas as pd  # For Data Maintenance
import numpy as np  # For Data Maintenance
import matplotlib as mt  # For Data Maintenance


class DataFrameInfo:
    def __init__(self, data, axis, results, numdata, resheader, revheader):
        self.data = data  # pandas.DataFrame (object)
        self.axis = axis  # prevh.AxisInfo (object)
        self.results = results  # 1dArray
        self.numdata = numdata  # int
        self.resheader = resheader  # string
        self.revheader = revheader  # string
