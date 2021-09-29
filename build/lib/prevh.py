# Libraries Imports
import pandas as pd
import numpy as np
import math as mt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random as rd


# DATA STRUCTS
class DataSetInfo:
    def __init__(self, rawdata, axisheader, posibleresults, datacount, resultsheader, relevationheader):
        self.rawdata = rawdata  # pandas.DataFrame (object)
        self.axisheader = axisheader  # list
        self.posibleresults = posibleresults  # list
        self.datacount = datacount  # int
        self.resultsheader = resultsheader  # string
        self.relevationheader = relevationheader  # string

    # Plot the rawdata from DataSetInfo
    def plot(self, **kwargs):
        # kwargs
        figx = kwargs.get('figx', 10)
        figy = kwargs.get('figy', 10)

        def genrdhexcolor(num):
            colors = []
            for i in range(num):
                sc = "#%06x" % rd.randint(0, 0xFFFFFF)
                colors += [sc]
            return colors

        title = kwargs.get('title', "Data Set without normalization")
        colors = genrdhexcolor(len(self.posibleresults))

        if len(self.axisheader) == 2 or len(self.axisheader) == 3:
            fig = plt.figure(figsize=(figx, figy))
            fig.suptitle(title, fontsize=16)
            if len(self.axisheader) == 3:
                ax = fig.add_subplot(111, projection='3d')
                for i in range(len(self.rawdata)):
                    for c, r in enumerate(self.posibleresults):
                        if r == self.rawdata.iat[i, 3]:
                            ax.scatter(self.rawdata.iat[i, 0], self.rawdata.iat[i, 1], self.rawdata.iat[i, 2], zdir='z', c=colors[c], s=15)
                            break
            else:
                ax = fig.add_subplot(111)
                for i in range(len(self.rawdata)):
                    for c, r in enumerate(self.posibleresults):
                        if r == self.rawdata.iat[i, 2]:
                            ax.scatter(self.rawdata.iat[i, 0], self.rawdata.iat[i, 1], c=colors[c], s=15)
                            break
            plt.legend(self.posibleresults, labelcolor=colors, markerscale=0, handletextpad=-1.5, shadow=True)
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.show()
        else:
            raise TypeError("Impossible to plot with less then 2 or more then 3 dimensions.")

    # Predict with raw data from DataSetInfo
    def predict(self, inputlist, **kwargs):

        def Euclidean_Dist(df1, df2, cols=self.axisheader):
            return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)

        predict_data = self.rawdata.copy()
        list_of_predict_results = []
        list_of_predict_inputs = []
        nneighbors = kwargs.get('nNeighbors', self.datacount)
        predict_mode = kwargs.get('mode', "parallel")
        # Input verifications
        if isinstance(nneighbors, int):
            nneighbors = np.full(shape=len(inputlist), fill_value=nneighbors).tolist()
        if not isinstance(inputlist, list):
            raise TypeError("The prediction input parameter must be a list of lists.")
        if not isinstance(nneighbors, list):
            raise TypeError("The nNeighbors parameter must be a list.")
        if len(nneighbors) != len(inputlist):
            raise TypeError("The input and nNeighbors parameters must have the same length.")
        # Starts Prediction
        for e, i in enumerate(inputlist):
            if not isinstance(i, list):
                raise TypeError("The prediction input must be a list of lists.")
            if len(i) != len(self.axisheader):
                raise TypeError("At least one of the prediction inputs does not match with axis length.")
            if nneighbors[e] > self.datacount or nneighbors[e] < 1 or not isinstance(nneighbors[e], int):
                raise TypeError("The nNeighbors parameter must be an integer and be in between 1 and " + str(self.datacount) + " (inclusive).")
            if predict_mode == "parallel": predict_data = self.rawdata.copy()
            predict_results = pd.DataFrame(columns=[self.resultsheader, "probability", "relevance"])
            predict_data.loc[len(predict_data)] = i + [None, "0"]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaleddata = scaler.fit_transform(predict_data[self.axisheader])
            predict_data = pd.DataFrame(scaleddata, columns=self.axisheader)
            predict_data.insert(len(self.axisheader), self.resultsheader, self.rawdata[self.resultsheader], True)
            predict_data.insert(len(self.axisheader) + 1, self.relevationheader, self.rawdata[self.relevationheader], True)
            predict_input = predict_data.loc[pd.isna(predict_data[self.resultsheader])]
            predict_input = (predict_input.drop([self.resultsheader, self.relevationheader], axis=1)).reset_index(drop=True)
            predict_data = predict_data.loc[pd.notna(predict_data[self.resultsheader])]
            predict_data["distance"] = Euclidean_Dist(predict_data, predict_input)
            predict_data = predict_data.sort_values("distance").reset_index(drop=True)
            predict_data = predict_data[predict_data.index < nneighbors[e]]
            predict_data[self.relevationheader] = predict_data[self.relevationheader].apply(lambda x: x * mt.sqrt(len(self.axisheader)))
            predict_data["powered distance"] = predict_data.apply(lambda x: (x["distance"] * x[self.relevationheader]), axis=1)
            powered_dist_sum = predict_data["powered distance"].sum()
            for p, r in enumerate(self.posibleresults):
                subset_df = predict_data[predict_data[self.resultsheader] == r]
                column_sum = subset_df["powered distance"].sum()
                if column_sum != 0:
                    if column_sum != predict_data["powered distance"].sum():
                        predict_results.loc[p] = [r, 1 - (column_sum/predict_data["powered distance"].sum()), mt.sqrt(len(self.axisheader)) * (1 - (column_sum/predict_data["powered distance"].sum()))]
                    else:
                        predict_results.loc[p] = [r, 1.0, mt.sqrt(len(self.axisheader))]
            list_of_predict_results += [[predict_results, e]]
        return PredictInfo(predict_data, list_of_predict_results)


class PredictInfo:
    # Self Creation Method
    def __init__(self, predict_data, predict_results):
        self.predict_data = predict_data  # pandas.DataFrame (object)
        self.predict_results = predict_results  # pandas.DataFrame (object)


# FUNCTIONS
def datasetfromDF(rawdata, **kwargs):  # (pd.DataFrame)
    # args
    header = kwargs.get('header', "FromFile")
    # header configuration
    if type(rawdata) is pd.core.frame.DataFrame:
        if header != "FromFile":
            if isinstance(header, list):
                try:
                    header += ["Result", "Relevance"]
                    rawdata.columns = header
                except TypeError("Please verify if the header values match with the number of columns."):
                    return None
            else:
                col = []
                for it in range(len(rawdata.columns)):
                    col += ["a" + str(it + 1)]
                col[len(rawdata.columns) - 2], col[len(rawdata.columns) - 1] = "Consequence", "Relevance"
                rawdata.columns = col
        colnum = len(rawdata.columns)
        axisheader = rawdata.columns[:colnum - 2]
        posibleresults = rawdata.iloc[:, colnum - 2].unique()
        datacount = rawdata.shape[0]
        resultsheader = rawdata.columns[colnum - 2]
        relevationheader = rawdata.columns[colnum - 1]
        if not rawdata[relevationheader].between(0, 1, inclusive="both").all():
            raise TypeError("At least one of the information relevance is not between 0 and 1.")
    else:
        raise TypeError("Please check if the input is an DataFrame (Pandas) object.")
    return DataSetInfo(rawdata, axisheader, posibleresults, datacount, resultsheader, relevationheader)


def datasetfromCSV(path, div, **kwargs):  # (string, string)
    # global return info
    rawdata = None
    # args
    header = kwargs.get('header', "FromFile")
    # header configuration
    try:
        rawdata = pd.read_csv(path, sep=div)
        if isinstance(header, list):
            try:
                header += ["Result", "Relevance"]
                rawdata.columns = header
            except TypeError("Please verify if the header values match with the number of columns."):
                return None
        elif header != "FromFile":
            col = []
            for it in range(len(rawdata.columns)):
                col += ["a" + str(it + 1)]
            col[len(rawdata.columns) - 2], col[len(rawdata.columns) - 1] = "Result", "Relevance"
            rawdata.columns = col
    except TypeError("Please check the path argument, only csv file are available."):
        return None
    # global return info
    colnum = len(rawdata.columns)
    axisheader = rawdata.columns[:colnum - 2]
    posibleresults = rawdata.iloc[:, colnum - 2].unique()
    datacount = rawdata.shape[0]
    resultsheader = rawdata.columns[colnum - 2]
    relevationheader = rawdata.columns[colnum - 1]
    if not rawdata[relevationheader].between(0, 1, inclusive="both").all():
        raise TypeError("At least one of the information relevance is not between 0 and 1.")
    return DataSetInfo(rawdata, axisheader, posibleresults, datacount, resultsheader, relevationheader)
