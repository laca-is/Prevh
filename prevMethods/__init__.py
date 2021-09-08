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

        title = kwargs.get('title', "Data Frame")
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
                ax.set_xlim(0, 1), ax.set_ylim(0, 1)
                ax.set_xticks(ticks), ax.set_yticks(ticks)
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

        data = None
        preinput = None
        predictresults = None
        nneighbors = kwargs.get('nNeighbors', self.datacount)
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
            predictdata = self.rawdata.copy()
            predictdata.loc[len(predictdata)] = i + [None, "0"]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaleddata = scaler.fit_transform(predictdata[self.axisheader])
            predictdata = pd.DataFrame(scaleddata, columns=self.axisheader)
            predictdata.insert(len(self.axisheader), self.resultsheader, self.rawdata[self.resultsheader], True)
            predictdata.insert(len(self.axisheader) + 1, self.relevationheader, self.rawdata[self.relevationheader], True)
            preinput = predictdata.loc[pd.isna(predictdata[self.resultsheader])]
            preinput = (preinput.drop([self.resultsheader, self.relevationheader], axis=1)).reset_index(drop=True)
            predictdata = predictdata.loc[pd.notna(predictdata[self.resultsheader])]
            data = predictdata.copy()
            predictdata[self.relevationheader] = predictdata[self.relevationheader].apply(lambda x: x * mt.sqrt(len(self.axisheader)))
            predictdata["distance"] = Euclidean_Dist(predictdata, preinput)
            predictdata = predictdata.sort_values("distance").reset_index(drop=True)
            predictdata = predictdata[predictdata.index < nneighbors[e]]
            print(predictdata)
        return PredictInfo(data, preinput, predictresults)


class PredictInfo:
    # Self Creation Method
    def __init__(self, useddata, usedinput, predictresults):
        self.useddata = useddata  # pandas.DataFrame (object)
        self.usedinput = usedinput  # pandas.DataFrame (object)
        self.predictresults = predictresults  # pandas.DataFrame (object)


# FUNCTIONS
def datasetfromDF(rawdata, **kwargs):  # (pd.DataFrame)
    # args
    header = kwargs.get('header', "FromFile")
    # header configuration
    if type(rawdata) is pd.core.frame.DataFrame:
        if header != "FromFile":
            if isinstance(header, list):
                try:
                    header += ["Consequence", "Relevance"]
                    rawdata.columns = header
                except TypeError("Please verify if the header values match with the number of columns."):
                    return None
            else:
                rawdata = pd.read_csv(path, div, header=None)
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
        if lambda x: rawdata[relevationheader].between(-0.1, 1.1) is False:
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
        if header == "FromFile":
            rawdata = pd.read_csv(path, div)
        elif isinstance(header, list):
            try:
                rawdata = pd.read_csv(path, div, header=None)
                header += ["Result", "Relevance"]
                rawdata.columns = header
            except TypeError("Please verify if the header values match with the number of columns."):
                return None
        else:
            rawdata = pd.read_csv(path, div, header=None)
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
    if not rawdata[relevationheader].between(0, 1, inclusive=True).all():
        raise TypeError("At least one of the information relevance is not between 0 and 1.")
    return DataSetInfo(rawdata, axisheader, posibleresults, datacount, resultsheader, relevationheader)


prevdata = datasetfromCSV("data1.csv", ",")
prevresults = prevdata.predict([[0.1, 0.1, 0.1], [0.2, 0.4, 0.2]], nNeighbors=4)
