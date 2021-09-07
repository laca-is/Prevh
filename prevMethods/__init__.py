# For Math and Information Uses
import pandas as pd  # For Data Maintenance
import numpy as np  # For Data Maintenance
import math as mt  # For math problems facilities
from sklearn.preprocessing import MinMaxScaler  # For Data previous normalization
# For Plot and Visualization Uses
import matplotlib.pyplot as plt  # For Plot information
from mpl_toolkits.mplot3d import axes3d  # For Plot information
import random as rd  # To generate random colors for Plot


# DATA TYPES
# The axisData class for the DataFrameInfo
class AxisInfo:
    def __init__(self, numaxis, header):
        self.numaxis = numaxis  # int
        self.header = header  # 1dArray


# The axisDispersion class ready for plot
class AxisDisperInfo:
    def __init__(self, axisdispinfo):
        self.axisdispinfo = axisdispinfo

    def plot(self, **kwargs):
        title = kwargs.get('title', "Axis Dispersion")
        orientation = kwargs.get('orientation', "horizontal")
        figx = kwargs.get('figx', 10)
        figy = kwargs.get('figy', 10)

        def horizontalorientation(num, div):
            ploty = round(num / div)
            plotx = num / round(ploty)
            plotx = round(plotx + 0.5)
            return plotx, ploty

        def verticalorientation(num, div):
            plotx = round(num / div)
            ploty = num / round(plotx)
            ploty = round(ploty + 0.5)
            return plotx, ploty

        ticksy = np.arange(0, 1.1, 0.1)
        pltnum = len(self.axisdispinfo)
        fig = plt.figure("Axis Dispersion Plot")
        fig.suptitle(title, fontsize=16)
        if orientation == "horizontal":
            if pltnum % 2 == 0:
                plotx, ploty = horizontalorientation(pltnum, 2)
            else:
                plotx, ploty = horizontalorientation(pltnum, 3)
        else:
            if pltnum % 2 == 0:
                plotx, ploty = verticalorientation(pltnum, 2)
            else:
                plotx, ploty = verticalorientation(pltnum, 3)
        print(self.axisdispinfo)
        for c, i in enumerate(self.axisdispinfo):
            ax = fig.add_subplot(plotx, ploty, c+1)
            ax.plot(i[1])
            plt.xticks(list(i[1].index))
            plt.yticks(ticksy)
            ax.title.set_text(i[0])
            plt.legend(i[1].columns)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()


# The main data information class for the prevh algorithm
class DataFrameInfo:
    # Self Creation Method
    def __init__(self, data, rawdata, axis, consequencieslist, datacount, consequencieheader, relevationheader, normalizedcolumns):
        self.data = data  # pandas.DataFrame (object)
        self.rawdata = rawdata  # pandas.DataFrame (object)
        self.axis = axis  # prevh.AxisInfo (object)
        self.consequencieslist = consequencieslist  # 1dArray
        self.datacount = datacount  # int
        self.consequencieheader = consequencieheader  # string
        self.relevationheader = relevationheader  # string
        self.normalizedcolumns = normalizedcolumns  # 1dArray

    # Plot the DataFrameInfo information if it´s possible
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
        dimensions = self.axis.numaxis
        colors = genrdhexcolor(len(self.consequencieslist))
        ticks = [0, 0.25, 0.5, 0.75, 1]
        if dimensions == 2 or dimensions == 3:
            fig = plt.figure(figsize=(figx, figy))
            fig.suptitle(title, fontsize=16)
            if dimensions == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_zlim(0, 1)
                ax.set_xticks(ticks), ax.set_yticks(ticks), ax.set_zticks(ticks)
                for i in range(len(self.data)):
                    for c, r in enumerate(self.consequencieslist):
                        if r == self.data.iat[i, 3]:
                            ax.scatter(self.data.iat[i, 0], self.data.iat[i, 1], self.data.iat[i, 2], zdir='z', c=colors[c], s=15)
                            break
            else:
                ax = fig.add_subplot(111)
                ax.set_xlim(0, 1), ax.set_ylim(0, 1)
                ax.set_xticks(ticks), ax.set_yticks(ticks)
                for i in range(len(self.data)):
                    for c, r in enumerate(self.consequencieslist):
                        if r == self.data.iat[i, 2]:
                            ax.scatter(self.data.iat[i, 0], self.data.iat[i, 1], c=colors[c], s=15)
                            break
            plt.legend(self.consequencieslist, labelcolor=colors, markerscale=0, handletextpad=-1.5, shadow=True)
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.show()
        else:
            raise TypeError("Impossible to plot with less then 2 or more then 3 dimensions.")

    # Make the axisDispersion of the current DataFrameInfo
    def axisdispersion(self):
        if type(self) is DataFrameInfo:
            axisdisp = []
            lines = np.arange(self.datacount)
            for a in self.axis.header:
                dic = {}
                for r in self.consequencieslist:
                    u = 0
                    if r not in dic:
                        dic[r] = []
                    for i in lines:
                        if r == self.data.at[i, self.consequencieheader]:
                            if i > 0:
                                if dic[r][i - 1] == 0:
                                    for j in range(i):
                                        dic[r][j] = None
                            dic[r].append(self.data.at[i, a])
                            u = self.data.at[i, a]
                        else:
                            dic[r].append(u)
                axisdisp += [[a] + [pd.DataFrame(dic, index=lines)]]
            return AxisDisperInfo(axisdisp)
        else:
            raise TypeError("Wrong type parameter, need to be a DataFrameInfo object.")

    # Apply the prognosis method in the current DataFrameInfo
    def apprognosismethod(self, inputpath, div, **kwargs):
        # Kwargs
        nneighbours = kwargs.get('nNeighbours', self.data.shape[0])
        if nneighbours > self.data.shape[0]:
            raise Exception("Please use an value lower than the dataset length (" + str(self.data.shape[0]) + ")")

        # Useful functions
        def euclideandist(n1, n2):
            return mt.pow((n1 - n2), 2)

        results, resulttable = [], []
        try:
            inputdata = pd.read_csv(inputpath, div, header=None)
        except TypeError("Please check the input csv file."):
            return None
        dimensions = self.axis.numaxis
        pointrelid = dimensions + 1
        for inputdat in range(len(inputdata)):
            # see the nearest
            datlist, preres = [], []
            emptycell = False
            for x in inputdata.loc[inputdat]:
                emptycell = pd.isna(x)
                if emptycell: break
            if emptycell:
                print("Missing values at the " + str(inputdat) + " line of your input file.")
                continue
            for dat in range(len(self.data)):
                for x in self.data.loc[dat]:
                    emptycell = pd.isna(x)
                    if emptycell:
                        break
                if emptycell:
                    print("Missing values at the " + str(dat) + " line of your raw file.")
                    continue
                axisdisplay, inpaxisdisplay = [], []
                euclidist, realrelev, res = 0, 0, ""
                for ax in range(self.axis.numaxis):
                    axisdisplay += [self.data.iat[dat, ax]]
                    inpaxisdisplay += [inputdata.iat[inputdat, ax]]
                    euclidist += euclideandist(inputdata.iat[inputdat, ax], self.data.iat[dat, ax])
                res = self.data.iat[dat, int(self.axis.numaxis)]
                euclidist = mt.sqrt(euclidist)
                realrelev = (1 - (euclidist / mt.sqrt(dimensions))) * self.data.iat[dat, pointrelid]
                datlist += [[inputdat] + [inpaxisdisplay] + [axisdisplay] + [res] + [euclidist] + [realrelev]]
            datlist.sort(key=lambda x: x[5])
            for r in self.consequencieslist:
                meancont = 0
                relevtotal = 0
                for d in datlist:
                    if d[3] == r:
                        relevtotal += d[5]
                        meancont += 1
                resulttable += [[inputdat] + [r] + [relevtotal/meancont]]
            resulttable.sort(key=lambda x: x[2])
            results += datlist
        return pd.DataFrame(resulttable, columns=["inputID", "Consequence", "Probability"]).sort_values("inputID")


# NATIVE FUNCTIONS
# The DataFrame Creation Function
def createdataframe(path, div, **kwargs):  # (string, string)
    # global return info
    data = None
    # args
    header = kwargs.get('header', "FromFile")
    # header configuration
    try:
        if header == "FromFile":
            data = pd.read_csv(path, div)
        elif isinstance(header, list):
            try:
                data = pd.read_csv(path, div, header=None)
                header += ["Consequence", "Relevance"]
                data.columns = header
            except TypeError("Please verify if the header values match with the number of columns."):
                return None
        else:
            data = pd.read_csv(path, div, header=None)
            col = []
            for it in range(len(data.columns)):
                col += ["a" + str(it + 1)]
            col[len(data.columns) - 2], col[len(data.columns) - 1] = "Consequence", "Relevance"
            data.columns = col
    except TypeError("Please check the path argument, only csv file are available."):
        return None
    # global return info
    colnum = len(data.columns)
    rawdata = data.copy()
    axis = AxisInfo(colnum - 2, data.columns[:colnum - 2])
    consequencieslist = data.iloc[:, colnum - 2].unique()
    datacount = data.shape[0]
    consequencieheader = data.columns[colnum - 2]
    relevationheader = data.columns[colnum - 1]
    # checking the need for normalization
    normalizedcolumns = []
    for s in data.columns[:colnum - 2]:
        if data[data[s] > 1].shape[0] != 0:
            normalizedcolumns += [s]
        if data[data[s] < 0].shape[0] != 0:
            normalizedcolumns += [s]
    normalizedcolumns = np.unique(normalizedcolumns).tolist()
    if normalizedcolumns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaleddata = scaler.fit_transform(data[normalizedcolumns])
        datanormalized = pd.DataFrame(scaleddata, columns=normalizedcolumns)
        for c in normalizedcolumns:
            data = data.assign(**{c: datanormalized[c]})
    return DataFrameInfo(data, rawdata, axis, consequencieslist, datacount, consequencieheader, relevationheader, normalizedcolumns)


infotest2 = createdataframe("data1.csv", ",")
print(infotest2.normalizedcolumns)
print(infotest2.data)
print(infotest2.rawdata)
print(infotest2.apprognosismethod("input1.csv", ",", nNeighbours=6))
