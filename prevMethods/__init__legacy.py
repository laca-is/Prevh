import pandas as pd
import numpy as np
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math as mt
import random as rd


class AxisInfo:
    def __init__(self, numaxis, header):
        self.numaxis = numaxis  # int
        self.header = header  # 1dArray


class AxisDisperInfo:
    def __init__(self, axisdispinfo):
        self.axisdispinfo = axisdispinfo

    def plot(self, figx, figy, **kwargs):
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

        title = kwargs.get('title', "Axis Dispersion")
        orientation = kwargs.get('orientation', "horizontal")
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
        plt.show()


class AllPointsPrognosisMethodInfo:
    def __init__(self, allpointsinfo):
        self.allpointsinfo = allpointsinfo


class DataFrameInfo:
    def __init__(self, data, axis, results, numdata, resheader, revheader):
        self.data = data  # pandas.DataFrame (object)
        self.axis = axis  # prevh.AxisInfo (object)
        self.results = results  # 1dArray
        self.numdata = numdata  # int
        self.resheader = resheader  # string
        self.revheader = revheader  # string

    def plot(self, figx, figy, **kwargs):
        def genrdhexcolor(num):
            colors = []
            for i in range(num):
                sc = "#%06x" % rd.randint(0, 0xFFFFFF)
                colors += [sc]
            return colors
        title = kwargs.get('title', "Data Frame")
        dimensions = self.axis.numaxis
        colors = genrdhexcolor(len(self.results))
        ticks = [0, 0.25, 0.5, 0.75, 1]
        if dimensions == 2 or dimensions == 3:
            fig = plt.figure(figsize=(figx, figy))
            fig.suptitle(title, fontsize=16)
            if dimensions == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_zlim(0, 1)
                ax.set_xticks(ticks), ax.set_yticks(ticks), ax.set_zticks(ticks)
                for i in range(len(self.data)):
                    for c, r in enumerate(self.results):
                        if r == self.data.iat[i, 3]:
                            ax.scatter(self.data.iat[i, 0], self.data.iat[i, 1], self.data.iat[i, 2], zdir='z', c=colors[c], s=15)
                            break
            else:
                ax = fig.add_subplot(111)
                ax.set_xlim(0, 1), ax.set_ylim(0, 1)
                ax.set_xticks(ticks), ax.set_yticks(ticks)
                for i in range(len(self.data)):
                    for c, r in enumerate(self.results):
                        if r == self.data.iat[i, 2]:
                            ax.scatter(self.data.iat[i, 0], self.data.iat[i, 1], c=colors[c], s=15)
                            break
            plt.legend(self.results, labelcolor=colors, markerscale=0, handletextpad=-1.5, shadow=True)
            plt.show()
        else:
            raise TypeError("Impossible to plot with less then 2 or more then 3 dimensions.")

    def axisdispersion(self):
        if type(self) is DataFrameInfo:
            axisdisp = []
            lines = np.arange(self.numdata)
            for a in self.axis.header:
                dic = {}
                for r in self.results:
                    u = 0
                    if r not in dic:
                        dic[r] = []
                    for i in lines:
                        if r == self.data.at[i, self.resheader]:
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

    def apprognosismethod(self, inputpath, div, **kwargs):
        def euclideandist(n1, n2):
            return mt.pow((n1 - n2), 2)
        if type(self) is DataFrameInfo:
            results = []
            autoadd = kwargs.get('PredOriPath', "horizontal")
            try:
                inputdata = pd.read_csv(inputpath, div, header=None)
            except TypeError("Please check if the input csv file has some unfilled spaces."):
                return None
            dimensions = self.axis.numaxis
            pointrelid = dimensions + 1
            for inputdat in range(len(inputdata)):
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
                resulttable = []
                for r in self.results:
                    meancont = 0
                    relevtotal = 0
                    for d in datlist:
                        if d[3] == r:
                            relevtotal += d[5]
                            meancont += 1
                    resulttable += [[inputdat] + [r] + [relevtotal/meancont]]
                resulttable.sort(key=lambda x: x[2])
                for i in resulttable:
                    print(i)
                results += datlist
            for i in results:
                print(i)
            return AllPointsPrognosisMethodInfo(results)
        else:
            raise TypeError("Wrong type parameter, need to be a DataFrameInfo object.")


def createdataframe(path, div, **kwargs):  # (string, string)
    header, col = kwargs.get('header', None), []
    try:
        data = pd.read_csv(path, div, header=None)
        numcol = len(data.columns)
        numdat = data.shape[0]
        results = data[numcol - 2].unique()
        for it in range(numcol):
            col += ["a" + str(it + 1)]
        col[numcol - 1], col[numcol - 2] = "relev", "result"
        if header is None:
            data.columns = col
            axisheader = col[:numcol - 2]
            reshead = col[numcol - 2]
            revhead = col[numcol - 1]
            axis = AxisInfo(numcol - 2, axisheader)
        else:
            try:
                data.columns = header
                axisheader = header[:numcol - 2]
                reshead = header[numcol - 2]
                revhead = header[numcol - 1]
                axis = AxisInfo(numcol - 2, axisheader)
            except TypeError("Please verify if the header values match with the number of columns."):
                return None
    except TypeError("Please check if the csv file has some unfilled spaces."):
        return None
    return DataFrameInfo(data, axis, results, numdat, reshead, revhead)


# TESTES DE EXECUÇÃO

header1 = ["Umidade", "Velocidade do Vento", "Clima", "Longitude",
           "Latitude", "Altitude", "Correntes Maritimas", "Continentalidade",
           "Massa de Ar", "Vegetação", "Humidade Relativa", "Relevo", "Resultado",
           "Relevancia"]


data1 = createdataframe("data1.csv", ",", header=header1)
data1.plot(12, 12)
data1.axisdispersion().plot(16, 16, title="Dispersão Axial", orientation="horizontal")

data1.apprognosismethod("input.csv", ",")
