from prevh import PrevhClassifier
from prevh import PrevhPlot
import pandas as pd

dataset = pd.read_csv("trainingdata.csv",",")
print(PrevhClassifier(dataset).predict_pertinence([[10,10,10]], k=6))
PrevhPlot(dataset).show()

