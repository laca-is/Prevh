import pandas as pd
from prevh import PrevhClassifier

trainingset = pd.read_csv("trainingdata.csv",",")
print(PrevhClassifier(trainingset).predict_pertinence([[10,10,10]], k=6))

