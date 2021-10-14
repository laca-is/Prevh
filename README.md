# Prevh
This is a python data mining library. 

# Installing
Package Documentation: https://pypi.org/project/prevhlib/

Installing with pip: pip install prevhlib

# Usage

## Dataset File sample

```text
axis1,axis2,axis3,cluster,relevance
10,10,10,Blue,1.0
15,15,15,Blue,1.0
20,20,20,Blue,1.0
45,45,45,Green,1.0
50,50,50,Green,1.0
55,55,55,Green,1.0
80,80,80,Red,1.0
85,85,85,Red,1.0
90,90,90,Red,1.0
```

## Python sample:

```python
import prevh as ph
# create the dataset
dataset = ph.datasetfromCSV("C:/trainingdata.csv", ",")
# executes the prediction for the follow information
predictions = dataset.predict([[11, 11, 11], [32, 32, 32], [91, 91, 91]], kNeighbors=6)
# to see how the normalization was made in the data set (Output = pandas.DataFrame)
print(predictions.predict_data)
# to see the prediction results (Output = list)
print(predictions.predict_results)
```