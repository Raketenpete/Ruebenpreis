import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd 
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings(action="ignore", module="sklearn", message="^internal gelsd")

def linearizer(lowerLimit, upperLimit): # untere und obere Grenze
    upperLimit += 1
    out = []
    while (lowerLimit < upperLimit):
        out.append(lowerLimit)
        lowerLimit += 1
    return out 

inputFile = "input.csv"
inputData = pd.read_csv(inputFile, delimiter=',', header=None)
y = inputData.iloc[:,0]  #erste Zahl wird ausgewählt, nächste wäre [:,1] etc.
length_y = len(y.index)

x= []
x = linearizer(0, length_y - 1) #minus 1 um position 0 zu kompensieren
x = pd.DataFrame(x)

model = LinearRegression()
model.fit(x, y)

#neues array, 1 position voraus
positionsToPredict = 0 
new_data = linearizer(length_y, (positionsToPredict + length_y))
new_data = pd.DataFrame(new_data)

prediction = model.predict(new_data)

print("\nVoraussagung aufgrund von linearer Regression: \n\n", prediction)