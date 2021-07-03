import pandas as pd
from pytz import timezone
import pytz
import matplotlib.pyplot as plt
import numpy as np
import math

dataset = pd.read_csv('SolarPrediction.csv')
dataset = dataset.sort_values(['UNIXTime'], ascending = [True])
hawaii= timezone('Pacific/Honolulu')
dataset.index =  pd.to_datetime(dataset['UNIXTime'], unit='s')
dataset.index = dataset.index.tz_localize(pytz.utc).tz_convert(hawaii)

dataset['DayOfYear'] = dataset.index.strftime('%j').astype(int)
dataset['TimeOfDay(s)'] = dataset.index.hour*60*60 + dataset.index.minute*60 + dataset.index.second

dataset.to_csv('sf.csv', index = False)

dataset.drop(['Data','Time','TimeSunRise','TimeSunSet', 'UNIXTime', 'Speed', 'WindDirection(Degrees)'], inplace=True, axis=1)

# highest degree of the polynomial function
num_order = 4
m = len(dataset)
engineered_features = [ 'Temperature', 'DayOfYear', 'TimeOfDay(s)', 'Humidity', 'Pressure']

for i in engineered_features:
    for j in range(2,num_order+1):
        new_name = str(i) + str(j)
        dataset[new_name] = np.power(dataset[i],j)

print(dataset.iloc[0])


def set_min_max(dataset):
    minmax = list()
    for i in range(len(dataset.iloc[0])):
        col_values = dataset.iloc[:,i].tolist()
        value_min = min(col_values)
        value_max = max(col_values)
        value_sum = sum(col_values)
        minmax.append([value_min, value_max, (value_sum/len(dataset))])
    return minmax

def normalize_dataset(dataset, minmax):
        for j in range(len(dataset.iloc[0])):

            dataset.iloc[:,j] = (dataset.iloc[:,j] - minmax[j][2]) / (minmax[j][1] - minmax[j][0])


#Plotting Features
# radiation_values = dataset.iloc[:,i].tolist()
# heights = [i for i in radiation_values]

plt.plot(range(1, m +1), sorted(dataset['Radiation']))
plt.ylabel("Radiance")
plt.xlabel("1...n")
plt.title("Solar Radiance")
plt.show()
# for i in range(1, len(dataset.columns)):
#     plt.title("Radiation vs " + dataset.columns[i])
#     plt.xlabel('Radiation')
#     plt.ylabel(dataset.columns[i])
#     plt.plot(dataset['Radiation'].sort_values(), dataset[dataset.columns[i]].sort_values())
#     plt.show()

# Normalizing data
minmax = set_min_max(dataset)
normalize_dataset(dataset, minmax)


# Separating data into training and testing 25:75

training_size = math.ceil(0.75 * len(dataset))
testing_size = len(dataset) - training_size

training_data = dataset.iloc[:training_size]
testing_data = dataset.iloc[training_size:]

training_data.to_csv('training_data.csv', index = False)
testing_data.to_csv('testing_data.csv', index = False)

print(testing_size)