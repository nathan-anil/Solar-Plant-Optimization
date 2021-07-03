import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('training_data.csv')

m = len(df)
X = df.values[:,1:len(df.iloc[0])+1]
X = np.hstack((np.ones((m,1)), X))
y = df.values[:,0]
theta = np.loadtxt('theta_values.txt')

predictions = X.dot(theta)

offset = 2

plt.plot(range(1, m +1), sorted(predictions), color ='red', label = 'predicted')
plt.plot(range(1, m +1), sorted(y), color ='blue', label = 'actual')
plt.ylabel('value')
plt.xlabel('iteration')
plt.legend();
plt.show()

correct = 0
very_bad = 0;
print(len(df))
for i in range(len(y)):
    if y[i]>0 and predictions[i] < 0 or y[i] < 0 and predictions[i] > 0:
        very_bad += 1

    elif y[i] > 0:
        if y[i]*(1-offset) <= predictions[i] and y[i]*(1+offset) >= predictions[i]:
            correct += 1
    else:
        if y[i]*(1-offset) >= predictions[i] and y[i]*(1+offset) <= predictions[i]:
            correct += 1

accuracy = correct/ (m)
print("correct: ",correct)
print("incorrect: ", m - correct)
print("completelyOff: ",very_bad)
print("accuracy: ",accuracy * 100, '%')

bic_values = np.loadtxt('BIC_values.txt')
model_order = [i for i in range(1,len(bic_values)+1)]
plt.plot(model_order, bic_values,linestyle='--', marker='o')
plt.ylabel('BIC')
plt.xlabel('Polynomial model order')
plt.title('BIC value on model order')
plt.grid()

plt.show()

def mean_squared_error(X, y, theta):
  predictions = X.dot(theta)
  errors = np.subtract(predictions, y)
  sqrErrors = np.square(errors)
  result = (1/m) * np.sum(sqrErrors)
  return result

std = mean_squared_error(X, y, theta)
print("Standard Deviation: ", std)
height = [correct, m-correct]
bars = ('correct', 'incorrect')
plt.bar(bars, height, color=['red', 'cyan'], width=.5)
plt.xticks(bars, bars)
plt.show()

