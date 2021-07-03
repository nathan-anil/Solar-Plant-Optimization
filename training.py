import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_data.csv')

X = df.values[:,1:len(df.iloc[0])+1]
y = df.values[:,0]

m = len(y)
X = np.hstack((np.ones((m,1)), X))


def compute_cost(X, y, theta):
  predictions = X.dot(theta)
  errors = np.subtract(predictions, y)
  sqrErrors = np.square(errors)
  J = (1 / (2 * m)) * np.sum(sqrErrors)
  return J

def mean_squared_error(X, y, theta):
  predictions = X.dot(theta)
  errors = np.subtract(predictions, y)
  sqrErrors = np.square(errors)
  result = (1/m) * np.sum(sqrErrors)
  return result

def gradient_descent(X, y, theta, alpha, iterations):

    cost_history = np.zeros(iterations)
    for i in range(iterations):

        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta =  (alpha / m) * X.transpose().dot(errors)
        theta = theta - sum_delta
        cost_history[i] = compute_cost(X, y, theta)


    return theta, cost_history


iterations = 1300

theta = np.zeros(len(X[0]))
alpha = .05;
theta_1, cost_history_1 = gradient_descent(X, y, theta, alpha, iterations)

alpha = 0.10;
theta_2, cost_history_2 = gradient_descent(X, y, theta, alpha, iterations)

alpha = 0.15;
theta_3, cost_history_3 = gradient_descent(X, y, theta, alpha, iterations)


alpha = 0.5;
theta_4, cost_history_4 = gradient_descent(X, y, theta, alpha, iterations)


alpha = .8;
theta_5, cost_history_5 = gradient_descent(X, y, theta, alpha, iterations)

# SAVE THETA VALUES TO FILE
final_theta  = theta_5
np.savetxt('theta_values.txt', final_theta)

plt.plot(range(1, iterations +1), cost_history_1, color ='purple', label = 'alpha = 0.05')
plt.plot(range(1, iterations +1), cost_history_2, color ='red', label = 'alpha = 0.10')
plt.plot(range(1, iterations +1), cost_history_3, color ='green', label = 'alpha = 0.15')
plt.plot(range(1, iterations +1), cost_history_4, color ='pink', label = 'alpha = 0.5')
plt.plot(range(1, iterations +1), cost_history_5, color ='blue', label = 'alpha = 0.8')

plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("Effect of Learning Rate On Convergence of Gradient Descent")
plt.legend()
plt.show()


# CALCULATING THE BAYESAN INFORMATION CRITERION


def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic

# number of parameters
num_params = len(final_theta)
print('Number of parameters: %d' % (num_params))
# calculate the error
mse = mean_squared_error(X, y, final_theta)
print('MSE: %.3f' % mse)
# calculate the bic
bic = calculate_bic(len(y), mse, num_params)
print('BIC: %.3f' % bic)
# add to file
file = open("BIC_values.txt", "a")  # append mode
file.write(str(bic))
file.write("\n")
file.close()