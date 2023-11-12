import pandas as pd
import numpy as np

# %%
def func_gaus(w1, w2, x1, x2, g=0.1):
    return np.exp(-((w1 - x1) ** 2 + (w2 - x2) ** 2) / (g ** 2))

#%%
#TRAINING
training = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 2], [2, 4, 6], [3, 3, 6], [6, 2, 8], [2, 3, 5], [4, 1, 5], [4, 4, 8], [3, 5, 8], [5, 1, 6], [9, 0, 9], [3, 1, 4], [0, 5, 5], [6, 3, 9], [5, 2, 7], [0, 3, 3], [1, 8, 9]]
# %%
delta = 1
y_summation = dict()
y_training = dict ()
#defining the summation layer + connection between pattern am=nd summation layers
for example, y in zip(training, [t[2] for t in training]):
    if y not in y_summation.keys():
        y_summation[y] = 0
        y_training[y] = [] 
    y_training[y] += [example]

print(sorted(y_summation.keys()))
print("Classes of summation layer + training examples\n", dict(sorted(y_training.items())))

# %%
#PREDICTING
x1, x2 =2, 7
print("X1:", x1, "X2:", x2)
#calculating y for the pattern layer
y_pattern = [func_gaus(t[0], t[1], x1, x2) for t in training]
print("The values of y pattern layer\n", y_pattern)
#calculating the values of summation layer
for i in range(len(y_pattern)):
    y_summation[training[i][2]] += y_pattern[i] 
print("The values of y summation layer\n",y_summation)
#defining the output
y_output = max(y_summation, key=lambda k: y_summation[k])
print("The predicted class is", y_output)
#cleaning the sum values
y_summation = {key: 0 for key in y_summation}
# %%
