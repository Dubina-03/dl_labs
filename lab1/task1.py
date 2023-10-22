import numpy as np
import math
import random as rd
import pandas as pd

rd.seed(123)
pd.set_option('display.max_columns', None)
#n - size
n = 3
x = [1, 0.3, 0.8]
w_old = [rd.random() for _ in range(len(x))]
y_r = 0.7

results = []
results.append({"x0" : 1, "x1" : 0.3, "x2" : 0.8, "Y" : 0.7, "new_w0" : w_old[0], "new_w1" : w_old[1], "new_w2" : w_old[2]})

for iter in range(1000):
    x_sum = np.dot(x, w_old)
    y_m = 1/(1+math.e**(-x_sum))

    error = y_m * (1 - y_m) * (y_r - y_m)
    d_w = [x_i*error for x_i in x]
    w_new = [w_old_i + d_w_i for w_old_i, d_w_i in zip(w_old, d_w)]

    results.append({"x0" : 1, "x1" : 0.3, "x2" : 0.8, "Y" : 0.7, "x_sum": x_sum, "y_m": y_m, "old_w0" : w_old[0], "old_w1" : w_old[1], "old_w2" : w_old[2], "new_w0" : w_new[0], "new_w1" : w_new[1], "new_w2" : w_new[2]})
    w_old = w_new
result_df = pd.DataFrame(results) 
print(result_df)

print("Режим розпізнавання")
x = [1, 0.5, 0.2]
w = [result_df.iloc[-1]["new_w0"], result_df.iloc[-1]["new_w1"], result_df.iloc[-1]["new_w2"]]
x_sum = np.dot(x, w)
y_m = 1/(1+math.e**(-x_sum))
print("X:", x, "w:", w, "\nY:", y_m)
