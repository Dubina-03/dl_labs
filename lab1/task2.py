import math
import pandas as pd
import random as rd

rd.seed(123)
pd.set_option('display.max_columns', None)

def y(x, w):
    x_s = sum(x_i * w_i for x_i, w_i in zip(x, w))
    return 1/(1 + math.e**(-x_s))

def error_3(y_m, y_r):
    return y_m * (1 - y_m) * (y_r - y_m)

def error_2(y_m, er_3, w_old):
    return y_m * (1 - y_m) * (er_3*w_old[1])

def new_w(x, error, w_old):
    d_w = [x_i * error for x_i in x]
    return [w_old_i + d_w_i for w_old_i, d_w_i in zip(w_old, d_w)]

X = 0.3
Y = 0.5
#1 to 1 to 1
w1_1, w1_2 = [0.05, rd.uniform(0, 1)], [0.05, rd.uniform(0, 1)]
#print(w1_1, w1_2)
results = []
results.append({"X" : 0.3, "Y" : 0.5, "new_w1_1": w1_1, "new_w1_2" : w1_2})
for iter in range(1000):
    y1_2 = y([1, X], w1_1)
    y1_3 = y([1, y1_2], w1_2)

    error1_3 = error_3(y1_3, Y)
    new_w1_2 = new_w([1, y1_2], error1_3, w1_2)

    error1_2 = error_2(y1_2, error1_3, w1_1)
    new_w1_1 = new_w([1, X], error1_2, w1_1) 
    
    results.append({"X" : X, "Y" : Y,  "old_w1_1" : w1_1, "old_w1_2" : w1_2, "ym1_2" : y1_2, "ym1_3" : y1_3, "new_w1_1" : new_w1_1, "new_w1_2" : new_w1_2})
    w1_1, w1_2 = new_w1_1, new_w1_2

result_df = pd.DataFrame(results) 
print(result_df)

print("Режим розпізнавання")
x = 0.1
w1_1, w1_2 = result_df.iloc[-1]["new_w1_1"], result_df.iloc[-1]["new_w1_2"]
y1_2 = y([1, x], w1_1)
y1_3 = y([1, y1_2], w1_2)
print("X:", x, "w1_1:", w1_1, "w1_2:", w1_2, "\nY:", y1_3)