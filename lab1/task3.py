import math
import pandas as pd
import random as rd
import numpy as np

rd.seed(123)
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def y(x, w):
    x_s = sum(x_i * w_i for x_i, w_i in zip(x, w))
    return 1/(1 + np.exp(-x_s))

def error_3(y_r, y_m):
    return y_m * (1 - y_m) * (y_r - y_m)

def error_2(y_m, er_3, w_old):
    #return y_m * (1 - y_m) * sum([er_3*w_old_i for w_old_i in w_old[1:]])
    return y_m * (1 - y_m) * er_3*w_old[1]

def new_w(x, error, w_old):
    d_w = [x_i * error for x_i in x]
    return [w_old_i + d_w_i for w_old_i, d_w_i in zip(w_old, d_w)]

X = [[1, 1], [2, 1], [1, 3], [3, 2], [2, 0], [4, 3], [5, 1], [2, 2], [2, 6], [1, 4]]
Y = [0.2, 0.3, 0.4, 0.5, 0.2, 0.7, 0.6, 0.4, 0.8, 0.5]
w1_1, w2_1, w3_1, w1_2 = [0.05, rd.uniform(0, 1), rd.uniform(0, 1)], [0.05, rd.uniform(0, 1), rd.uniform(0, 1)], [0.05, rd.uniform(0, 1), rd.uniform(0, 1)], [0.05, rd.uniform(0, 1), rd.uniform(0, 1), rd.uniform(0, 1)]
#print(w1_1, w1_2)
#X = [[1, 2], [2, 1], [1, 1], [0, 1], [1, 0]]
#Y = [0.3, 0.3, 0.2, 0.1, 0.1]
#1 to 1 to 1
#w1_1, w2_1, w3_1, w1_2 = [0.05, 0.1, 0.2], [0.05, 0.4, 0.5], [0.05, 0.3, 0.6], [0.05, 0.7, 0.8, 0.9]
results = []
results.append({"X" : X[0], "Y" : Y[0], "new_w1_1" : w1_1, "new_w2_1" : w2_1, "new_w3_1" : w3_1, "new_w1_2" : w1_2})
for iter in range(1000):
    for j in range(len(X)):
        if iter == 0:
            if j < 4:
                j += 1
            else:
                break
        y1_2 = y([1, X[j][0], X[j][1]], w1_1)
        y2_2 = y([1, X[j][0], X[j][1]], w2_1)
        y3_2 = y([1, X[j][0], X[j][1]], w3_1)
        y1_3 = y([1, y1_2, y2_2, y3_2], w1_2)

        error1_3 = error_3(Y[j], y1_3)
        new_w1_2 = new_w([1, y1_2, y2_2, y3_2], error1_3, w1_2)

        error1_2 = error_2(y1_2, error1_3, w1_1)
        new_w1_1 = new_w([1, X[j][0], X[j][1]], error1_2, w1_1) 

        error2_2 = error_2(y2_2, error1_3, w2_1)
        new_w2_1 = new_w([1, X[j][0], X[j][1]], error2_2, w2_1) 

        error3_2 = error_2(y3_2, error1_3, w3_1)
        new_w3_1 = new_w([1, X[j][0], X[j][1]], error3_2, w3_1) 
        
        results.append({"X" : X[j], "Y" : Y[j],  "old_w1_1" : w1_1, "old_w2_1" : w2_1, "old_w3_1" : w3_1, "old_w1_2" : w1_2, "ym1_2" : y1_2, "ym2_2" : y2_2, "ym3_2" : y3_2, "ym1_3" : y1_3, "new_w1_1" : new_w1_1, "new_w2_1" : new_w2_1, "new_w3_1" : new_w3_1, "new_w1_2" : new_w1_2})
        w1_1, w2_1, w3_1, w1_2 = new_w1_1, new_w2_1, new_w3_1, new_w1_2

result_df = pd.DataFrame(results) 
#Will print only last 20 records
print(result_df)

#recognition
def recog_y(x, w1_1, w2_1, w3_1, w1_2):
    y1_2 = y([1, x[0], x[1]], w1_1)
    y2_2 = y([1, x[0], x[1]], w2_1)
    y3_2 = y([1, x[0], x[1]], w3_1)
    y1_3 = y([1, y1_2, y2_2, y3_2], w1_2)
    return y1_3

x = [4, 2]
w1_1 = result_df.iloc[-1]["new_w1_1"]
w2_1 = result_df.iloc[-1]["new_w2_1"]
w3_1 = result_df.iloc[-1]["new_w3_1"]
w1_2 = result_df.iloc[-1]["new_w1_2"]

print("Режим розпізнавання")
y_res = recog_y(x, w1_1, w2_1, w3_1, w1_2)
print("X:", x, "w1_1:", w1_1, "w2_1:", w2_1, "w3_1:", w3_1, "w1_2:", w1_2, "\nY:", y_res)