# Validate
from sklearn.metrics import mean_squared_error
import numpy as np
import math

def mean_absolute_error(y_true, y_predict):
    # 1/N * |y_true - y_predict|
    err = y_predict - y_true
    sum_error = float(np.sum(np.absolute(err)))
    return sum_error/len(y_true)

def validate(test_data, train_matrix=None, neighbors=None):
    y_true = []
    y_predict = []
    random_predict = []
    for rindex, row in test_data.iterrows():
        y_true.append(int(row.score))
        random_predict.append(3)
        neighbor_score = []
        for i in neighbors.loc[row.user_id]:
            neighbor_score.append(train_matrix.iloc[i][row.film_id])
        y_predict.append(sum(neighbor_score)/len(neighbor_score))
    print("loop end")
    mse = mean_squared_error(np.asarray(y_true), np.asarray(y_predict))
    mae = mean_absolute_error(np.asarray(y_true), np.asarray(y_predict))
    print("mean absolute error is", mae )
    print("mean root squared error is", math.sqrt(mse))
    return [mae, math.sqrt(mse)]
