# - MovieIDs range from 1 to 17770 sequentially.
# - CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.

from sklearn.model_selection import train_test_split
import pandas as pd
import re

# Configurations
data_path = "/Users/yuchaoma/Documents/recommender_system/netflix-prize-data/example.txt"
# film_range = [1,20000]
# user_range = [1,2650000]
film_range = [1,1000]
user_range = [1,10000]
neighbor_count = 5

default_score = (1 + 5) / 2.0

# Load data

content = open(data_path, "r").read()
film_id = None
data = []
## row[film_id, user_id, score, date]
for line in content.split("\n"):
    if re.match(re.compile("^\d+\:$"), line.strip()):
        film_id = re.match(re.compile("^(\d+)\:$"), line.strip()).groups()[0]
    else:
        if line.count(",") == 2:
            row = line.strip().split(",")
            row[0:0] = [film_id]
            data.append(row)

## Filter data: Only contains a small portion of original data
fdata = []
for row in data:
    film_id = int(row[0])
    user_id = int(row[1])
    if film_id >= film_range[0] and film_id <= film_range[1]:
        if user_id >= user_range[0] and user_id <= user_range[1]:
            fdata.append(row)
data = fdata
data_df = pd.DataFrame(data, columns=["film_id", "user_id", "score", "date"])

import numpy
# TODO set default value to data_matrix_user_item
features = data_df["film_id"].unique()

# Split data to train and test
train_data, test_data = train_test_split(data_df, test_size=0.3, random_state=0)

users_scores =  train_data.groupby("user_id").aggregate({"film_id": { "film_ids": lambda x: [i for i in x ]},"score": {"scores": lambda x: [i for i in x]}})
# TODO this is slow
train = pd.DataFrame(columns = features)
for rindex, user_scores in users_scores.iterrows():
    train.loc[rindex] = [default_score for i in range(len(features))]
    for index, film_id in enumerate(user_scores.film_id.film_ids):
        train.loc[rindex][film_id] = user_scores.score.scores[index]

test = pd.DataFrame(columns = features)
users_scores =  test_data.groupby("user_id").aggregate({"film_id": { "film_ids": lambda x: [i for i in x ]},"score": {"scores": lambda x: [i for i in x]}})
for rindex, user_scores in users_scores.iterrows():
    test.loc[rindex] = [default_score for i in range(len(features))]
    for index, film_id in enumerate(user_scores.film_id.film_ids):
        test.loc[rindex][film_id] = user_scores.score.scores[index]

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=neighbor_count, algorithm='ball_tree').fit(train)
distances, indices = nbrs.kneighbors(test)
neighbors = pd.DataFrame(indices, index = test.index)

y_true = []
y_predict = []
random_predict = []
for rindex, row in test_data.iterrows():
    y_true.append(int(row.score))
    random_predict.append(3)
    neighbor_score = []
    for i in neighbors.loc[row.user_id]:
        neighbor_score.append(train.irow(i)[row.film_id])
    y_predict.append(sum(neighbor_score)/neighbor_count)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(numpy.asarray(y_true), numpy.asarray(y_predict))
# mse = mean_squared_error(numpy.asarray(y_true), numpy.asarray(random_predict))
import math
print(math.sqrt(mse))





# cf_model = CFModel.new()
# cf_model.train(train_data)
# predict_result = cf_model.predict(test_data)
# rmsq = calculate_rmsq(predict,test)
# print rmsq
