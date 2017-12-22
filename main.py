# - MovieIDs range from 1 to 17770 sequentially.
# - CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.


# TODO 1. speed is too slow
# use neighbors who has rated the same item
# 2. use item to item

from sklearn.model_selection import train_test_split
import pandas as pd
# Configurations
data_path = "/Users/yuchaoma/Documents/recommender_system/netflix-prize-data/example.txt"
# film_range = [1,20000]
# user_range = [1,2650000]
film_range = [1,1000]
user_range = [1,100000]
neighbor_count = 5
default_score = (1 + 5) / 2.0

# Load data
from preprocess import file_to_dataframe, to_sparse_dataframe
data_df = file_to_dataframe(data_path, user_range=user_range, film_range=film_range)
# features
import numpy
features = data_df["film_id"].unique()
# Split data to train and test
train_data, test_data = train_test_split(data_df, test_size=0.3, random_state=0)
# To feed data
train_matrix = to_sparse_dataframe(train_data, features=features, default_score= default_score)
test_user_ids = test_data["user_id"].unique()
test_users_data = train_data.loc[train_data["user_id"].isin(test_user_ids)]
# TODO We should not choose user that we haven't seen.
test_data = test_data.loc[test_data["user_id"].isin(test_users_data["user_id"].unique())]
test_matrix = to_sparse_dataframe(test_users_data, features=features, default_score= default_score)

from sklearn.neighbors import NearestNeighbors
from cf_validate import validate


print("start Model")
for neighbor_count in range(5,6):
    # train Model
    # import pdb;pdb.set_trace()
    nbrs = NearestNeighbors(n_neighbors=neighbor_count, n_jobs=-1,
                            metric='cosine', algorithm='brute').fit(train_matrix)
    distances, indices = nbrs.kneighbors(test_matrix)
    neighbors = pd.DataFrame(indices, index = test_matrix.index)
    # validate Model
    print("start validate")
    validate(test_data, neighbors=neighbors, train_matrix=train_matrix)
