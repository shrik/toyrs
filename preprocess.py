
import re
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def file_to_dataframe(path, user_range=None, film_range=None):
    content = open(path, "r").read()
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
    return pd.DataFrame(fdata, columns=["film_id", "user_id", "score", "date"])


# @params data: [film_id,user_id,score,date] DataFrame
# @return SparseDataFrame columns=features index=collect(user_id)
def to_sparse_dataframe(data, features=None, default_score=0):
    users_scores =  data.groupby("user_id").aggregate({"film_id": { "film_ids": lambda x: [i for i in x ]},"score": {"scores": lambda x: [i for i in x]}})
    scores = []
    row_ind = []
    col_ind = []
    features_hash = {}
    for index, i in enumerate(features):
        features_hash[i] = index
    user_index = 0
    for rindex, user_scores in users_scores.iterrows():
        for index, film_id in enumerate(user_scores.film_id.film_ids):
            scores.append(float(user_scores.score.scores[index]))
            row_ind.append(user_index)
            col_ind.append(features_hash[film_id])
        user_index += 1
    m = csr_matrix((scores, (row_ind, col_ind)), shape=(len(users_scores), len(features)), dtype=np.float)
    sdf = pd.SparseDataFrame(m, columns=features, index=users_scores.index)
    return sdf.fillna(default_score)
