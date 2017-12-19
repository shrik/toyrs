# Latex
# $$ \hat{r}_{ui} = \frac{1}{|N_i(u)|}  \sum_{v \in N_i(u)}{r_{vi}} $$
# pseudo code
# expect_rate(user, item) = 1/count(neighbors_rated_i(user)) * \
#        sum(neighbors_rated_i(user).map{|u| ratings(u,i)})


# pseudo code
#

from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
indices
distances
