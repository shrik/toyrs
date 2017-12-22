#sparse matrix
import numpy as np
from scipy.sparse import csr_matrix
m = csr_matrix(([1,2,3], ([1,2,3], [4,5,6])),shape=(8,8), dtype=np.int8)
import pandas as pd
sdf = pd.SparseDataFrame(m)
sdf.fillna(3.0)
sdf.density
