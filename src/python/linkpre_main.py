import numpy as np
from scipy.sparse import eye,diags
from scipy.sparse.linalg import svds,inv
from sklearn.utils.extmath import randomized_svd
import GraphUtil
from link_prediction import generate_sample, calculate_Q,approx_Q,evaluate
import time

filename="wiki.txt"
g = GraphUtil.read_data(filename,directed=False)


print("---begin generate---")
train,test=generate_sample(g, test_ratio=0.3)
g_new=GraphUtil.Graph(g.n,directed=g.directed)
for u,v in train:
    g_new.add_edge(u,v)
print("---end generate---")


print("---begin embedding---")
Q=calculate_Q(g_new)
# print("begin approx_Q")
Q2=approx_Q(g_new,1000)
forest_error=np.linalg.norm(Q.toarray()-Q2.toarray(), 'fro')/np.linalg.norm(Q.toarray(), 'fro')
print("forest_error=",forest_error)
print("---end embedding---")

print("---begin evaluate---")
precision2=evaluate(test,Q2)
print("precision2=",precision2)

Q2.data=np.log(Q2.data*1000)
print("begin rsvd")
k=128
start_time=time.time()
ur, sr, vr = randomized_svd(Q2, n_components=k, random_state=42)
embedding_rsvd = ur @ np.diag(np.sqrt(sr)) 
Q3=ur@diags(sr)@vr
precision3=evaluate(test,Q3)
print("precision3=",precision3)
end_time=time.time()

svd_error=np.linalg.norm(Q3-Q2.toarray(), 'fro')/np.linalg.norm(Q2.toarray(), 'fro')
print("svd_error=",svd_error)
print("time=",end_time-start_time)
