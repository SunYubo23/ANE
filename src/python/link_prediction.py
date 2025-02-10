import random
import GraphUtil
from scipy.sparse import eye,lil_matrix,csc_matrix
from scipy.sparse.linalg import svds,inv
from sklearn.utils.extmath import randomized_svd
import numpy as np

def generate_sample(graph, test_ratio=0.3):
    edges = list(zip(graph.u, graph.v))
    
    num_test_edges = int(len(edges) * test_ratio)
    test_positive_edges = random.sample(edges, num_test_edges)
    print("test set (positive sample) with ", num_test_edges, " edges")

    training_edges = list(set(edges) - set(test_positive_edges))
    print("train set with ", len(training_edges), " edges")

    all_possible_edges = set()
    for u in range(graph.n):
        for v in range(u + 1, graph.n): 
            all_possible_edges.add((u, v))
    test_negative_edges = random.sample(all_possible_edges - set(edges), num_test_edges)
    print("test set (negative sample) with ", num_test_edges, " edges")

    test_edges = [(edge, 1) for edge in test_positive_edges] + [(edge, 0) for edge in test_negative_edges]
    
    return training_edges, test_edges

def calculate_Q(graph):
    A = GraphUtil.adjacency_matrix(graph)
    D = GraphUtil.degree_matrix(graph)
    L = D - A
    I = eye(L.shape[0], format='csc')
    Q = inv(I + L)
    return Q

def approx_Q(graph,sample_num):
    n=graph.n
    d=graph.degree_vector()
    in_forests = np.full(n, False)
    next_node = np.zeros(n, dtype=int)
    root = np.zeros(n, dtype=int)
    ans = lil_matrix((n, n), dtype=float)
    nbr = graph.neighbor_list()

    for _ in range(sample_num):
        in_forests.fill(False)
        for src in range(n):
            u = src
            while not in_forests[u]:
                if random.random() * (d[u] + 1) < 1:
                    in_forests[u] = True
                    root[u] = u
                    ans[u, u] += 1 / sample_num
                    break
                next_node[u] = random.choice(nbr[u])
                u = next_node[u]
            
            r = root[u]
            u = src
            while not in_forests[u]:
                in_forests[u] = True
                root[u] = r
                ans[u, r] += 1 / sample_num 
                u = next_node[u]
    ans_csc = ans.tocsc()
    return ans_csc


def evaluate(test_edges, Q):
    scores = []
    for (u, v), label in test_edges:
        score=Q[u,v]
        scores.append(((u, v), score, label))
    scores.sort(key=lambda x: x[1], reverse=True)

    top_k_count = int(len(scores) * 0.5)
    top_k_edges = scores[:top_k_count]

    correct_positive = sum(1 for _, _, label in top_k_edges if label == 1)  # 
    total_samples = len(top_k_edges) 
    precision = correct_positive / total_samples
    print("total samples: ", total_samples)

    return precision