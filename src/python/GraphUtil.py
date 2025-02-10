import numpy as np
from scipy.sparse import csc_matrix,diags
from scipy.sparse.linalg import inv

class Graph:
    def __init__(self, n, m=None, u=None, v=None, w=None, directed=None):
        self.n = n
        self.m = m if m is not None else 0  
        self.u = u if u is not None else []  
        self.v = v if v is not None else []  
        self.w = w if w is not None else []  
        self.directed = directed if directed is not None else False 
    
    def add_edge(self, u, v, w=None):
        if not self.directed and u > v:
            u, v = v, u  
        self.u.append(u)
        self.v.append(v)
        self.w.append(w if w is not None else 1)  
        self.m += 1
    def neighbor_list(self):
        neighbors = [[] for _ in range(self.n)]
        for i in range(self.m):
            u, v = self.u[i], self.v[i]
            neighbors[u].append(v)
            if not self.directed:
                neighbors[v].append(u)
        return neighbors
    
    def degree_vector(self):
        degree = [0] * self.n
        for i in range(self.m):
            u, v = self.u[i], self.v[i]
            degree[u] += 1
            if not self.directed:
                degree[v] += 1
        return degree        


def read_data(filename, directed):
    u, v, w = [], [], []

    with open(f"../data/{filename}", "r") as file:
        lines = file.readlines()
    n = int(lines[0].strip())
    lines = lines[1:] 
    print(n)

    for line in lines:
        line = line.strip()
        if line.startswith('#') or line.startswith('%'):
            continue
        split_line = line.split()
        
        uu = int(split_line[0])
        vv = int(split_line[1])
        ww = 1.0 if len(split_line) < 3 else float(split_line[2])
        
        u.append(uu)
        v.append(vv)
        w.append(ww)
    
    m = len(u) 

    return Graph(n, m, u, v, w, directed)

def adjacency_matrix(graph):
    n = graph.n
    data = graph.w
    row = graph.u
    col = graph.v
    matrix = csc_matrix((data, (row, col)), shape=(n, n))

    if not graph.directed:
        matrix = matrix + matrix.T
    return matrix

def degree_matrix(graph):
    n = graph.n         
    m = graph.m         
    u = graph.u        
    v = graph.v         
    w = graph.w          
    directed = graph.directed  
    
    row = []
    col = []
    data = []

    for i in range(m):
        if directed:
            row.append(u[i])
            col.append(u[i])
            data.append(w[i])  
            
            row.append(v[i])
            col.append(v[i])
            data.append(w[i])  
        else:
            row.append(u[i])
            col.append(u[i])
            data.append(w[i])  
            
            row.append(v[i])
            col.append(v[i])
            data.append(w[i])  
    
    D = csc_matrix((data, (row, col)), shape=(n, n))
    
    return D


def transition_matrix(graph):
    d = graph.degree_vector()
    A = adjacency_matrix(graph)
    
    for i in range(graph.n):
        if d[i]==0:
            d[i]=1e6
        else:
            d[i]=1/d[i]
    D_inv = diags(d)
    P = D_inv @ A
    for i in range(len(d)):
        if d[i] == 1e6:
            P[i, :] = 0
    return P