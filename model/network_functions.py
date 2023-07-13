import numpy as np
from scipy.sparse import diags
import networkx as nx 

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return data[s<m]

# Create a class for generating line, graph and star matrices by code copying
def stochastic_network_matrix(n_agents):
    """
    Weights for static network

    Designed as a stochastic matrix as given in Nedic's paper
    Stochastic matrix based on nodal degrees
    """
    if n_agents == 1:
        return np.array([[1.]])
    elif n_agents == 2:
        weight_matrix = np.array([[3./4., 1./4.], \
                                [1./4., 3./4.]])
        return weight_matrix
    elif n_agents == 3:
        weight_matrix = 0.5*np.eye(3) + \
        0.5*np.array([[2./3., 1./3., 0.], \
                      [1./3., 1./3., 1./3.], \
                      [0., 1./3., 2./3.]])
    elif n_agents == 4:
        weight_matrix = np.array([[2./3., 1./6., 0., 1./6.], \
                                [1./6., 2./3.,1./6., 0.],\
                                [0., 1./6., 2./3., 1./6.],\
                                [1./6., 0., 1./6., 2./3.]])
        return weight_matrix
    else:
        raise Exception('Current implementation only for 1 to 4 agents')
    return weight_matrix

def convert_to_ds(A):
    err = 0.1
    while err > 1e-3:
        A_prev = A
        # Make row stochastic
        A = A/np.sum(A, axis=1).reshape(-1,1)
        # Make column stochastic
        A = np.transpose(A.T/np.sum(A, axis=0).reshape(-1,1))
        err = np.linalg.norm(A-A_prev)
    return A

def graph_diameter(A):
    """
    Input: Adjacency matrix
    """
    G = nx.from_numpy_matrix(A)
    return nx.diameter(G)
    
def generate_connected_graph_edges(n, n_edges):
    """
    Generate a random connected graph with specified number of edges
    """
    assert n >= 2
    E_MIN = n-1
    E_MAX = 0.5*(n**2 - n)
    assert n_edges >= E_MIN
    assert n_edges <= E_MAX
    A = np.zeros((n,n))
    in_line = [0, 1]
    A[0, 1] = 1
    out_of_line = [i for i in range(2, n)]
    for i in range(2, n):
        node_in_line_idx = np.random.randint(0, len(in_line), 1)
        node_in_line = in_line[node_in_line_idx[0]]
        A[i, node_in_line] = 1
        in_line.append(i)
        out_of_line.remove(i)
    
    A = A + np.transpose(A)
    A = A + np.eye(n)
    
    for i in range(n_edges - E_MIN):
        can_add = 0
        while can_add == 0:
            random_agent = np.random.randint(0, n, 1)[0]
            can_add = n - np.sum(A[:, random_agent])
        
        n_indices = np.where(A[:,random_agent]==0)[0].tolist()
        random_connection = np.random.randint(0, n, 1)[0]
        A[random_agent, random_connection] = 1
    A = A + np.transpose(A)
    A = np.array(A>0)*1
    A = A.astype(np.float)
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigs = list(np.linalg.eig(L)[0])
    # Remove one of the zeros
    eigs.remove(np.min(eigs))
    # Sparsest cut: minimize the ratio of the number of edges across the cut 
    # divided by the number of vertices in the smaller half of the partition
    fiedler_eig = np.min(eigs)
    A_base = A
    A = convert_to_ds(A)
    return A, D, A_base, fiedler_eig, graph_diameter(A)
    
def generate_connected_graph(n):
    A = np.zeros((n,n))
    for i in range(n-1):
        n_neighbor = np.random.randint(1, i+2)
        neighbor = np.random.randint(0, i+1, n_neighbor)
        for j in neighbor:
            A[i+1, j] = 1

    A = A + np.transpose(A)
    A = A + np.eye(n)
    A = convert_to_ds(A)
    return A


if __name__ == '__main__':
    print (generate_connected_graph_edges(n=5, n_edges=10))
