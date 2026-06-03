import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
 
 
def create_walk_operator(G):
    #Creates the Walk operator for a given unweighted graph using the Extended Staggered Quantum Walk approach
    H = nx.Graph()
    clique_map = {}
    
    #create the clique-inserted graph H and the clique map
    for v in G.nodes():
        neighbors = sorted(G.neighbors(v), key=lambda x: (str(type(x)), str(x)))
        clique_nodes = [(v, u) for u in neighbors]
        H.add_nodes_from(clique_nodes)
        clique_map[v] = clique_nodes

        # internal clique edges
        for i in range(len(clique_nodes)):
            for j in range(i + 1, len(clique_nodes)):
                H.add_edge(clique_nodes[i], clique_nodes[j])

    #cross edges between cliques
    for u, v in G.edges():
        H.add_edge((u, v), (v, u), weight=1.0)

    node_list = list(H.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}
    dim = len(node_list)
 
    #compute the alpha tesselation and its operator
    I = np.eye(dim)
    P_alpha = np.zeros((dim, dim))
    for v, clique_nodes in clique_map.items():
        if not clique_nodes:
            continue
        d = len(clique_nodes)
        #for a unweighted graph, the normalization is just the inv of the degree
        amp = 1.0 / np.sqrt(d)  
        state = np.zeros(dim)
        for n in clique_nodes:
            state[node_index[n]] = amp
        P_alpha += np.outer(state, state)
    R_alpha = I - 2.0 * P_alpha
 
    #beta tesselation and operator
    P_beta = np.zeros((dim, dim))
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for u, v in G.edges():
        state = np.zeros(dim)
        state[node_index[(u, v)]] = inv_sqrt2
        state[node_index[(v, u)]] = inv_sqrt2
        P_beta += np.outer(state, state)
    R_beta = I - 2.0 * P_beta
 
    #get the walker operator
    W = R_beta @ R_alpha
 
    return {
        'W': W,
        'R_alpha': R_alpha,
        'R_beta': R_beta,
        'H': H,
        'clique_map': clique_map,
        'node_list': node_list,
        'node_index': node_index,
        'dim': dim,
        'G': G,
    }
 
 
def _build_initial_state(walk_result, V_max):
    #defines the initial state

    clique_map = walk_result['clique_map']
    node_index = walk_result['node_index']
    dim = walk_result['dim']
 
    init_nodes = []
    #get the nodes for the given V_max
    for i in V_max:
        init_nodes.extend(clique_map[i])
    #create the state as a vector
    psi0 = np.zeros(dim, dtype=complex)
    #notice that |E_init| here is the nummber of nodes in the cliques that represent V_max
    amp = 1.0 / np.sqrt(len(init_nodes))
    for n in init_nodes:
        psi0[node_index[n]] = amp
    return psi0


def run_walk(walk_result, V_max, epsilon=1e-4, max_steps=1000, min_steps=50, verbose=False):

    G = walk_result['G']
    W = walk_result['W']
    node_index = walk_result['node_index']
 
    #get the original edges
    edges = [(u, v) for u, v in G.edges()]
    m = len(edges)
    edge_index = {frozenset(e): k for k, e in enumerate(edges)}
    
    #an aux function that computes the probabilities according to the paper
    #and stores it in a vector
    def _edge_prob_vector(state):
        P = np.zeros(m)
        amps_sq = np.abs(state) ** 2
        for k, (u, v) in enumerate(edges):
            P[k] = amps_sq[node_index[(u, v)]] + amps_sq[node_index[(v, u)]]
        return P
 
    psi = _build_initial_state(walk_result, V_max)
 
    #t=0
    P_sum = _edge_prob_vector(psi).copy()   
    #compute the previous set of probabilities     
    Pi_prev = P_sum.copy()  
    #this will be a list of the differences between the current and previous probability                    
    diff_history = []
 
    converged = False
    T_final = 0
    for t in range(1, max_steps + 1):
        #apply walk operator
        psi = W @ psi
        #sum the edge probabilities
        P_sum += _edge_prob_vector(psi)
        #and divide it by t+1
        Pi_curr = P_sum / (t + 1)
        #compute the norm between the cirrent and previous probability vectors
        diff = np.linalg.norm(Pi_curr - Pi_prev)
        #store it in the history
        diff_history.append(diff)
        #just some logs to help
        if verbose and (t < 10 or t % 100 == 0):
            print("t=%5d   ||Pi^T - Pi^(T-1)|| = %.6e" % (t, diff))
        #if it reaches convergente, it stops
        if diff < epsilon and t >= min_steps:
            T_final = t
            converged = True
            Pi_prev = Pi_curr
            break

        Pi_prev = Pi_curr
        T_final = t

    return {
        'Pi': Pi_prev,
        'edges': edges,
        'edge_index': edge_index,
        'steps_run': T_final,
        'converged': converged,
        'diff_history': diff_history,
        'psi_final': psi,
    }
 
 
def path_weight(G, Pi, edge_index, u, v):
    #compute path weight according to the paper definition (takes into account the edge prob)
    if u == v:
        return 0.0
    try:
        path = nx.shortest_path(G, source=u, target=v)
    except nx.NetworkXNoPath:
        return 0.0
    prod = 1.0
    for a, b in zip(path[:-1], path[1:]):
        prod *= Pi[edge_index[frozenset({a, b})]]
    return prod / G.degree(v)
 

#this is the application of a classical analysis from the probability distribution 
def procedure(G, Pi, edge_index, q, refine = True):

    remaining = set(G.nodes())
    communities = {}
 
    #order the degrees
    def sort_key(x):
        return (G.degree(x), str(type(x)), str(x))
 
    # outer loop: bounded by |V(G)| iterations (one community per pass)
    for _ in range(G.number_of_nodes() + 1):
        
        #when V is an empty set, this stops
        if not remaining:
            break
    
        #pick the node with the highest degree
        v_i = max(remaining, key=sort_key)

        #defne Nbd according to the paper
        Nbd = set(G.neighbors(v_i)) & remaining

        #the v_j to be tested can not be on Nbd(v_i) or v_i
        candidates = (remaining - Nbd) - {v_i}
        for v_j in candidates:
            Nj = set(G.neighbors(v_j))
            #should not be the case, as we are analysing connected graphs
            if len(Nj) == 0:
                continue

            #condition from the paper
            if len(Nj & Nbd) >= len(Nj) / 2:
                Nbd.add(v_j)
        
        # build C(v_i) using the path-weight threshold q
        C = {v_i}
        for v_j in Nbd:
            #note that this already needs the compuetd Pi (limiting prob array)
            if path_weight(G, Pi, edge_index, v_i, v_j) < q:
                C.add(v_j)

        #procedure 2 described in the paper 
        #it is just an extra step inside procedure 1
        if refine:
            to_remove = set()
            for v_j in C:
                if v_j == v_i:
                    continue
                #check the d_C(vi)(vj)
                internal_deg = sum(1 for nb in G.neighbors(v_j) if nb in C)
                #compare it to the degree of v_j
                #this way, we check how much of the neighbours from v_j are from its community
                if internal_deg <= G.degree(v_j) / 2:
                    to_remove.add(v_j)
            #update the community
            #the v_j that did not pass will become a candidate again
            C -= to_remove

        communities[v_i] = sorted(C, key=lambda x: (str(type(x)), str(x)))
        remaining -= C
 
    return communities
 
 
def detect_communities(G, V_max, q=None, epsilon=1e-4, max_steps=5000,
                       min_steps=50, refine=True, verbose=False):
    
    #this is the function that puts everything together
    if q is None:
        #this is the value presented in the paper for q, so this will be our default
        q = 1.0 / max(G.number_of_edges(), 1)
    
    #compute the walk operator 
    walk_operator = create_walk_operator(G)
    #run it and store the results
    walk_run = run_walk(walk_operator, V_max, epsilon=epsilon, max_steps=max_steps,
                        min_steps=min_steps, verbose=verbose)
    
    #get the limiting probability vector Pi and the edge index
    Pi = walk_run['Pi']
    edge_index = walk_run['edge_index']
    
    #run the procedure to estimate communities 
    comms = procedure(G, Pi, edge_index, q, refine=refine)
    
    #return the communities and other useful info (it can be used for the visualization)
    return {
        'communities': comms,
        'Pi': Pi,
        'edges': walk_run['edges'],
        'edge_index': edge_index,
        'steps_run': walk_run['steps_run'],
        'converged': walk_run['converged'],
        'q': q,
        'walk_operator': walk_operator,
        'walk_run': walk_run,
    }
 
 
def visualize_communities(G, result, seed=42, show_edge_probs=True):

    comms = result['communities']
    Pi = result['Pi']
    edges = result['edges']
    edge_index = result['edge_index']
    q = result['q']
    
    # assign colors to communities
    #focus on communities with more than 1 node, otherwise, they are just a singleton
    non_singleton = [rep for rep, m in comms.items() if len(m) > 1]
    cmap = plt.get_cmap('tab20')
    rep_color = {rep: cmap(i % 20) for i, rep in enumerate(non_singleton)}
 
    node_color = []
    for v in G.nodes():
        owner = None
        for rep, members in comms.items():
            if v in members:
                owner = rep
                break
        if owner in rep_color:
            node_color.append(rep_color[owner])
        else:
            node_color.append((0.7, 0.7, 0.7, 1.0))  # gray for singletons
 
    pos = nx.spring_layout(G, seed=seed)
 
    n_panels = 1 + int(show_edge_probs)
    fig, axes = plt.subplots(1, n_panels, figsize=(16,6))
    if n_panels == 1:
        axes = [axes]
    ax_iter = iter(axes)
 
    # graph with communities
    ax = next(ax_iter)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, edge_color='lightgray')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                           node_size=350, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    n_real = len(non_singleton)
    n_single = len(comms) - n_real
    ax.set_title("Communities on G")
    ax.axis('off')

    #histogram with the limiting probabilities
    #just so I can compare with the paper
    if show_edge_probs:
        ax = next(ax_iter)
        order = np.argsort(-Pi)
        labels = ["%s-%s" % (edges[i][0], edges[i][1]) for i in order]
        ax.bar(range(len(Pi)), Pi[order], color='steelblue')
        ax.set_xticks(range(len(Pi)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("p(e)")
        ax.set_title("Edge-probability vector $\\Pi$  "
                     "(T = %d, %s)"
                     % (result['steps_run'],
                        'converged' if result['converged'] else 'max_steps reached'))
 
    plt.tight_layout()
    plt.show()
 

def vertices_of_max_degree(G):
    #gets the vertices with max degree
    if G.number_of_nodes() == 0:
        return []
    d_max = max(dict(G.degree()).values())
    return [v for v, d in G.degree() if d == d_max]
 
 
def top_k_degree_vertices(G, k):
    #gets the top k vertices with highest degree
    ranked = sorted(G.nodes(),
                    key=lambda v: (-G.degree(v), str(type(v)), str(v)))
    return ranked[:k]


def modularity(G, communities):
    #compute the modularity of the given communities
    m = G.number_of_edges()
    if m == 0:
        return 0.0

    #compute the denominator
    two_m = 2.0 * m
    deg = dict(G.degree())

    Q = 0.0
    for members in communities.values():
        #we are already restricting to the terms that are not zero in the sum
        S = set(members)
        intra_edges = sum(1 for u, v in G.edges() if u in S and v in S)
        sum_deg = sum(deg[v] for v in S)
        #the formula uses B_uv = A_uv - (d_u * d_v) / (2m)
        #so the first term reflects the adjacency contribution
        #we insert *2 due to double counting in the expression
        Q += (2.0 * intra_edges) / two_m - (sum_deg / two_m) ** 2

    return Q