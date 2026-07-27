# single component in cubic box with Random Geometric Graph method
import numpy as np
import sys
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import Counter


# Parameters
N = int(sys.argv[1])             # Number of chains
f = int(sys.argv[2])             # Binding sites per chain
p = float(sys.argv[3])           # Binding probability
n_repeat = int(sys.argv[4])      # How many times to repeat
L_final=float(sys.argv[5])
r_detect=float(sys.argv[6])

rng = np.random.default_rng()

def generate_shuffled_array(N, f):
    array = np.repeat(np.arange(N), f)
    np.random.shuffle(array)
    return array


def sample_with_limit(bond_list, N, f, seed=None):
    if seed is not None:
        random.seed(seed)

    result = []
    counts = Counter()

    # 打乱原列表
    candidates = bond_list[:]
    random.shuffle(candidates)

    for x in candidates:  # bond lists，like (1, 2)
        i, j = x
        if i!=j:
            if counts[i] < f and counts[j] < f:  # each vertex contain < f edges
                result.append(x)
                counts[i] += 1
                counts[j] += 1
                if len(result) >= N:
                    break
        else:
            if counts[i] < f-1 and counts[j] < f-1:  # each vertex contain < f edges
                result.append(x)
                counts[i] += 1
                counts[j] += 1
                if len(result) >= N:
                    break


    if len(result) < N:
        raise ValueError("no enough candidates, check  N and f value")

    return result


def generate_bonds_by_cutoff(array, p, pos, L, R):
    Nf = len(array)
    N=len(pos)
    f=int(Nf/N)
    num_bonds = int(Nf * p / 2)
    num_bonds = min(num_bonds, Nf // 2)
    bonds = []
    count=0
    idx=0
    list11=[]

    delta = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
    delta[...,0] = minimum_image(delta[...,0], L)
    delta[...,1] = minimum_image(delta[...,1], L)
    delta[...,2] = minimum_image(delta[...,2], L)
    dist = np.sqrt(np.sum(delta**2, axis=-1))
    
    for ii in range(N):
        for jj in range(ii,N):
            if dist[ii,jj]<R:
                if ii==jj:
                    list11.extend([(ii, jj)] * int(f/2))
                else:
                    list11.extend([(ii, jj)] * f)

    b=sample_with_limit(list11, num_bonds, f)
    
    p_new=2*len(b)/Nf


    return b,p_new


def judge_spanning(N, position, bonds, L, rcut):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(bonds)
    clusters = list(nx.connected_components(G))

    signal=0
    for cluster in clusters:
        if len(cluster)<=1:
            continue
        subgraph = G.subgraph(cluster)  # induced subgraph on this cluster
        nodes = np.array(list(subgraph.nodes()))
        edges = np.array(list(subgraph.edges()))
        
        N1 = len(nodes)
        Lx = L
        Ly = L
        Lz = L

        # shifts along xyz
        shifts = [N, 2*N, 3*N]
        lengths = [Lx, Ly, Lz]

        for dim in range(3):  # 0=x,1=y,2=z
            shift = shifts[dim]
            L1 = lengths[dim]

            G1 = nx.Graph()
            # 1. original bond
            for i, j in edges:
                G1.add_edge(i, j)
            # 2. image bond
            bonds_img = edges + shift
            for i, j in bonds_img:
                G1.add_edge(i, j)
            # 3. cross boundary bond (+ axis)
            for i, j in edges:
                if abs(position[i, dim]-position[j, dim]) >=L1-rcut:
                    G1.add_edge(i, j+shift)
                    G1.add_edge(i+shift, j)
                    G1.remove_edge(i, j)
                    G1.remove_edge(i+shift, j+shift)
            
            start_node = np.random.choice(nodes)
            try:
                path = nx.shortest_path(G1, source=start_node, target=start_node+shift)
                # print("find spanning path")
                # print(start_node,start_node+shift)
                # print(cluster)
                # print(N1)
                # print(nodes)
                # print(G1.edges())
                # print(path)
                signal=1
                return signal
            except nx.NetworkXNoPath:
                # print("no path")
                continue
    return signal

def build_graph_and_get_cluster_sizes(N, bonds):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(bonds)
    clusters = list(nx.connected_components(G))
    cluster_sizes = [len(c) for c in clusters]

    return cluster_sizes

def count_loops_from_bonds(N, bonds):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(bonds)  # nx.Graph removes multiple edges
    
    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G)
    L_total = E - V + C

    L_total=L_total+len(bonds)-G.number_of_edges()
    
    return L_total

def sample_positions(N, L):
    # uniform in [0, L)
    return rng.random((N, 3)) * L

def minimum_image(d, L):
    return d - np.round(d/L)*L


def main():
    max_cluster_size = N
    cluster_hist_accumulate = np.zeros(max_cluster_size + 1)  # Index = cluster size

    # print(N,f)
    maxsize=[]
    loop_num=[]

    chi_sum = []               # susceptibility-like moment sum over repeats
    sizes_arr = np.arange(1, max_cluster_size + 1)  # reused each repeat
    p_gel=[]
    intra_binding=[]
    rep=0
    spanningcl=[]
    while rep<n_repeat:
        array = generate_shuffled_array(N, f)
        pos=sample_positions(N, L_final)
        bond_list,p_new = generate_bonds_by_cutoff(array, p, pos, L_final,r_detect)
        if p_new<int(N*f * p / 2)*2/(N*f):
            print("can not find the pair number to meet the p you give!")
            continue
        else:
            rep+=1
        cluster_sizes = build_graph_and_get_cluster_sizes(N, bond_list)
        cll=judge_spanning(N, pos, bond_list, L_final, r_detect)
        spanningcl.append(cll)
        
        n_intra=0
        for k in range(len(bond_list)):
            if bond_list[k][0]==bond_list[k][1]:
                n_intra+=1
        intra_binding.append(n_intra)
        
        bonds_no_self = [(u, v) for u, v in bond_list if u != v]

        L_total = count_loops_from_bonds(N,bonds_no_self)
        
        loop_num.append(n_intra+L_total)

        maxsize.append(max(cluster_sizes))

        # Accumulate histogram
        hist, _ = np.histogram(cluster_sizes, bins=np.arange(1, max_cluster_size + 2))
        hist_norm = hist / max_cluster_size
        cluster_hist_accumulate[1:len(hist)+1] += hist_norm

        smax = max(cluster_sizes)
        kmax=hist_norm[smax - 1]

        denom = np.sum(sizes_arr * hist_norm) - kmax * smax
        if denom > 0:
            m22=np.sum((sizes_arr**2) * hist_norm) - kmax * (smax**2)
            chi_run = m22 / denom
        else:
            chi_run = 0.0

        chi_sum.append(chi_run)
        p_gel.append(1-denom)

    maxsize_ave=np.mean(maxsize)
    spanningcl_ave=np.mean(spanningcl)
    intra_binding_ave=np.mean(intra_binding)
    intra_binding_std=np.std(intra_binding)
    loop_num_ave=np.mean(loop_num)
    loop_num_std=np.std(loop_num)
    chi_avg = np.mean(chi_sum)
    p_gel_ave=np.mean(p_gel)
    # Average
    avg_hist = cluster_hist_accumulate / n_repeat
    sizes = np.arange(1, max_cluster_size + 1)
    sizes = sizes/max_cluster_size
    nonzero = avg_hist[1:] > 0

    # cc=N*65/(L_final*L_final*L_final)
    # output_filename = "f_"+str(f)+"_N_"+str(N)+"_L_"+str(L_final)+"_r_"+str(r_detect)+"_max_cluster_intra_loopnum_m2_vs_p.txt"
    # # max_cluster = max(cluster_sizes)
    # with open(output_filename, "a") as f1:
    #     f1.write(f"{p:.6f}\t{spanningcl_ave:.6f}\t{maxsize_ave:.6f}\t{maxsize_ave/N:.6f}\t{intra_binding_ave:.6f}\t{intra_binding_std:.6f}\t{loop_num_ave:.6f}\t{loop_num_std:.6f}\t{chi_avg:.6f}\t{p_gel_ave:.6f}\n")

    cc=N*65/(L_final*L_final*L_final)
    output_filename = "phase_diagram_f_"+str(f)+"_L_"+str(L_final)+"_r_"+str(r_detect)+"_max_cluster_intra_loopnum_m2_vs_p_2.txt"
    with open(output_filename, "a") as f1:
        f1.write(f"{cc:.6f}\t{N:.0f}\t{p:.6f}\t{spanningcl_ave:.6f}\t{maxsize_ave:.6f}\t{maxsize_ave/N:.6f}\t{intra_binding_ave:.6f}\t{intra_binding_std:.6f}\t{loop_num_ave:.6f}\t{loop_num_std:.6f}\t{chi_avg:.6f}\t{p_gel_ave:.6f}\n")
        
    output_filename = f"./cl_dis/cluster_size_distribution_N{N}_f{f}_p{p_new:.3f}_L{L_final:.3f}_r{r_detect:.3f}_.txt"
    with open(output_filename, "w") as f2:
        f2.write("# cluster_size\tavg_count\n")
        for s, c in zip(sizes, avg_hist[1:]):
            if c > 0:  # nonzero parts
                f2.write(f"{s}\t{c:.6f}\n")
    print("Average cluster size distribution over", n_repeat, "runs.")
    print("Sizes:", sizes[nonzero])
    print("Avg counts:", avg_hist[1:][nonzero])
    
    # Plot
    # plt.bar(sizes, avg_hist[1:], align='center', edgecolor='black')
    # plt.xlabel("Cluster size")
    # plt.ylabel("Average count")
    # plt.title(f"Average cluster size distribution\n(N={N}, f={f}, p={p}, repeats={n_repeat})")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()

