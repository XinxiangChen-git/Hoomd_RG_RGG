# single component with Random Graph method
import numpy as np
import sys
import networkx as nx
import random
import matplotlib.pyplot as plt


# Parameters
N = int(sys.argv[1])             # Number of chains
f = int(sys.argv[2])             # Binding sites per chain
p = float(sys.argv[3])           # Binding probability
n_repeat = int(sys.argv[4])      # How many times to repeat

def generate_shuffled_array(N, f):
    array = np.repeat(np.arange(N), f)
    np.random.shuffle(array)
    return array

def generate_bonds_by_order(array, p):
    Nf = len(array)
    num_bonds = int(Nf * p / 2)
    num_bonds = min(num_bonds, Nf // 2)
    bonds = []

    for i in range(num_bonds):
        idx = 2 * i
        if idx + 1 < Nf:
            a = array[idx]
            b = array[idx + 1]
            # if a != b:  # Avoid intra-chain bonding
            bonds.append((a, b))

    return bonds

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
    G.add_edges_from(bonds)  # nx.Graph will remove multiple edges

    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G)
    L_total = E - V + C

    L_total=L_total+len(bonds)-G.number_of_edges() ## add multiple edges back

    return L_total

def main():
    max_cluster_size = N
    cluster_hist_accumulate = np.zeros(max_cluster_size + 1)  # Index = cluster size

    print(N,f)
    maxsize=[]
    loop_num=[]

    chi_sum = []               # susceptibility-like moment sum over repeats
    sizes_arr = np.arange(1, max_cluster_size + 1)  # reused each repeat
    p_gel=[]
    intra_binding=[]
    for rep in range(n_repeat):
        array = generate_shuffled_array(N, f)
        bond_list = generate_bonds_by_order(array, p)
        cluster_sizes = build_graph_and_get_cluster_sizes(N, bond_list)
        
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

    output_filename = "f_"+str(f)+"_max_cluster_intra_loopnum_m2_vs_p.txt"
    with open(output_filename, "a") as f1:
        f1.write(f"{p:.6f}\t{maxsize_ave:.6f}\t{maxsize_ave/N:.6f}\t{intra_binding_ave:.6f}\t{intra_binding_std:.6f}\t{loop_num_ave:.6f}\t{loop_num_std:.6f}\t{chi_avg:.6f}\t{p_gel_ave:.6f}\n")

    output_filename = f"cluster_size_distribution_N{N}_f{f}_p{p:.3f}_.txt"
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


