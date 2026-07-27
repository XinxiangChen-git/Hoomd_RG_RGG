# two components with Random Graph method
import numpy as np
import sys
import networkx as nx
import random
import matplotlib.pyplot as plt


# Parameters
f1 = int(sys.argv[1])             # Binding sites per chain
f2 = int(sys.argv[2])             # Binding sites per chain
n_repeat = int(sys.argv[3])      # How many times to repeat

def y_max_boundary(x, f1, f2):
    a = f2 / f1
    K = (f1 - 1) * (f2 - 1)
    r = ((1 - x) / (1 + x)) * a
    return K * np.minimum(r, 1.0 / r)

def generate_feasible_points(f1, f2, N_tot, nx, ny):
    K = (f1 - 1) * (f2 - 1)
    a = f2 / f1
    points = []
    xs = np.linspace(-0.99, 0.99, nx)
    for x in xs:
        ymax = y_max_boundary(x, f1, f2)
        ys = np.linspace(0.01, ymax, ny)  # avoid y=0
        N1 = N_tot * (1 + x) / 2
        N2 = N_tot * (1 - x) / 2
        for y in ys:
            r = ((1 - x) / (1 + x)) * a
            p1 = np.sqrt((y / K) * r)
            p2 = p1 * N1 * f1 / (N2 * f2) if N2 > 0 else np.nan
            if 0 < p1 <= 1 and 0 < p2 <= 1:
                points.append((int(round(N1)), int(round(N2)), p1))
    return points


# Parameters
def generate_shuffled_array(N1, f1, N2, f2):
    array1 = np.repeat(np.arange(N1), f1)
    np.random.shuffle(array1)
    array2 = np.repeat(np.arange(N2), f2)
    np.random.shuffle(array2)
    return array1,array2

def generate_bonds_by_order(array1, array2, N1, f1, N2, f2, p1):
    Nf = N1*f1
    num_bonds = min(min(int(Nf * p1), N2*f2), N1*f1)
    bonds = []

    for i in range(num_bonds):
        # idx = 2 * i
        a = array1[i]
        b = array2[i]+N1
        bonds.append((a, b))

    return bonds

def build_graph_and_get_cluster_sizes(N1, N2, bonds):
    G = nx.Graph()
    G.add_nodes_from(range(N1+N2))
    G.add_edges_from(bonds)
    clusters = list(nx.connected_components(G))
    cluster_sizes = [len(c) for c in clusters]
    return cluster_sizes

def count_loops_from_bonds(N1, N2, bonds):
    G = nx.Graph()
    G.add_nodes_from(range(N1+N2))
    G.add_edges_from(bonds)  # nx.Graph will remove multiple edges

    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G)
    L_total = E - V + C

    intra=len(bonds)-E ## add multiple edges back

    return L_total, intra

def main():

    # arr11 = generate_feasible_points(f1, f2, 100000, 40, 40)
    data = np.loadtxt("1_final_epsilon1_binding_information_.txt", usecols=(2, 3, 10)) # get N, p from simulation resluts

    values_list = data.tolist()
    values_list=np.array(values_list)

    # find N1=250 case
    index=values_list[:,0]==250
    arr11=values_list[:,:][index]

    for ii in range(len(arr11)):
        N1=int(arr11[ii][0])
        N2=int(arr11[ii][1])
        p1=arr11[ii][2]
        max_cluster_size = N1+N2
        cluster_hist_accumulate = np.zeros(max_cluster_size + 1)  # Index = cluster size

        maxsize=[]
        loop_num=[]
        overlapping_bond=[]
        for rep in range(n_repeat):
            array1,array2 = generate_shuffled_array(N1, f1, N2, f2)
            bond_list = generate_bonds_by_order(array1,array2, N1, f1, N2, f2, p1)
            cluster_sizes = build_graph_and_get_cluster_sizes(N1, N2, bond_list)

            L_total, intra = count_loops_from_bonds(N1, N2, bond_list)
            p1_real=len(bond_list)/(N1*f1)
            p2_real=len(bond_list)/(N2*f2)
            
            loop_num.append(L_total+intra)

            maxsize.append(max(cluster_sizes))

            overlapping_bond.append(intra)

            # Accumulate histogram
            hist, _ = np.histogram(cluster_sizes, bins=np.arange(1, max_cluster_size + 2))
            hist_norm = hist / max_cluster_size
            cluster_hist_accumulate[1:len(hist)+1] += hist_norm

        maxsize_ave=np.mean(maxsize)
        loop_num_ave=np.mean(loop_num)
        loop_num_std=np.std(loop_num)
        overlapping_bond_ave=np.mean(overlapping_bond)
        overlapping_bond_std=np.std(overlapping_bond)
        
        # Average
        avg_hist = cluster_hist_accumulate / n_repeat
        sizes = np.arange(1, max_cluster_size + 1)
        sizes = sizes/max_cluster_size
        nonzero = avg_hist[1:] > 0

        output_filename = "f1_"+str(f1)+"f2_"+str(f2)+"_max_cluster_intra_loop_.txt"
        with open(output_filename, "a") as fout:
            fout.write(f"{N1}\t{N2}\t{f1}\t{f2}\t{(N1-N2)/(N1+N2):.6f}\t{p1_real*(f1-1)*p2_real*(f2-1):.6f}\t{maxsize_ave/(N1+N2):.6f}\t{loop_num_ave:.6f}\t{loop_num_std:.6f}\t{overlapping_bond_ave:.6f}\t{overlapping_bond_std:.6f}\n")

        output_filename = f"cluster_size_distribution_N1_{N1}_N2_{N2}_f1_{f1}_f2_{f2}_p1_{p1:.3f}.txt"
        with open(output_filename, "w") as fout:
            fout.write("# cluster_size\tavg_count\n")
            for s, c in zip(sizes, avg_hist[1:]):
                if c > 0:  # nonzero parts
                    fout.write(f"{s}\t{c:.6f}\n")
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


