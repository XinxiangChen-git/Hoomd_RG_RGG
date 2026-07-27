# two components in cubic box with Random Geometric Graph method
import numpy as np
import sys
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import Counter


# Parameters
f1 = int(sys.argv[1])             # Binding sites per chain
f2 = int(sys.argv[2])             # Binding sites per chain
n_repeat = int(sys.argv[3])      # How many times to repeat
L_final=float(sys.argv[4])
r_detect=float(sys.argv[5])

rng = np.random.default_rng()

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

def sample_with_limit(bond_list, N, f1, f2, seed=None):
    if seed is not None:
        random.seed(seed)

    result = []
    counts = Counter()

    # scramble original list
    candidates = bond_list[:]
    random.shuffle(candidates)

    for x in candidates:  # bond lists，like (1, 2)
        i, j = x
        if counts[i] < f1 and counts[j] < f2:  # each vertex contain < fi edges
            result.append(x)
            counts[i] += 1
            counts[j] += 1
            if len(result) >= N:
                break

    if len(result) < N:
        indd=1
    else:
        indd=0

    return result,indd


def generate_bonds_by_cutoff(array1,array2, N1, f1, N2, f2, p1, pos1, pos2, L, R):
    Nf = N1*f1
    num_bonds = min(min(int(Nf * p1), N2*f2), N1*f1)

    delta = pos1[:, None, :] - pos2[None, :, :]  # (N, N, 3)
    delta[...,0] = minimum_image(delta[...,0], L)
    delta[...,1] = minimum_image(delta[...,1], L)
    delta[...,2] = minimum_image(delta[...,2], L)
    dist = np.sqrt(np.sum(delta**2, axis=-1))


    list12=[]
    mask = dist < R
    ii, jj = np.where(mask)
    pairs = np.column_stack((ii, jj + N1))
    pairs_repeated = np.repeat(pairs, min(f1, f2), axis=0)
    list12.extend(map(tuple, pairs_repeated))


    bonds,indd=sample_with_limit(list12, num_bonds, f1,f2)
    
    p_new=len(bonds)/Nf

    return bonds,p_new

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
                continue
    return signal


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
    G.add_edges_from(bonds)  # nx.Graph removes multiple edges

    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G)
    L_total = E - V + C

    intra=len(bonds)-E # add overlapping edges back

    return L_total, intra


def sample_positions(N, L):
    # uniform in [0, L)
    return rng.random((N, 3)) * L

def minimum_image(d, L):
    return d - np.round(d/L)*L


def main():

    # arr11 = generate_feasible_points(f1, f2, Ntot, 40, 40)
    data = np.loadtxt("1_final_epsilon1_binding_information_.txt", usecols=(2, 3, 10)) # get N, p from simulation resluts
    
    values_list = data.tolist()
    
    values_list=np.array(values_list)

    index=values_list[:,0]==250
    values_list=values_list[:,:][index]

    for ii in range(len(values_list)):
        N1=int(values_list[ii][0])
        N2=int(values_list[ii][1])
        p1=values_list[ii][2]
        max_cluster_size = N1+N2
        cluster_hist_accumulate = np.zeros(max_cluster_size + 1)  # Index = cluster size

        maxsize=[]
        loop_num=[]
        overlapping_bond=[]
        rep=0
        rep_fail=0
        spanningcl=[]
        while rep<n_repeat:
            array1,array2 = generate_shuffled_array(N1, f1, N2, f2)
            pos1=sample_positions(N1, L_final)
            pos2=sample_positions(N2, L_final)
            bond_list,p_new = generate_bonds_by_cutoff(array1,array2, N1, f1, N2, f2, p1,pos1,pos2,L_final,r_detect)
            if p_new<int(N1*f1 * p1)/(N1*f1):
                rep_fail+=1
                if rep_fail<n_repeat:
                    continue
                else:
                    print(N1,N2,p1,"always can not find the pair number to meet the p you give!")
                    p1_real=p1
                    p2_real=N1*f1*p1/(N2*f2)
                    break
            else:
                rep+=1
            pos=np.append(pos1,pos2,axis=0)
            
            N_tot=N1+N2
            cluster_sizes = build_graph_and_get_cluster_sizes(N1, N2, bond_list)
            cll=judge_spanning(N_tot, pos, bond_list, L_final, r_detect)
            spanningcl.append(cll)

            L_total, intra = count_loops_from_bonds(N1, N2, bond_list)
            p1_real=len(bond_list)/(N1*f1)
            p2_real=len(bond_list)/(N2*f2)
            
            loop_num.append(L_total)

            maxsize.append(max(cluster_sizes))

            overlapping_bond.append(intra)

            # Accumulate histogram
            hist, _ = np.histogram(cluster_sizes, bins=np.arange(1, max_cluster_size + 2))
            hist_norm = hist / max_cluster_size
            cluster_hist_accumulate[1:len(hist)+1] += hist_norm

        if rep_fail==n_repeat:
            continue
        maxsize_ave=np.mean(maxsize)
        loop_num_ave=np.mean(loop_num)
        loop_num_std=np.std(loop_num)
        overlapping_bond_ave=np.mean(overlapping_bond)
        overlapping_bond_std=np.std(overlapping_bond)
        spanningcl_ave=np.mean(spanningcl)
        print(N1,N2,f1,f2,(N1-N2)/(N1+N2),p1_real*(f1-1)*p2_real*(f2-1),spanningcl_ave,maxsize_ave/(N1+N2),loop_num_ave,loop_num_std,overlapping_bond_ave,overlapping_bond_std)
        # Average
        avg_hist = cluster_hist_accumulate / n_repeat
        sizes = np.arange(1, max_cluster_size + 1)
        sizes = sizes/max_cluster_size
        nonzero = avg_hist[1:] > 0

        # c1=N1*65/(L_final*L_final*L_final)
        # c2=N2*23/(L_final*L_final*L_final)
        # output_filename = "map_sim_f1_"+str(f1)+"f2_"+str(f2)+"_L_"+str(L_final)+"_r_"+str(r_detect)+"_max_cluster_intra_loop_2.txt"
        # # max_cluster = max(cluster_sizes)
        # with open(output_filename, "a") as fout:
        #     fout.write(f"{N1}\t{N2}\t{c1}\t{c2}\t{f1}\t{f2}\t{(N1-N2)/(N1+N2):.6f}\t{p1_real*(f1-1)*p2_real*(f2-1):.6f}\t{spanningcl_ave:.6f}\t{maxsize_ave/(N1+N2):.6f}\t{loop_num_ave:.6f}\t{loop_num_std:.6f}\t{overlapping_bond_ave:.6f}\t{overlapping_bond_std:.6f}\n")

        c1=N1*65/(L_final*L_final*L_final)
        c2=N2*23/(L_final*L_final*L_final)
        output_filename = "map_sim_phase_diagram_f1_"+str(f1)+"f2_"+str(f2)+"_L_"+str(L_final)+"_r_"+str(r_detect)+"_max_cluster_intra_loopnum_m2_vs_p_2.txt"
        with open(output_filename, "a") as fout:
            fout.write(f"{N1}\t{N2}\t{c1}\t{c2}\t{f1}\t{f2}\t{p1_real:.6f}\t{p2_real:.6f}\t{spanningcl_ave:.6f}\t{maxsize_ave/(N1+N2):.6f}\t{loop_num_ave:.6f}\t{loop_num_std:.6f}\t{overlapping_bond_ave:.6f}\t{overlapping_bond_std:.6f}\n")

        output_filename = f"./cl_dis/cluster_size_distribution_Ly_{L_final}_N1_{N1}_N2_{N2}_f1_{f1}_f2_{f2}_p1_{p1:.3f}.txt"
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
