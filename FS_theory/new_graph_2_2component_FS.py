# Flory–Stockmayer theory to get cluster size distribution and giant fraction for two components

import math
import numpy as np
import matplotlib.pyplot as plt

def N_mn(m, n, fA, fB, pA, pB, NA, NB):
    """Compute N_{m,n} from Stockmayer (using log factorials for stability)."""
    if m <= 0 or n <= 0 or m > NA or n > NB:
        return 0.0
    if (fA*m - m - n + 1) < 0 or (fB*n - n - m + 1) < 0:
        return 0.0

    # parameter
    x = (pB * (1 - pA)**(fA - 1)) / (1 - pB)
    y = (pA * (1 - pB)**(fB - 1)) / (1 - pA)
    K = (fA * NA) * ((1 - pA) * (1 - pB)) / pB

    if x <= 0 or y <= 0 or K <= 0:
        return 0.0

    # log-space avoid NAN
    log_num = math.lgamma(fA*m - m + 1) + math.lgamma(fB*n - n + 1)
    log_den = (
        math.lgamma(fA*m - m - n + 2)
        + math.lgamma(fB*n - n - m + 2)
        + math.lgamma(m + 1)
        + math.lgamma(n + 1)
    )

    log_val = math.log(K) + m*math.log(x) + n*math.log(y) + (log_num - log_den)

    try:
        val = math.exp(log_val)
    except OverflowError:
        val = 0.0
    return val

def cluster_distribution(fA, fB, pA, NA, NB, smax=None):
    """Compute n_s = sum_{m+n=s} N_{m,n}, including s=1 monomers."""
    pB = pA * fA * NA / (fB * NB)
    if smax is None:
        smax = int(NA + NB)

    ns = np.zeros(smax + 1)

    # --- s=1: isolated monomers (no bonds) ---
    ns[1] = NA * (1 - pA)**fA + NB * (1 - pB)**fB

    # --- s>=2: Stockmayer trees ---
    for s in range(2, smax + 1):
        total = 0.0
        for m in range(1, min(s-1, int(NA)) + 1):
            n = s - m
            if n <= 0 or n > NB:
                continue
            total += N_mn(m, n, fA, fB, pA, pB, NA, NB)
        ns[s] = total

    return np.arange(smax + 1), ns, pB


def giant_cluster_fraction(fA, fB, pA, NA, NB, tol=1e-12, max_iter=10000):
    # Initialize guesses near 1 (no percolation)
     # Stoichiometric constraint
    pB = pA * (fA * NA) / (fB * NB)
    if not (0.0 <= pB <= 1.0):
        raise ValueError(f"Stoichiometric pB={pB:.6f} not in [0,1]. "
                         f"Adjust pA or NA/NB/fA/fB.")

    # Fixed-point iteration for uA, uB
    uA, uB = 0.0, 0.0  # start subcritical side
    for _ in range(max_iter):
        uA_new = ((1.0 - pA) + pA * uB) ** (fA - 1)
        uB_new = ((1.0 - pB) + pB * uA) ** (fB - 1)
        if abs(uA_new - uA) < tol and abs(uB_new - uB) < tol:
            uA, uB = uA_new, uB_new
            break
        uA, uB = uA_new, uB_new
    else:
        print("Warning: fixed-point iteration did not converge (near critical?).")

    # Giant fractions for each species (USE uA for A→B, uB for B→A!)
    SA = 1.0 - ((1.0 - pA) + pA * uB) ** fA
    SB = 1.0 - ((1.0 - pB) + pB * uA) ** fB
    S  = (NA * SA + NB * SB) / (NA + NB)

    return uA, uB, SA, SB, S

if __name__ == "__main__":
    fA, fB = 10, 3

    data = np.loadtxt("1_final_epsilon1_binding_information_.txt", usecols=(2, 3, 10)) # get N, p from simulation resluts
    
    values_list = data.tolist()
    
    values_list=np.array(values_list)

    index=values_list[:,0]==250
    values_list=values_list[:,:][index]
    
    NA_list=values_list[:,0]
    NB_list=values_list[:,1]
    pA_list=values_list[:,2]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(pA_list)))
    # plt.figure(figsize=(8,6))

    for NA, NB, pA, c in zip(NA_list, NB_list, pA_list, colors):

        Smax=NA+NB

        s, ns, pB = cluster_distribution(fA, fB, pA, NA, NB)

        s=s/Smax
        ns=ns/Smax

        uA, uB, GA, GB, G = giant_cluster_fraction(fA, fB, pA, NA, NB)

        print(NA, NB, pA, G)

        # save file
        fname = f"./cl_dis_1/cluster_size_distribution_N1_{NA}_N2_{NB}_f1_{fA}_f2_{fB}_p1_{pA}.txt"
        np.savetxt(fname, np.column_stack((s, ns)), fmt="%.6e", header=f"s   n_s")
        print(f"Data saved to {fname}")

        # plot figure
        # mask = (ns > 0)
        # plt.figure(figsize=(6,4))
        # plt.loglog(s[mask]/maxcl, ns[mask], 'o-', ms=3)
        # plt.xlabel("Cluster size s = m + n")
        # plt.ylabel("n_s")
        # plt.title(f"fA={fA}, fB={fB}, pA={pA}, pB={pB:.3f}")
        # plt.ylim(1e-5, 2e0)
        # # plt.xlim(1e0, 6e2)
        # plt.tight_layout()
        # plt.show()
