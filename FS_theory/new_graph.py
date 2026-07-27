# Flory–Stockmayer theory to get cluster size distribution and giant fraction for single component
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

def n_theory_log(p, N, f):
    if p <= 0 or p >= 1:
        return 0.0  # avoid log(0)
    log_num = gammaln((f - 1) * N + 1)
    log_den = gammaln(N + 1) + gammaln((f - 2) * N + 3)
    log_prob = (N - 1) * np.log(p) + ((f - 2) * N + 2) * np.log(1 - p)
    log_result = np.log(f) + log_num - log_den + log_prob
    return np.exp(log_result)

def n_theory(p, N, f):
    numerator = math.factorial((f - 1) * N)
    denominator = math.factorial(N) * math.factorial((f - 2) * N + 2)
    prob_term = (p ** (N - 1)) * ((1 - p) ** ((f - 2) * N + 2))
    result = f * (numerator / denominator) * prob_term
    return result

# ========== main ==========
def main():
    f = 10  # sticker #
    N_vals = np.arange(1, 2001)
    # p_list = np.round(np.linspace(0.1, 1.0, 10), 2)

    p_list = [0.01, 0.06, 0.11, 0.21, 0.47, 0.85]

    plt.figure(figsize=(10, 6))

    for p in p_list:
        n_vals = np.array([n_theory_log(p, N, f) for N in N_vals])
        plt.loglog(N_vals / 2000, n_vals, label=f"p={p:.2f}")


    # plot figure
    plt.xlabel("N / 2000 (log scale)")
    plt.ylabel("n(p, N) (log scale)")
    plt.title(f"Log-Log Plot: n(p, N) vs N (f={f})")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # file name
    out_file = f"n_vs_N_f{f}_all_p.txt"

    # header
    header_parts = []
    for p in p_list:
        header_parts.append(f"N_scaled_p{p:.2f}")
        header_parts.append(f"n(p={p:.2f})")
    header_line = "\t".join(header_parts)

    all_data = []
    for i, N in enumerate(N_vals):
        row_parts = []
        for p in p_list:
            N_scaled = N / 2000
            n_val = n_theory_log(p, N, f)
            row_parts.append(f"{N_scaled:.6f}")
            row_parts.append(f"{n_val:.6e}")
        all_data.append("\t".join(row_parts))

    with open(out_file, "w") as fout:
        fout.write(header_line + "\n")
        fout.write("\n".join(all_data))


if __name__ == "__main__":
    main()
