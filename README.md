# Hoomd_RG_RGG
MD simulation for percolation network and Random (Geometric) Graph methods

This repository contains molecular-dynamics simulations and theoretical/network
models for gelation and cluster formation in reversible associative polymer
systems. Two architectures are considered:

- **Homoassociative system:** one polymer species with multiple self-associating
  stickers.
- **Heteroassociative system:** RNA-like chains and trimer-like chains with
  complementary stickers.

The repository connects four levels of description:

1. Molecular-dynamics simulations generate particle trajectories.
2. Trajectory-analysis scripts extract binding, loops, spanning clusters, and
   cluster-size distributions.
3. Random graph (RG) and random geometric graph (RGG) models reproduce network
   formation using measured binding probabilities.
4. Flory–Stockmayer (FS) theory predicts finite-cluster distributions and giant
   cluster fractions.

## Repository structure

```text
code/
├── FS_theory/
│   ├── new_graph.py
│   ├── new_graph_2_2component_FS.py
│   └── 1_final_epsilon1_binding_information_.txt
├── RG/
│   ├── graph_sim_new.py
│   ├── graph_sim_new_2.py
│   └── 1_final_epsilon1_binding_information_.txt
├── RGG/
│   ├── graph_sim_new.py
│   ├── graph_sim_new_map_sim.py
│   └── 1_final_epsilon1_binding_information_.txt
└── simulation/
    ├── homo/
    │   ├── test_v4_1.py
    │   ├── test1.py
    │   ├── test2.py
    │   └── a.sh
    └── hetero/
        ├── simulation3.py
        ├── test1.py
        ├── test2.py
        └── a.sh
```


## Model overview

### Homoassociative system

The default model contains linear chains of length 65 with 10 regularly spaced
binding domains. The simulation uses:

- WCA excluded-volume interactions;
- FENE-WCA bonded interactions;
- a short-range tabulated attraction between compatible sticker beads;
- Langevin dynamics at reduced temperature \(k_\mathrm{B}T=1\).

The attraction strength is controlled by `eps1`.

### Heteroassociative system

The default two-component model contains:

- RNA-like chains: 65 beads and 10 stickers per chain;
- trimer-like chains: 23 beads and 3 stickers per chain;
- only complementary RNA–trimer sticker attraction;
- WCA repulsion, harmonic bonds, and a harmonic bending potential.

The production simulation uses a fixed sticker attraction
\(\epsilon_{\mathrm{sp}}/k_\mathrm{B}T=6\). The command-line parameter `kd`
controls the bending stiffness.

## Requirements

The theoretical and graph-model scripts require:

```text
Python 3
NumPy
SciPy
Matplotlib
NetworkX
```

Trajectory analysis additionally requires:

```text
gsd
freud-analysis
pandas
```

The two simulation folders use different generations of the HOOMD-blue API:

- `simulation/homo/test_v4_1.py` uses the modern `hoomd.Simulation` API
  (HOOMD-blue 4-style code).
- `simulation/hetero/simulation3.py` uses the legacy
  `hoomd.context.initialize` API (HOOMD-blue 2.x).

Use separate environments for these two simulations. A single HOOMD-blue
installation will generally not run both scripts unchanged.

A basic analysis/theory environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib networkx gsd freud-analysis pandas
```

## Input data shared by theory and graph models

The file

```text
1_final_epsilon1_binding_information_.txt
```

contains simulation-derived state points. The two-component FS, RG, and RGG
scripts read zero-based columns 2, 3, and 10:

```text
column 2 -> N1
column 3 -> N2
column 10 -> p1
```

These scripts currently select only rows satisfying `N1 == 250`. Change the
corresponding Boolean filter in each script to analyze other compositions.
Run each script from its own directory so that this input file is found.

## Flory–Stockmayer theory

### Single-component theory

`FS_theory/new_graph.py` evaluates the single-component FS distribution
\(n_s\) in log-gamma form to avoid factorial overflow:

```bash
cd FS_theory
python new_graph.py
```

The default parameters are:

```text
f = 10
s = 1, ..., 2000
p = 0.01, 0.06, 0.11, 0.21, 0.47, 0.85
```

The script plots \(n_s\) against the normalized cluster size \(s/2000\) and
writes all curves to:

```text
n_vs_N_f10_all_p.txt
```

### Two-component theory

`FS_theory/new_graph_2_2component_FS.py` evaluates the Stockmayer distribution
\(N_{m,n}\), sums it at fixed total cluster size \(s=m+n\), and calculates the
giant-cluster fractions of both species using fixed-point equations.

```bash
cd FS_theory
mkdir -p cl_dis_1
python new_graph_2_2component_FS.py
```

Default functionalities are \(f_A=10\) and \(f_B=3\). Bond conservation sets

\[
p_B = \frac{p_A f_A N_A}{f_B N_B}.
\]

Each selected composition produces a separate cluster-distribution file in
`cl_dis_1/`.

## Random graph models

The RG models ignore spatial positions. Stickers are randomly paired subject
to the prescribed conversion and functionality.

### Single-component RG

`RG/graph_sim_new.py` takes four positional arguments:

```bash
cd RG
python graph_sim_new.py N f p n_repeat
```

Example:

```bash
python graph_sim_new.py 2000 10 0.47 100
```

The script measures:

- largest-cluster size and fraction;
- intrachain/self bonds;
- total cycle rank, including parallel edges;
- susceptibility-like finite-cluster moment;
- estimated giant-cluster fraction;
- average cluster-size distribution.

Summary data are appended to:

```text
f_<f>_max_cluster_intra_loopnum_m2_vs_p.txt
```

A separate distribution is written as:

```text
cluster_size_distribution_N<N>_f<f>_p<p>_.txt
```

### Two-component RG

`RG/graph_sim_new_2.py` creates only heterotypic edges between species 1 and 2.
The state points and \(p_1\) values are read from the shared simulation table.

```bash
cd RG
python graph_sim_new_2.py f1 f2 n_repeat
```

Example:

```bash
python graph_sim_new_2.py 10 3 100
```

The script reports the largest-cluster fraction, loop/cycle statistics,
overlapping parallel bonds, and a cluster-size distribution for every selected
composition.

## Random geometric graph models

The RGG models place chain-level vertices uniformly in a periodic cubic box.
Edges can form only between vertices separated by less than `r_detect`, after
which bonds are sampled to match the requested conversion without exceeding
the vertex functionality.

The scripts also test whether a connected component spans a periodic direction.

### Single-component RGG

```bash
cd RGG
mkdir -p cl_dis
python graph_sim_new.py N f p n_repeat L r_detect
```

Example:

```bash
python graph_sim_new.py 2000 10 0.47 100 80 5
```

The chain-monomer concentration used in the summary is

\[
c = \frac{65N}{L^3}.
\]

The output includes concentration, imposed conversion, spanning probability,
largest-cluster fraction, self-bond and cycle statistics, susceptibility, and
giant-cluster fraction.

### Two-component RGG mapped to simulation

`RGG/graph_sim_new_map_sim.py` reads \(N_1\), \(N_2\), and \(p_1\) from the
shared simulation table and generates only heterotypic geometric edges:

```bash
cd RGG
mkdir -p cl_dis
python graph_sim_new_map_sim.py f1 f2 n_repeat L r_detect
```

Example:

```bash
python graph_sim_new_map_sim.py 10 3 100 80 5
```

The two concentrations are calculated as

\[
c_1=\frac{65N_1}{L^3},
\qquad
c_2=\frac{23N_2}{L^3}.
\]

If the geometric candidate pool is too small to realize the requested
conversion, that state point is skipped after repeated failed attempts.

## Molecular-dynamics simulations

### Homoassociative simulation

Run:

```bash
cd simulation/homo
python test_v4_1.py sample n_polymer chain_length n_stickers box_size epsilon_sp
```

Example:

```bash
python test_v4_1.py 1 2000 65 10 80 9.0
```

The simulation first relaxes the initial configuration, compresses it into the
target cubic box, activates the sticker attraction, and performs the production
run. It writes:

```text
log_cnf_<parameters>.txt
trajectory_cnf_<parameters>.gsd
```

`simulation/homo/a.sh` provides a parameter-sweep template. Before using it,
correct the script name:

```text
teset_v4_1.py  ->  test_v4_1.py
```

### Heteroassociative simulation

Run:

```bash
cd simulation/hetero
mkdir -p kd_<kd>
python simulation3.py sample Yytrimer Yyrna Ly_final kd
```

Example:

```bash
mkdir -p kd_5
python simulation3.py 1 10 10 80 5
```

The actual numbers of chains are:

```text
N_trimer = 10 * Yytrimer
N_RNA    = 10 * Yyrna
```

The simulation initializes the two species in separate regions, equilibrates
and compresses the system, activates complementary sticker attraction, and
runs production dynamics. Logs and trajectories are written to `kd_<kd>/`.

The supplied `a.sh` sweeps chain numbers and `kd`, but the corresponding
`kd_<kd>/` output directories must exist before the jobs start.

## Trajectory analysis

The analysis scripts infer sticker bonds using a distance cutoff of
\(0.5\sigma\), combine these contacts with permanent backbone bonds, and then
construct chain-level clusters.

### Cluster-size distributions: `test1.py`

Homoassociative:

```bash
cd simulation/homo
mkdir -p cl_dis_1
python test1.py sample n_polymer chain_length n_stickers box_size epsilon_sp
```

Heteroassociative:

```bash
cd simulation/hetero/kd_<kd>
mkdir -p cl_dis_1
python ../test1.py sample Yytrimer Yyrna Ly_final kd
```

The output has three columns:

```text
cluster size
cluster size / total number of chains
time-averaged cluster count
```

The homo script discards the first 50 frames. The hetero script discards the
first 80 frames and additionally normalizes the average counts by the total
number of chains.

### Binding and topology analysis: `test2.py`

Homoassociative:

```bash
python test2.py sample n_polymer chain_length n_stickers box_size epsilon_sp
```

The output columns are:

```text
time
binding probability
number of sticker pairs
mean number of degree-one nodes
cycle count including intrachain bonds
number of edges removed to break cycles
flag for sticker aggregates larger than a pair
number of intrachain bonds
number of interchain bonds
spanning-cluster indicator
```

Heteroassociative:

```bash
cd simulation/hetero/kd_<kd>
python ../test2.py sample Yytrimer Yyrna Ly_final kd
```

The output columns are:

```text
time
RNA-side binding probability
trimer-side binding probability
number of sticker pairs
number of degree-one nodes
number of edges removed to break cycles
cycle count
number of bridge bonds
number of local loop bonds
number of singly bound trimers
spanning-cluster indicator
largest-cluster fraction
```

## Typical workflow

1. Run a homo- or heteroassociative molecular-dynamics simulation.
2. Analyze the GSD trajectory with `test1.py` and `test2.py`.
3. Average the measured binding data across trajectories and assemble
   `1_final_epsilon1_binding_information_.txt`.
4. Run the FS, RG, and RGG models using the same \(N_i\), \(f_i\), and \(p_i\).
5. Compare cluster-size distributions, giant-cluster fractions, loop
   statistics, and percolation boundaries across the four descriptions.

## Important implementation notes

- Run scripts from their own directories because several input and output paths
  are relative.
- Create `cl_dis/`, `cl_dis_1/`, and `kd_<kd>/` directories before running
  scripts that write into them.
- Most summary files are opened in append mode. Remove or rename old output
  files before starting a statistically independent sweep if duplicate rows
  are not desired.
- Random seeds are not exposed consistently through the command line, so exact
  numerical repetition requires adding fixed seeds to NumPy, Python `random`,
  and HOOMD-blue.
- The RGG distance matrices scale quadratically with the number of vertices and
  can require substantial memory for large systems.
- `FS_theory/new_graph.py` contains an unused direct-factorial helper that
  references `math` without importing it. The production calculation uses the
  stable `gammaln` implementation and is unaffected.

## Citation

If these scripts contribute to published work, cite the associated article and
the software repository. Add the final citation and DOI here when available.

## License

No license file is included in the supplied archive. Add a `LICENSE` file before
public release so that reuse and redistribution terms are explicit.
