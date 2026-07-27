# calculate cluster size distribution for simulation

import gsd
import gsd.hoomd
import time
import numpy as np
import random 
import math
import sys
import freud

sample=sys.argv[1]
Yytrimer = int(sys.argv[2])
Yyrna = int(sys.argv[3])
Ly_final = float(sys.argv[4])
kd= float(sys.argv[5])

N_trimer1 = 10        # trimers in x direction
N_trimer2 = Yytrimer  # trimers in y direction
trimer_len = 23       # Monomers in z direction
trimer_tot = N_trimer1*N_trimer2*trimer_len # total monomers of trimers

N_sticky_each_trimer_chian = 3

sticky_trimer_tot=N_trimer1*N_trimer2*N_sticky_each_trimer_chian

N_rna1=10             # rna chains in x direction
N_rna2=Yyrna             # rna chains in y direction
rna_len =65           # length of rna chain
rna_tot = N_rna1*N_rna2*rna_len #total rna monomers

N_sticky_each_rna_chian = int((rna_len-rna_len%(5+1))/(5+1))

sticky_rna_tot=N_rna1*N_rna2*N_sticky_each_rna_chian

total_chains = N_trimer1 * N_trimer2 + N_rna1 * N_rna2


filename1="trajectory_cnf_"+str(sample)+"_kd_"+str(kd)+"_numTrimers_"+str(int(N_trimer1*N_trimer2))+"_numRNA_"+str(int(N_rna1*N_rna2))+"_Ly_"+str(Ly_final)+"_2.gsd"

traj = gsd.hoomd.open(filename1, 'r')

bondarray  = traj[0].bonds.group

N_tot=trimer_tot+rna_tot

Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
box=freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
box1=np.asarray([Lx,Ly,Lz])
pp=0
maxn=0
maxn_rna=0
maxn_trimer=0

bins = np.arange(1, total_chains + 2)  # Bins: 1 to total_chains
hist_accum = np.zeros(len(bins) - 1)
frame_count = 0

for i in range(len(traj)):
	if i<80:
		continue
	points =traj[i].particles.position
	cluster = freud.cluster.Cluster()
	cluster.compute((box,points),neighbors={"r_max": 0.5})
	r=cluster.cluster_idx
	b=cluster.cluster_keys
	cl_props = freud.cluster.ClusterProperties()
	cl_props.compute((box,points), cluster.cluster_idx)
	n=cl_props.sizes  ###n[i] will return the value of the size of each cluster
	cl2index=n==2
	keys=np.asarray(b,dtype=object)
	keyswith2=np.asarray(keys[cl2index].tolist())

	bonds_array = np.copy(bondarray)
	if len(keyswith2)>0:
		bonds_array  = np.append(bondarray,keyswith2,axis=0)  
	bonds_array=np.sort(bonds_array,axis=1)    
	bonds_array=np.unique(bonds_array,axis=0)      

	system = freud.AABBQuery.from_system((box,points))

	dr_array = box.wrap(points[bonds_array[:, 1]] - points[bonds_array[:, 0]])
	neighbors = freud.locality.NeighborList.from_arrays(len(points),len(points),bonds_array[:, 0],bonds_array[:, 1],dr_array)

	cl = freud.cluster.Cluster()
	cl1=cl.compute(system=system, neighbors=neighbors)

	cl_props = freud.cluster.ClusterProperties()
	cl_props.compute(system, cl.cluster_idx)
	n=cl_props.sizes
	b=cl1.cluster_keys
	
	new_cl_list=[]
	cl_size_list=[]
	for kk in range(len(b)):
		cl_list0=[]
		for ll in range(len(b[kk])):
			if b[kk][ll]<trimer_tot:
				head=b[kk][ll]-b[kk][ll]%trimer_len
				cl_list0.append(head)
			else:
				head=b[kk][ll]-(b[kk][ll]-trimer_tot)%rna_len
				cl_list0.append(head)
		cl_list0=np.unique(cl_list0)
		new_cl_list.append(cl_list0)
		cl_size_list.append(len(cl_list0))
	hist, _ = np.histogram(cl_size_list, bins=bins)
	hist_accum += hist
	frame_count += 1

avg_hist = hist_accum / frame_count
avg_hist = avg_hist/ total_chains
normalized_cluster_size = bins[:-1] / total_chains


filename2="./cl_dis_1/"+str(sample)+"_kd_"+str(kd)+"_2_cluster_numRNAs_"+str(int(N_rna1*N_rna2))+"_numtrimers_"+str(int(N_trimer1*N_trimer2))+"_Ly_"+str(Ly_final)+"_.txt"
np.savetxt(filename2, np.vstack((bins[:-1],normalized_cluster_size, avg_hist)).T,
           header="Cluster size\tNormalized size\tAverage count", fmt='%d\t%.10f\t%.10f')

exit()


