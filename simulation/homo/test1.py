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
n_polymer = int(sys.argv[2])
N_length = int(sys.argv[3])
f_sticker=int(sys.argv[4])
L_final = float(sys.argv[5])
eps1= float(sys.argv[6])  ## Interaction strength of sticky potential
sigmas= 1 # float(sys.argv[6]) # samll radius>=(sqrt(2)-1)sigma=0.414sigma

real_l=int((N_length-f_sticker)/(f_sticker+1))


filename1="trajectory_cnf_"+str(sample)+"_epsilon1_"+str(eps1)+"_polymer_"+str(int(n_polymer))+"_length_"+str(N_length)+"_sticker_"+str(f_sticker)+"_sigmas_"+str(sigmas)+"_boxsize_"+str(L_final)+".gsd"

total_chains = n_polymer

traj = gsd.hoomd.open(filename1, 'rb')

bondarray  = traj[0].bonds.group
types = traj[0].particles.typeid

N_tot=n_polymer*N_length
re_size = n_polymer*(N_length-1)

sticker_index=types==1
sticker_index=sticker_index[0:N_tot]

Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
box=freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
box1=np.asarray([Lx,Ly,Lz])
dt=0.005*(traj[1].configuration.step-traj[0].configuration.step)

bins = np.arange(1, total_chains + 2)  # Bins: 1 to total_chains
hist_accum = np.zeros(len(bins) - 1)
frame_count = 0

for i in range(len(traj)):
	
	if i<50:
		continue
	points =traj[i].particles.position[0:N_tot]
	bondarray  = traj[0].bonds.group[0:re_size]

	type1_indices = np.where(types == 1)[0]

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
	distances = np.linalg.norm(box.wrap(points[bonds_array[:, 1]] - points[bonds_array[:, 0]]),axis=1)
	neighbors = freud.locality.NeighborList.from_arrays(len(points),len(points),
	                bonds_array[:, 0],
	                bonds_array[:, 1],
	                distances,
	            )

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
			head=b[kk][ll]-b[kk][ll]%N_length
			cl_list0.append(head)
		cl_list0=np.unique(cl_list0)
		new_cl_list.append(cl_list0)
		cl_size_list.append(len(cl_list0))
	hist, _ = np.histogram(cl_size_list, bins=bins)
	hist_accum += hist
	frame_count += 1

avg_hist = hist_accum / frame_count
normalized_cluster_size = bins[:-1] / total_chains

filename2="./cl_dis_1/"+str(sample)+"_2_cluster_epsilon1_"+str(eps1)+"_polymer_"+str(int(n_polymer))+"_length_"+str(N_length)+"_sticker_"+str(f_sticker)+"_sigmas_"+str(sigmas)+"_boxsize_"+str(L_final)+"_.txt"
np.savetxt(filename2, np.vstack((bins[:-1],normalized_cluster_size, avg_hist)).T,
           header="Cluster size\tNormalized size\tAverage count", fmt='%d\t%.10f\t%.10f')

exit()

