#calculate binding information for simulation

import gsd
import gsd.hoomd
import time
import numpy as np
import random 
import math
import sys
import pandas
import freud
import networkx as nx

sample=sys.argv[1]
Yytrimer = int(sys.argv[2])
Yyrna = int(sys.argv[3])
Ly_final = float(sys.argv[4])
epsilon1 = 6.0
kd=float(sys.argv[5])

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

filename1="trajectory_cnf_"+str(sample)+"_kd_"+str(kd)+"_numTrimers_"+str(int(N_trimer1*N_trimer2))+"_numRNA_"+str(int(N_rna1*N_rna2))+"_Ly_"+str(Ly_final)+"_2.gsd"


traj = gsd.hoomd.open(filename1, 'r')

bondarray  = traj[0].bonds.group

N_tot=trimer_tot+rna_tot

Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
box=freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
box1=np.asarray([Lx,Ly,Lz])
dt=0.005*(traj[1].configuration.step-traj[0].configuration.step)


def find_spanning_path(points, bonds_array, box, tol=1.5):
    
    N = len(points)
    Lx, Ly, Lz = box

    # shifts along xyz
    shifts = [N, 2*N, 3*N]
    lengths = [Lx, Ly, Lz]

    for dim in range(3):  # 0=x,1=y,2=z
        shift = shifts[dim]
        L = lengths[dim]

        G = nx.Graph()
        # 1. original bond
        for i, j in bonds_array:
            G.add_edge(i, j)
        # 2. image bond
        bonds_img = bonds_array + shift
        for i, j in bonds_img:
            G.add_edge(i, j)
        # 3. cross boundary bond (+ axis)
        for i, j in bonds_array:
            if points[i, dim] >= L/2 - tol and points[j, dim] <= -L/2 + tol:
                G.add_edge(i, j+shift)
                G.add_edge(i+shift, j)
                G.remove_edge(i, j)
                G.remove_edge(i+shift, j+shift)
            if points[j, dim] >= L/2 - tol and points[i, dim] <= -L/2 + tol:
                G.add_edge(i, j+shift)
                G.add_edge(i+shift, j)
                G.remove_edge(i, j)
                G.remove_edge(i+shift, j+shift)

        start_node = np.random.choice(list(range(N)))
        try:
        	path = nx.shortest_path(G, source=start_node, target=start_node+shift)
        	# print("find spanning path")
        	return path, dim, 1
        except nx.NetworkXNoPath:
        	# print("no path")
        	continue
    return None, None, 0

for i in range(len(traj)):
	# i=1
	# i=-1
	if i<50:
		continue

	points =traj[i].particles.position
	bondarray  = traj[i].bonds.group

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

	if len(keyswith2)==0:
		filename2=str(sample)+"_3_epsilon1_"+str(epsilon1)+"_kd_"+str(kd)+"_1_binding_information_numTrimers_"+str(int(N_trimer1*N_trimer2))+"_numRNA_"+str(int(N_rna1*N_rna2))+"_Ly_"+str(Ly_final)+"_.txt"
		a=np.append(i*dt,len(keyswith2)/(N_rna1*N_rna2*N_sticky_each_rna_chian))
		a=np.append(a,len(keyswith2)/(N_trimer1*N_trimer2*N_sticky_each_trimer_chian))
		a=np.append(a,len(keyswith2))
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,0)
		a=a.reshape(1,a.shape[0])
		with open(filename2,'a+') as f3:
			np.savetxt(f3,a)
		continue
	
	bonds_array = np.copy(bondarray)
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
	cluster_keys = cl1.cluster_keys
	cluster_idx = cl.cluster_idx

	bonds_array = np.column_stack((neighbors.query_point_indices, neighbors.point_indices))

	# Ensure cluster_keys contains integers
	cluster_keys1 = list(range(len(cluster_keys)))  # Replace lists with unique integer IDs

	maxcl_size=sum(np.array(cluster_keys[0])<trimer_tot)/trimer_len+sum(np.array(cluster_keys[0])>=trimer_tot)/rna_len
	
	# Create a dictionary to store bonds for each cluster
	bonds_by_cluster = {key: [] for key in cluster_keys1}

	# Filter bonds based on cluster indices
	for bond in bonds_array:
	    p1, p2 = bond  # Get particle indices in the bond
	    cluster1 = cluster_idx[p1]
	    cluster2 = cluster_idx[p2]

	    # Add the bond to the corresponding cluster if both particles belong to the same cluster
	    if cluster1 == cluster2:
	        bonds_by_cluster[cluster1].append((p1, p2))

	# Convert bond lists to arrays for easier handling
	for key in bonds_by_cluster:
	    bonds_by_cluster[key] = np.array(bonds_by_cluster[key])
	
	spanning_clusters=[]
	for kkk in range(len(cluster_keys)):
		if len(cluster_keys[kkk])<=max(trimer_len,rna_len):
			continue
		cluster_indices = cluster_keys[kkk]
		cl_point=points[cluster_keys[kkk]]
		
		global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(cluster_indices)}
		local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(cluster_indices)}
		cl_bonds_array_global = bonds_by_cluster[kkk]
		cl_bonds_array = np.array([[global_to_local[i], global_to_local[j]] 
                               for i, j in cl_bonds_array_global])
		
		path, dim, spanclust=find_spanning_path(cl_point, cl_bonds_array, box1, tol=1.5)
		
		spanning_clusters.append(spanclust)
	
	spanclust=sum(spanning_clusters)

	interbind_bond=[]
	nnnn=0
	for j in range(len(bonds_by_cluster)):
		ppp=0
		newlis=[]
		for k in range(len(bonds_by_cluster[j])):
			if bonds_by_cluster[j][k][0]<trimer_tot:
				a1=bonds_by_cluster[j][k][0]-(bonds_by_cluster[j][k][0]%trimer_len)
			else:
				a1=bonds_by_cluster[j][k][0]-((bonds_by_cluster[j][k][0]-trimer_tot)%rna_len)
			if bonds_by_cluster[j][k][1]<trimer_tot:
				a2=bonds_by_cluster[j][k][1]-(bonds_by_cluster[j][k][1]%trimer_len)
			else:
				a2=bonds_by_cluster[j][k][1]-((bonds_by_cluster[j][k][1]-trimer_tot)%rna_len)
			if a1!=a2:
				if ppp==0:
					newlis=np.append(a1,a2)
					newlis=newlis.reshape(1,newlis.shape[0])
					ppp=1
				else:
					c=np.append(a1,a2)
					c=c.reshape(1,c.shape[0])
					newlis=np.append(newlis,c,axis=0)
		nnnn+=len(newlis)
		interbind_bond.append(newlis)
	singlenum=[]
	cycle_number=[]
	edges_to_remove_break_cycles=[]

	for ii in range(len(interbind_bond)):
		if len(interbind_bond[ii])!=0:
			
			vertices = np.unique(np.array(interbind_bond[ii]).flatten())  # Numeric vertices
			edges = interbind_bond[ii]  # Numeric edges
			G = nx.MultiGraph()
			G.add_nodes_from(vertices)  # Add vertices
			G.add_edges_from(edges)     # Add edges
			
			## get singlenum
			single_nodes = [node for node, degree in G.degree() if degree == 1]
			singlenum=np.append(singlenum,len(single_nodes))

			simple_G = nx.Graph(G)
			E = simple_G.number_of_edges()
			V = simple_G.number_of_nodes()
			C = nx.number_connected_components(simple_G)
			cycle_number = np.append(cycle_number,E - V + C+(len(edges)-E))
			
			mst = nx.minimum_spanning_tree(simple_G)

			edges_to_remove = list(set(simple_G.edges) - set(mst.edges))

			edges_to_remove_break_cycles=np.append(edges_to_remove_break_cycles,len(edges_to_remove)+(len(edges)-E))


	ppp=0
	newlis=[]
	for k in range(len(keyswith2)):
		a1=keyswith2[k][0]-(keyswith2[k][0]%trimer_len)
		a2=keyswith2[k][1]-((keyswith2[k][1]-trimer_tot)%rna_len)
		if ppp==0:
			newlis=np.append(a1,a2)
			newlis=newlis.reshape(1,newlis.shape[0])
			ppp=1
		else:
			c=np.append(a1,a2)
			c=c.reshape(1,c.shape[0])
			newlis=np.append(newlis,c,axis=0)

	trimerlis=np.unique(newlis[:,0])
	n_bridge=0
	n_loop=0
	n_binding=0
	nn=0
	for k in range(len(trimerlis)):
		indx1=np.where(newlis==trimerlis[k])
		if len(indx1[0])==1:
			n_binding=n_binding+1
			nn=nn+1
			continue
		else:
			nn=nn+len(indx1[0])
			rna_lis=[]
			for j in range(len(indx1[0])):
				rna_lis=np.append(rna_lis,newlis[indx1[0][j]][1])
			n_loop=n_loop+(len(rna_lis)-len(np.unique(rna_lis)))
			n_bridge=n_bridge+(len(rna_lis)-1-(len(rna_lis)-len(np.unique(rna_lis))))

	if spanclust>=1:
		spanclust=1
	else:
		spanclust=0

	filename2=str(sample)+"_3_epsilon1_"+str(epsilon1)+"_kd_"+str(kd)+"_1_binding_information_numTrimers_"+str(int(N_trimer1*N_trimer2))+"_numRNA_"+str(int(N_rna1*N_rna2))+"_Ly_"+str(Ly_final)+"_.txt"
	a=np.append(i*dt,len(keyswith2)/(N_rna1*N_rna2*N_sticky_each_rna_chian))
	a=np.append(a,len(keyswith2)/(N_trimer1*N_trimer2*N_sticky_each_trimer_chian))
	a=np.append(a,len(keyswith2))
	a=np.append(a,np.sum(singlenum))
	a=np.append(a,np.sum(edges_to_remove_break_cycles))
	a=np.append(a,np.sum(cycle_number))
	a=np.append(a,n_bridge)
	a=np.append(a,n_loop)
	a=np.append(a,n_binding)
	a=np.append(a,spanclust)
	a=np.append(a,maxcl_size/(N_trimer1*N_trimer2+N_rna1*N_rna2))
	a=a.reshape(1,a.shape[0])
	with open(filename2,'a+') as f3:
		np.savetxt(f3,a)
 
exit()


