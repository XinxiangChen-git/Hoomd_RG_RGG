#calculate binding information for simulation

import gsd
import gsd.hoomd
import time
import numpy as np
import random 
import math
import sys
import freud
import matplotlib.pyplot as plt
import networkx as nx

sample=sys.argv[1]
n_polymer = int(sys.argv[2])
N_length = int(sys.argv[3])
f_sticker=int(sys.argv[4])
L_final = float(sys.argv[5])
eps1= float(sys.argv[6])  ## Interaction strength of sticky potential
sigmas= 1 # float(sys.argv[6]) # samll radius>=(sqrt(2)-1)sigma=0.414sigma

real_l=int((N_length-f_sticker)/(f_sticker+1))

filename1="trajectory_cnf_"+str(sample)+"_epsilon1_"+str(eps1)+"_polymer_"+str(int(n_polymer))+"_length_"+str(N_length)+"_sticker_"+str(f_sticker)+"_sigmas_"+str(sigmas)+"_boxsize_"+str(L_final)+".gsd"

traj = gsd.hoomd.open(filename1, 'rb')

bondarray  = traj[0].bonds.group
types = traj[0].particles.typeid

sticker_index=types==1

N_tot=n_polymer*N_length

Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
box=freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
box1=np.asarray([Lx,Ly,Lz])
dt=0.005*(traj[1].configuration.step-traj[0].configuration.step)

for i in range(len(traj)):
	
	if i<len(traj)/2:
		continue
	if i%50!=0:
		continue

	points =traj[i].particles.position
	bondarray  = traj[0].bonds.group

	type1_indices = np.where(types == 1)[0]
	
	cluster = freud.cluster.Cluster()
	cluster.compute((box,points[sticker_index]),neighbors={"r_max": 0.5})
	r=cluster.cluster_idx
	b=cluster.cluster_keys
	
	keyswith2=[]
	iii=0
	inddd=0
	for j in range(len(b)):
		if len(b[j])>2:
			# print(b[i])
			inddd=1
		if len(b[j])==2:
			iii+=1
		
	cl_props = freud.cluster.ClusterProperties()
	cl_props.compute((box,points[sticker_index]), cluster.cluster_idx)
	n=cl_props.sizes  ###n[i] will return the value of the size of each cluster
	cl2index=n==2
	keys=np.asarray(b,dtype=object)
	keyswith2=np.asarray(keys[cl2index].tolist())
	if len(keyswith2)==0:
		filename2=str(sample)+"_2_epsilon1_"+str(eps1)+"_1_binding_information_polymer_"+str(int(n_polymer))+"_length_"+str(N_length)+"_sticker_"+str(f_sticker)+"_boxsize_"+str(L_final)+"_.txt"
		a=np.append(i*dt,len(keyswith2)*2/(n_polymer*f_sticker))
		a=np.append(a,len(keyswith2))
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,inddd)
		a=np.append(a,0)
		a=np.append(a,0)
		a=np.append(a,0)
		a=a.reshape(1,a.shape[0])
		with open(filename2,'a+') as f3:
			np.savetxt(f3,a)
		continue


	if len(keyswith2)>0:
		keyswith2=type1_indices[keyswith2]
		newlis=keyswith2-keyswith2%N_length
		n_intra=0
		for k in range(len(newlis)):
			if newlis[k][0]==newlis[k][1]:
				n_intra+=1
		n_inter=len(newlis)-n_intra

	bonds_array = np.copy(bondarray)
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
	cluster_idx = cl.cluster_idx

	points_x=np.append(points,traj[i].particles.position-[Lx,0,0],axis=0)
	bonds_array_x = np.append(bonds_array,np.copy(bonds_array+[len(points),len(points)]),axis=0)
	
	points_y=np.append(points,traj[i].particles.position-[0,Ly,0],axis=0)
	bonds_array_y = np.append(bonds_array,np.copy(bonds_array+[len(points),len(points)]),axis=0)
	
	points_z=np.append(points,traj[i].particles.position-[0,0,Lz],axis=0)
	bonds_array_z = np.append(bonds_array,np.copy(bonds_array+[len(points),len(points)]),axis=0)
	
	for j in range(len(points_x)):
		if points_x[j,0]>=Lx/2.0-1.5:
			index1=np.where(bonds_array_x==j)
			n1=np.shape(index1)[1]
			for l in range(n1):
				if index1[1][l]==0:
					pair_p=bonds_array_x[index1[0][l]][1]
					if points_x[pair_p][0]<=(-Lx/2.0+1.5):
						bonds_array_x[index1[0][l]][1]=pair_p+len(points)
						index2=np.where(bonds_array_x==bonds_array_x[index1[0][l]][0]+len(points))
						n2=np.shape(index2)[1]
						for mm in range(n2):
							if index2[1][mm]==0 and bonds_array_x[index2[0][mm]][1]==pair_p+len(points):
								bonds_array_x[index2[0][mm]][1]=pair_p
				else:
					pair_p=bonds_array_x[index1[0][l]][0]
					if points_x[pair_p][0]<=(-Lx/2.0+1.5):
						bonds_array_x[index1[0][l]][0]=pair_p+len(points)
						index2=np.where(bonds_array_x==bonds_array_x[index1[0][l]][1]+len(points))
						n2=np.shape(index2)[1]
						for mm in range(n2):
							if index2[1][mm]==1 and bonds_array_x[index2[0][mm]][0]==pair_p+len(points):
								bonds_array_x[index2[0][mm]][0]=pair_p
		
		if points_y[j,1]>=Ly/2.0-1.5:
			index1=np.where(bonds_array_y==j)
			n1=np.shape(index1)[1]
			for l in range(n1):
				if index1[1][l]==0:
					pair_p=bonds_array_y[index1[0][l]][1]
					if points_y[pair_p][1]<=(-Ly/2.0+1.5):
						bonds_array_y[index1[0][l]][1]=pair_p+len(points)
						index2=np.where(bonds_array_y==bonds_array_y[index1[0][l]][0]+len(points))
						n2=np.shape(index2)[1]
						for mm in range(n2):
							if index2[1][mm]==0 and bonds_array_y[index2[0][mm]][1]==pair_p+len(points):
								bonds_array_y[index2[0][mm]][1]=pair_p
				else:
					pair_p=bonds_array_y[index1[0][l]][0]
					if points_y[pair_p][1]<=(-Ly/2.0+1.5):
						bonds_array_y[index1[0][l]][0]=pair_p+len(points)
						index2=np.where(bonds_array_y==bonds_array_y[index1[0][l]][1]+len(points))
						n2=np.shape(index2)[1]
						for mm in range(n2):
							if index2[1][mm]==1 and bonds_array_y[index2[0][mm]][0]==pair_p+len(points):
								bonds_array_y[index2[0][mm]][0]=pair_p
		
		if points_z[j,2]>=Lz/2.0-1.5:
			index1=np.where(bonds_array_z==j)
			n1=np.shape(index1)[1]
			for l in range(n1):
				if index1[1][l]==0:
					pair_p=bonds_array_z[index1[0][l]][1]
					if points_z[pair_p][2]<=(-Lz/2.0+1.5):
						bonds_array_z[index1[0][l]][1]=pair_p+len(points)
						index2=np.where(bonds_array_z==bonds_array_z[index1[0][l]][0]+len(points))
						n2=np.shape(index2)[1]
						for mm in range(n2):
							if index2[1][mm]==0 and bonds_array_z[index2[0][mm]][1]==pair_p+len(points):
								bonds_array_z[index2[0][mm]][1]=pair_p
				else:
					pair_p=bonds_array_z[index1[0][l]][0]
					if points_z[pair_p][2]<=(-Lz/2.0+1.5):
						bonds_array_z[index1[0][l]][0]=pair_p+len(points)
						index2=np.where(bonds_array_z==bonds_array_z[index1[0][l]][1]+len(points))
						n2=np.shape(index2)[1]
						for mm in range(n2):
							if index2[1][mm]==1 and bonds_array_z[index2[0][mm]][0]==pair_p+len(points):
								bonds_array_z[index2[0][mm]][0]=pair_p
	
	spanclust=0
	spanning_index=[]

	for k in range(len(b)):
		partids=np.asarray(b[k])
		if len(b[k])>65:
			pos = points[partids]
			check=0
			for j in range(3):
				fstcheck=pos[:,j]<=-box1[j]/2+2.0
				sndcheck=pos[:,j]>=box1[j]/2-2.0
				trdcheck=(pos[:,j]>=-2.0)&(pos[:,j]<=2.0)
				forcheck=(pos[:,j]>=-box1[j]/4-2.0)&(pos[:,j]<=-box1[j]/4+2.0)
				fifcheck=(pos[:,j]>=box1[j]/4-2.0)&(pos[:,j]<=box1[j]/4+2.0)
				sum1=np.sum(fstcheck)
				sum2=np.sum(sndcheck)
				sum3=np.sum(trdcheck)
				sum4=np.sum(forcheck)
				sum5=np.sum(fifcheck)
				check=np.logical_and(np.logical_and(np.logical_and(np.logical_and(sum1,sum2),sum3),sum4),sum5)
				
				if check==1:
					b_b=np.asarray(b[k])
					b_2=np.append(b_b,b_b+len(points),axis=0)
					b_2=np.sort(b_2)
					
					if j==0:
						# print('------candidate--------')
						pos_x=np.append(pos,pos-[Lx,0,0],axis=0)
						front=b_2[0]
						queue=np.asarray(front)
						father=np.asarray([[front,front]])
						while (b_2[0]+len(points)) not in queue:
							index1=np.where(bonds_array_x==front)
							n1=np.shape(index1)[1]
							for l in range(n1):
								if index1[1][l]==0:
									pair_p=bonds_array_x[index1[0][l]][1]
									if pair_p not in father[:,0]:
										queue=np.append(queue,pair_p)
										father=np.append(father,[[pair_p,front]],axis=0)
								else:
									pair_p=bonds_array_x[index1[0][l]][0]
									if pair_p not in father[:,0]:
										queue=np.append(queue,pair_p)
										father=np.append(father,[[pair_p,front]],axis=0)
							queue=np.delete(queue,0)
							if len(father[:,0])==len(b_2) or len(queue)==0:
								check=0
								break
							front=queue[0]
					elif j==1:
						# print('------candidate--------')
						pos_y=np.append(pos,pos-[0,Ly,0],axis=0)
						front=b_2[0]
						queue=np.asarray(front)
						father=np.asarray([[front,front]])
						while (b_2[0]+len(points)) not in queue:
							index1=np.where(bonds_array_y==front)
							n1=np.shape(index1)[1]
							for l in range(n1):
								if index1[1][l]==0:
									pair_p=bonds_array_y[index1[0][l]][1]
									if pair_p not in father[:,0]:
										queue=np.append(queue,pair_p)
										father=np.append(father,[[pair_p,front]],axis=0)
								else:
									pair_p=bonds_array_y[index1[0][l]][0]
									if pair_p not in father[:,0]:
										queue=np.append(queue,pair_p)
										father=np.append(father,[[pair_p,front]],axis=0)
							queue=np.delete(queue,0)
							if len(queue)==0:
								check=0
								break
							front=queue[0]
							
					else:
						# print('------candidate--------')
						pos_z=np.append(pos,pos-[0,0,Lz],axis=0)
						front=b_2[0]
						queue=np.asarray(front)
						father=np.asarray([[front,front]])
						while (b_2[0]+len(points)) not in queue:
							index1=np.where(bonds_array_z==front)
							n1=np.shape(index1)[1]
							for l in range(n1):
								if index1[1][l]==0:
									pair_p=bonds_array_z[index1[0][l]][1]
									if pair_p not in father[:,0]:
										queue=np.append(queue,pair_p)
										father=np.append(father,[[pair_p,front]],axis=0)
								else:
									pair_p=bonds_array_z[index1[0][l]][0]
									if pair_p not in father[:,0]:
										queue=np.append(queue,pair_p)
										father=np.append(father,[[pair_p,front]],axis=0)
							queue=np.delete(queue,0)
							if len(father[:,0])==len(b_2) or len(queue)==0:
								check=0
								break
							front=queue[0]
							
					if check==0:
						# print('-----------no path------------')
						continue
					else:
						spanning_index=np.append(spanning_index,k)
						spanclust+=1
						break
	
	bonds_array = np.column_stack((neighbors.query_point_indices, neighbors.point_indices))
	cluster_keys1 = list(range(len(b)))  # Replace lists with unique integer IDs
	
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

	interbind_bond=[]
	
	nnnn=0
	for j in range(len(bonds_by_cluster)):
		ppp=0
		newlis=[]
		newlis=bonds_by_cluster[j]-bonds_by_cluster[j]%N_length
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
			cycle_number = np.append(cycle_number,E - V + C)

			mst = nx.minimum_spanning_tree(simple_G)

			edges_to_remove = list(set(simple_G.edges) - set(mst.edges))

			edges_to_remove_break_cycles=np.append(edges_to_remove_break_cycles,len(edges_to_remove))

	if spanclust>=1:
		spanclust=1
	else:
		spanclust=0

	filename2=str(sample)+"_2_epsilon1_"+str(eps1)+"_1_binding_information_polymer_"+str(int(n_polymer))+"_length_"+str(N_length)+"_sticker_"+str(f_sticker)+"_boxsize_"+str(L_final)+"_.txt"
	a=np.append(i*dt,len(keyswith2)*2/(n_polymer*f_sticker))
	a=np.append(a,len(keyswith2))
	a=np.append(a,np.mean(singlenum))
	a=np.append(a,np.mean(cycle_number)+n_intra)
	a=np.append(a,np.mean(edges_to_remove_break_cycles)+n_intra)
	a=np.append(a,inddd)
	a=np.append(a,n_intra)
	a=np.append(a,n_inter)
	a=np.append(a,spanclust)
	a=a.reshape(1,a.shape[0])
	with open(filename2,'a+') as f3:
		np.savetxt(f3,a)

exit()

