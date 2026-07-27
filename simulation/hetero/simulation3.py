#!/usr/bin/python3
import hoomd
import hoomd.md
#import gsd
#import gsd.hoomd
import time
import numpy
import random 
import math
import sys
########################################################################
#           Set parameters
#######################################################################
sample=sys.argv[1]
Yytrimer = int(sys.argv[2])
Yyrna = int(sys.argv[3])
Ly_final = float(sys.argv[4])


N_trimer1 = 10        # trimers in x direction
N_trimer2 = Yytrimer  # trimers in y direction
trimer_len = 23       # Monomers in z direction
trimer_tot = N_trimer1*N_trimer2*trimer_len # total monomers of trimers

N_rna1 = 10           # rna chains in x direction
N_rna2=Yyrna             # rna chains in y direction
rna_len =65           # length of rna chain
rna_tot = N_rna1*N_rna2*rna_len #total rna monomers

###Diameter of each type of monomers###
dia=[]
dia.append(1.0)
dia.append(1.0)
dia.append(1.0)
dia.append(1.0)
########################################define_a_list/1-D_array

###Characteristic distance for each monomers###
sigma=[]  
sigma.append([1.,1.,1.,1.])
sigma.append([1.,1.,1.,1.])
sigma.append([1.,1.,1.,1.])
sigma.append([1.,1.,1.,1.])
########################################define_a_list/2-D_array

r0 = math.pow(2.,1./6.) ##Equilibrium bond length

trimer_Mono_space = r0  # r0*sigma # spacing between two monomers of trimer chain in z direction
trimer_spacex = r0      # spacing between two monomers of trimer chain along x direction
trimer_spacey = r0      # spacing between two monomers of trimer chain along y direction

###( control this to control the density of rna monomers)############

rna_spacex=8.5       #spacing between the rna chains in x direction 
rna_spacey=8.5       #spacing between the rna chain in y direction
rna_Mono_space = r0  #spacing between two monomers of rna chain in z direction
################################################################


Lx = N_trimer1*trimer_spacex      # Length of simulation box in x direction  
Ly = N_trimer2*trimer_spacey      # Lenght of simulation box in y direction
Lz = trimer_Mono_space*trimer_len # length of simulation box in z direction

trimer_box_Volume = Lx*Ly*Lz      #Total volume on with trimers


delLx1=N_rna1*rna_spacex    #Extension in x direction of simulation box after intializing trimers
Ly1=N_rna2*rna_spacey       #Length of simulation box in y direction in accordance with rna's 
Lz1=rna_Mono_space*rna_len  #Length of simulation box in z direction in accordance with rna's

epsilon = 1.0     ## Interaction strength for WCA
epsilon1 = 6.0    ## Interaction strength of sticky potential
a = 1.0         #alpha parameter in WCA
rcut =  (2/a)**(1./6) #cut- off of WCA potential
rmax = 0.5 #cut-off of sticky potential
rmin = 0.03 #minimum distance to which force is calculated
ka = 30 #Harmonic bond strength
kd = float(sys.argv[5]) # Harmonic angle potential
dt =0.005 # time step
kbT = 1.0 #0.008314*Temperature


Lx = Lx+delLx1         #Extend Lx 

Ly_small=min(Ly,Ly1)    # smallest of Ly lengths after comparing trimer's and rna's
Ly=max(Ly,Ly1)          # largest of Ly lengths after comparing trimer's and rna's

Lz_small = min(Lz,Lz1)  # smallest of Lz lengths after comparing trimer's and rna's
Lz = max(Lz,Lz1)        # largest of Lz lengths after comparing trimer's and rna'


Lxby2=Lx/2.
Lyby2=Ly/2.
Lzby2=Lz/2.

Volume = Lx*Ly*Lz      #Volume of the total simulation box
No_trimer_chains = N_trimer1*N_trimer2 
No_rna_chains = N_rna1*N_rna2

density_trimers = trimer_tot/Volume ##Density of trimer monomers in the system
density_rna = rna_tot/Volume        ##Density of rna monomers in the system



###Defining sticky potential#####
def sticky(r, rmin, rmax, epsilon, sigma):
    V = -epsilon*(math.cos(2*math.pi*r/(sigma*2.0*rmax))+1)
    F = -2*math.pi*epsilon*(math.sin(2*math.pi*r/(sigma*2.0*rmax)))/(sigma*2.0*rmax);
    return (V, F)

################################################
#  Random number seed
###############################################
seed1 = random.randint(1,999999999)
seed2 = random.randint(1,999999999)
seed3 = random.randint(1,999999999)


###################################################################################
#                 Initialize the system
###################################################################################
context = hoomd.context.initialize("--mode=gpu")
#print(trimer_Mono_space*trimer_len-Lzby2,-Lzby2,Lz)
z_coords_trimer=numpy.arange(-Lzby2+trimer_Mono_space,-Lzby2+trimer_Mono_space*trimer_len+trimer_Mono_space,trimer_Mono_space)
y_coords_trimer=numpy.arange(-Lyby2+trimer_spacey,-Lyby2+trimer_spacey*N_trimer2+trimer_spacey,trimer_spacey)
x_coords_trimer=numpy.arange(-Lxby2+trimer_spacex,-Lxby2+trimer_spacex*N_trimer1+trimer_spacex,trimer_spacex)

#########################################################################
x_pos = x_coords_trimer[len(x_coords_trimer)-1] #Marker position along x direction to start rna chains

################Lattice creation for RNA monomers ##################
z_coords_rna = numpy.linspace(-Lzby2+rna_Mono_space, Lzby2-rna_Mono_space, rna_len)
y_coords_rna = numpy.linspace(-Lyby2+rna_spacey, Lyby2- rna_spacey, N_rna2)
x_coords_rna = numpy.arange(x_pos+rna_spacex,x_pos+rna_spacex*N_rna1+rna_spacex,rna_spacex)# Lxby2constraint - Poly_space, Nx_constraint)
################################################################
Lx = Lx+rna_spacex # Extending the box in x direction to include all monomers (in order to make the first particle and the last particle far away from each other along x-direction)


snapshot = hoomd.data.make_snapshot(N=rna_tot+trimer_tot,box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),particle_types=['A','B','C','D'],bond_types=['polymer'],angle_types=['polymer']);
N_bond_rna= (rna_len-1)*N_rna1*N_rna2
N_bond_trimer = (trimer_len-1)*N_trimer1*N_trimer2
re_size = N_bond_rna+N_bond_trimer

N_angle_rna= (rna_len-2)*N_rna1*N_rna2
N_angle_trimer = (trimer_len-2)*N_trimer1*N_trimer2
re_size_angle = N_angle_rna+N_angle_trimer


count = 0
count_angle = 0
snapshot.bonds.resize( re_size ) ## creat a bond group to store the bonds
snapshot.angles.resize( re_size_angle ) ## creat an agnle group to store the angles

#trimer chain initialization
for i in range(N_trimer1):
    for j in range(N_trimer2):
        for k in range(trimer_len):

            index = N_trimer2*trimer_len*i+j*trimer_len+k #index of trimer monomer
            #writing trimer monomer position
            snapshot.particles.position[index,0]=x_coords_trimer[i] 
            snapshot.particles.position[index,1]=y_coords_trimer[j]
            snapshot.particles.position[index,2]=z_coords_trimer[k]
            
            #Specifying the particle type to trimer monomers
            if k%11==0:
                snapshot.particles.typeid[index] = 0    #0 indicates sticky region in trimer chain
            else:
                snapshot.particles.typeid[index] = 1    #non sticky region of trimer chain is indicated by 1
            snapshot.particles.diameter[index] = dia[0]  #diameter of trimer monomers
            
            snapshot.particles.mass[index]=1.0  #mass of the trimer monomer

            
            #Bond group initialization
            index1=N_trimer2*trimer_len*i+j*trimer_len+k+1  #index of consecutive monomer in the trimer chain
            if k<trimer_len-1 and count< N_bond_trimer:
                snapshot.bonds.group[count,0] = index
                snapshot.bonds.group[count,1] = index1
                snapshot.bonds.typeid[count]=0
                count+=1

            #Angle group initialization
            index2=N_trimer2*trimer_len*i+j*trimer_len+k+2  #index of next consecutive monomer in the trimer chain
            if k<trimer_len-2 and count_angle< N_angle_trimer:
                snapshot.angles.group[count_angle,0] = index
                snapshot.angles.group[count_angle,1] = index1
                snapshot.angles.group[count_angle,2] = index2
                snapshot.angles.typeid[count_angle]=0
                count_angle+=1



count2=count
count_angle2=count_angle
#rna chain initialization
for i in range(N_rna1):
    for j in range(N_rna2):
        for k in range(rna_len):

            index3 = trimer_tot+N_rna2*rna_len*i+j*rna_len+k    #index of monomer of the rna chain
            #Writing position of rna monomers
            snapshot.particles.position[index3,0]=x_coords_rna[i]
            snapshot.particles.position[index3,1]=y_coords_rna[j]
            snapshot.particles.position[index3,2]=z_coords_rna[k]
            
            #Specifying the particle type of rna monomers
            if (k-(rna_len+5-rna_len%(5+1)))%(5+1)==0 :
                snapshot.particles.typeid[index3] = 3   # 3 indicates the sticky region of rna chain
            else:
                snapshot.particles.typeid[index3] = 2   #2 indicates the non-sticky region of trimer chain
            snapshot.particles.diameter[index3] = dia[0] #diameter of the rna monomers
            
            snapshot.particles.mass[index3]=1.0         #mass of the rna monomers

            
            #Bond group initialization
            index4 = trimer_tot+N_rna2*rna_len*i+j*rna_len+k+1    #index of the consecutive monomer in the rna chain
            if k<rna_len-1 and count2< re_size:
                snapshot.bonds.group[count2,0] = index3
                snapshot.bonds.group[count2,1] = index4
                snapshot.bonds.typeid[count2]=0
                count2+=1

            #Angle group initialization
            index5 = trimer_tot+N_rna2*rna_len*i+j*rna_len+k+2    #index of the consecutive monomer in the rna chain
            if k<rna_len-2 and count_angle2< re_size_angle:
                snapshot.angles.group[count_angle2,0] = index3
                snapshot.angles.group[count_angle2,1] = index4
                snapshot.angles.group[count_angle2,2] = index5
                snapshot.angles.typeid[count_angle2]=0
                count_angle2+=1

system = hoomd.init.read_snapshot(snapshot); ##Initialize the system(Snapshots are set to time_step 0, and should not be used to restart a simulation)
all = hoomd.group.all() #Creates a particle group from all particles in the simulation.

####################Equilibrtion Run######################################################

#################################################################################################
#               Force initialization

################################################################################################
#####Create neighborlist######################################

nl = hoomd.md.nlist.tree(r_buff = 0.5, check_period = 1) ## in order to accelerate pair force calculation
nl.reset_exclusions(exclusions = []); ## select which interactions should be excluded from the pair interaction calculation

######Equilibration using WCA interactions####################################
wca = hoomd.md.pair.lj(r_cut= rcut,nlist = nl)

wca.set_params(mode='shift') ##shift the potential to make sure that it is 0 at the cutoff
wca.pair_coeff.set('A','A', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut, alpha = a)
wca.pair_coeff.set('B','B', epsilon = epsilon, sigma =sigma[1][1],r_cut = sigma[1][1]*rcut, alpha = a)
wca.pair_coeff.set('C','C', epsilon = epsilon, sigma =sigma[2][2],r_cut = sigma[2][2]*rcut, alpha = a)
wca.pair_coeff.set('D','D', epsilon = epsilon, sigma =sigma[3][3],r_cut = sigma[3][3]*rcut, alpha = a)
wca.pair_coeff.set('A','B', epsilon = epsilon, sigma =sigma[0][1],r_cut = sigma[0][1]*rcut, alpha = a)
wca.pair_coeff.set('A','C', epsilon = epsilon, sigma =sigma[0][2],r_cut = sigma[0][2]*rcut, alpha = a)
wca.pair_coeff.set('A','D', epsilon = epsilon, sigma =sigma[0][3],r_cut = sigma[0][3]*rcut, alpha = a)
wca.pair_coeff.set('B','C', epsilon = epsilon, sigma =sigma[1][2],r_cut = sigma[1][2]*rcut, alpha = a)
wca.pair_coeff.set('B','D', epsilon = epsilon, sigma =sigma[1][3],r_cut = sigma[1][3]*rcut, alpha = a)
wca.pair_coeff.set('C','D', epsilon = epsilon, sigma =sigma[2][3],r_cut = sigma[2][3]*rcut, alpha = a)

##################################################################################

hoomd.md.integrate.mode_standard(dt= dt)
###Set harmonic bonds ############################################################
harmonic= hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('polymer', k =ka, r0 = r0)

########Choosing Harmonic angle potential#################
harmonic1 = hoomd.md.angle.harmonic()
harmonic1.angle_coeff.set('polymer', k=kd, t0=3.14)

##########Set langevin integrator##################################################
integrator =  hoomd.md.integrate.langevin(group = all, kT = kbT,seed =seed3)

########################################################################
#           Run the simulation
############################################################################
hoomd.run(5e6);
########################################################################


####1. compress the system########

#### Take a snapshot from previous simulation############
final=  system.take_snapshot(particles=True,bonds=True,pairs=True,all= True,dtype = 'float')
############################################
Lx=final.box.Lx
Ly=final.box.Ly
Lz=final.box.Lz

box_resize=hoomd.update.box_resize(Lx = hoomd.variant.linear_interp([(0, Lx), (1e7, Ly_final)]),Ly = hoomd.variant.linear_interp([(0, Ly), (1e7,Ly_final)]),Lz = hoomd.variant.linear_interp([(0, Lz), (1e7, Ly_final)]),period=1000)
hoomd.run(1e7)
box_resize.disable()
###################################################################

nl = hoomd.md.nlist.tree(r_buff = 0.5, check_period = 1) ## in order to accelerate pair force calculation
nl.reset_exclusions(exclusions = []); ## select which interactions should be excluded from the pair interaction calculation

######Equilibration using WCA interactions####################################
wca = hoomd.md.pair.lj(r_cut= rcut,nlist = nl)

wca.set_params(mode='shift') ##shift the potential to make sure that it is 0 at the cutoff
wca.pair_coeff.set('A','A', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut, alpha = a)
wca.pair_coeff.set('B','B', epsilon = epsilon, sigma =sigma[1][1],r_cut = sigma[1][1]*rcut, alpha = a)
wca.pair_coeff.set('C','C', epsilon = epsilon, sigma =sigma[2][2],r_cut = sigma[2][2]*rcut, alpha = a)
wca.pair_coeff.set('D','D', epsilon = epsilon, sigma =sigma[3][3],r_cut = sigma[3][3]*rcut, alpha = a)
wca.pair_coeff.set('A','B', epsilon = epsilon, sigma =sigma[0][1],r_cut = sigma[0][1]*rcut, alpha = a)
wca.pair_coeff.set('A','C', epsilon = epsilon, sigma =sigma[0][2],r_cut = sigma[0][2]*rcut, alpha = a)
wca.pair_coeff.set('A','D', epsilon = epsilon, sigma =sigma[0][3],r_cut = sigma[0][3]*rcut, alpha = a)
wca.pair_coeff.set('B','C', epsilon = epsilon, sigma =sigma[1][2],r_cut = sigma[1][2]*rcut, alpha = a)
wca.pair_coeff.set('B','D', epsilon = epsilon, sigma =sigma[1][3],r_cut = sigma[1][3]*rcut, alpha = a)
wca.pair_coeff.set('C','D', epsilon = epsilon, sigma =sigma[2][3],r_cut = sigma[2][3]*rcut, alpha = a)

##################################################################################

hoomd.md.integrate.mode_standard(dt= dt)
###Set harmonic bonds ############################################################
harmonic= hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('polymer', k =ka, r0 = r0)

########Choosing Harmonic angle potential#################
harmonic1 = hoomd.md.angle.harmonic()
harmonic1.angle_coeff.set('polymer', k=kd, t0=3.14)

##########Set langevin integrator##################################################

hoomd.run(5e6);


####2. unwrap the chain to positive y########

def adjust_chain_and_box(N_trimer, n_trimer, N_RNA, n_RNA, Ly_final):

    trimer_chains = [[i for i in range(j * N_trimer, (j + 1) * N_trimer)] for j in range(n_trimer)]
    RNA_chains = [[i for i in range(N_trimer * n_trimer + j * N_RNA, N_trimer * n_trimer + (j + 1) * N_RNA)] for j in range(n_RNA)]

    # Assuming 'system' is your simulation system and accessible here
    global system
    snapshot = system.take_snapshot(particles=True,bonds=True,pairs=True,all= True,dtype = 'float')

    chains = trimer_chains + RNA_chains  # Combine trimer and RNA chains
    Ly = snapshot.box.Ly
    half_Ly = Ly / 2.0

    for chain in chains:
        # Get y positions of the current chain
        y_positions = [snapshot.particles.position[i][1] for i in chain]
        y_positions1 = [snapshot.particles.position[i][1] for i in chain]
        # Detect if the chain crosses the boundary by checking for a large gap between successive particles
        count_cross=0
        for i in range(len(chain)-1):
            #count_cross=0
            if abs(y_positions[i+1] - y_positions[i]) > half_Ly or abs(y_positions1[i+1] - y_positions1[i]) > half_Ly:
                if (y_positions1[i] - y_positions1[i+1])>half_Ly:
                    count_cross+=1
                elif (y_positions1[i+1] - y_positions1[i])>half_Ly:
                    count_cross-=1
                snapshot.particles.position[chain[i+1]][1] += Ly*count_cross
                y_positions[i+1]+=Ly*count_cross

    # Apply the adjusted positions back to the system
    system.restore_snapshot(snapshot)
        
    # Now adjust the box size if needed
    new_Ly = Ly_final
    system.box = hoomd.data.boxdim(Lx=snapshot.box.Lx, Ly=new_Ly, Lz=snapshot.box.Lz)


# ###################################################################

final=  system.take_snapshot(particles=True,bonds=True,pairs=True,all= True,dtype = 'float')

Lx=final.box.Lx
Ly=final.box.Ly
Lz=final.box.Lz
####################################################################
position = final.particles.position
bonds = final.bonds.group
bondstype = final.bonds.typeid
angles = final.angles.group
anglestype = final.angles.typeid
typeid = final.particles.typeid
image = final.particles.image
vel=final.particles.velocity

context = hoomd.context.initialize("--mode=gpu")

snapshot = hoomd.data.make_snapshot(N=rna_tot+trimer_tot,box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),particle_types=['A','B','C','D'],bond_types=['polymer'],angle_types=['polymer']);

####Read into system variables###############
snapshot.bonds.resize( re_size )
snapshot.angles.resize( re_size_angle )
snapshot.particles.position[:]=position[:]
snapshot.particles.image[:]=image[:]
snapshot.particles.typeid[:]=typeid[:]
snapshot.particles.velocity[:]=vel[:]
snapshot.bonds.group[:]=bonds[:]
snapshot.bonds.typeid[:]=bondstype[:]
snapshot.angles.group[:]=angles[:]
snapshot.angles.typeid[:]=anglestype[:]

system = hoomd.init.read_snapshot(snapshot);############ initialize the system with previously run configuration###
all = hoomd.group.all()         ###Create group#########


#################################################################################################
#               Force initialization
################################################################################################
####Create neighborlist#############################
nl = hoomd.md.nlist.tree(r_buff = 0.5, check_period = 1)
nl.reset_exclusions(exclusions = []);
####################################################

#####Setting WCA interacion coefficients between monomers########
wca = hoomd.md.pair.lj(r_cut= rcut,nlist = nl)

wca.set_params(mode='shift')
wca.pair_coeff.set('A','A', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut, alpha =a)
wca.pair_coeff.set('B','B', epsilon = epsilon, sigma =sigma[1][1],r_cut = sigma[1][1]*rcut, alpha =a)
wca.pair_coeff.set('C','C', epsilon = epsilon, sigma =sigma[2][2],r_cut = sigma[2][2]*rcut, alpha =a)
wca.pair_coeff.set('D','D', epsilon = epsilon, sigma =sigma[3][3],r_cut = sigma[3][3]*rcut, alpha =a)
wca.pair_coeff.set('A','B', epsilon = epsilon, sigma =sigma[0][1],r_cut = sigma[0][1]*rcut, alpha = a)
wca.pair_coeff.set('A','C', epsilon = epsilon, sigma =sigma[0][2],r_cut = sigma[0][2]*rcut, alpha = a)
wca.pair_coeff.set('A','D', epsilon =0.0 , sigma =sigma[0][3],r_cut = sigma[0][3]*rcut, alpha = a)
wca.pair_coeff.set('B','C', epsilon = epsilon, sigma =sigma[1][2],r_cut = sigma[1][2]*rcut, alpha = a)
wca.pair_coeff.set('B','D', epsilon = epsilon, sigma =sigma[1][3],r_cut = sigma[1][3]*rcut, alpha = a)
wca.pair_coeff.set('C','D', epsilon = epsilon, sigma =sigma[2][3],r_cut = sigma[2][3]*rcut, alpha = a)
#####################################################################

######Setting coefficients for stick potential###############
table = hoomd.md.pair.table(width=10000, nlist=nl)

table.pair_coeff.set('A', 'A', func=sticky, rmin=rmin*sigma[0][0], rmax=rmax*sigma[0][0], coeff=dict(epsilon=0, sigma=sigma[0][0]))
table.pair_coeff.set('D', 'D', func=sticky, rmin=rmin*sigma[3][3], rmax=rmax*sigma[3][3], coeff=dict(epsilon=0, sigma=sigma[3][3]))
table.pair_coeff.set('C', 'C', func=sticky, rmin=rmin*sigma[2][2], rmax=rmax*sigma[2][2], coeff=dict(epsilon=0, sigma=sigma[2][2]))
table.pair_coeff.set('B', 'B', func=sticky, rmin=rmin*sigma[1][1], rmax=rmax*sigma[1][1], coeff=dict(epsilon=0, sigma=sigma[1][1]))
table.pair_coeff.set('A', 'B', func=sticky, rmin=rmin*sigma[0][1], rmax=rmax*sigma[0][1], coeff=dict(epsilon=0, sigma=sigma[0][1]))
table.pair_coeff.set('A', 'C', func=sticky, rmin=rmin*sigma[0][2], rmax=rmax*sigma[0][2], coeff=dict(epsilon=0, sigma=sigma[0][2]))
table.pair_coeff.set('A', 'D', func=sticky, rmin=rmin*sigma[0][3], rmax=rmax*sigma[0][3], coeff=dict(epsilon=epsilon1, sigma=sigma[0][3]))
table.pair_coeff.set('B', 'C', func=sticky, rmin=rmin*sigma[1][2], rmax=rmax*sigma[1][2], coeff=dict(epsilon=0, sigma=sigma[1][2]))
table.pair_coeff.set('B', 'D', func=sticky, rmin=rmin*sigma[1][3], rmax=rmax*sigma[1][3], coeff=dict(epsilon=0, sigma=sigma[1][3]))
table.pair_coeff.set('C', 'D', func=sticky, rmin=rmin*sigma[2][3], rmax=rmax*sigma[2][3], coeff=dict(epsilon=0, sigma=sigma[2][3]))
########################################################################
hoomd.md.integrate.mode_standard(dt= dt)
########Choosing Harmonic bonds#################
harmonic= hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('polymer', k =ka, r0 = r0)
########Choosing Harmonic angle potential#################
harmonic1 = hoomd.md.angle.harmonic()
harmonic1.angle_coeff.set('polymer', k=kd, t0=3.14)
#######Integrate using langevin dynamics############
integrator =  hoomd.md.integrate.langevin(group = all, kT = kbT,seed =seed3)
##########################################################################################
#                   Write output
###########################################################################################

filename1="./kd_"+str(kd)+"/log_cnf_"+str(sample)+"_kd_"+str(kd)+"_numTrimers_"+str(int(N_trimer1*N_trimer2))+"_numRNA_"+str(int(N_rna1*N_rna2))+"_kbT_"+str(kbT)+"_Ly_"+str(Ly_final)+"_2.txt"
hoomd.analyze.log(filename = filename1,quantities = ['potential_energy','bond_harmonic_energy','pair_lj_energy','pair_table_energy','temperature','pressure'], period=1e4,overwrite=True)

#filename2="trajectory_cnf_"+str(sample)+"_rna_dens_"+str(density_rna)+"_trimer_dens_"+str(density_trimers)+"_kbT_"+str(kbT)+".gsd"
filename2="./kd_"+str(kd)+"/trajectory_cnf_"+str(sample)+"_kd_"+str(kd)+"_numTrimers_"+str(int(N_trimer1*N_trimer2))+"_numRNA_"+str(int(N_rna1*N_rna2))+"_Ly_"+str(Ly_final)+"_2.gsd"

hoomd.dump.gsd(filename2, period = 1e6, group = all,overwrite =True,dynamic=['momentum'])


########################################################################
#           Run the simulation
############################################################################
hoomd.run(2e8);
#

