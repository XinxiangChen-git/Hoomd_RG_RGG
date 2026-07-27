import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import time
import numpy
import random
import math
import sys
import datetime
import time

# Parameters
sample=sys.argv[1]
n_polymer = int(sys.argv[2])
N_length = int(sys.argv[3])
f_sticker=int(sys.argv[4])
L_final = float(sys.argv[5])

real_l=int((N_length-f_sticker)/(f_sticker+1))

dt=0.005
eps = 1.0  ## Interaction strength for WCA
rcut = math.pow(2.,1./6.) #cut- off of wca potential
sigma = 1
sigmas= 1 # samll radius>=(sqrt(2)-1)sigma=0.414sigma
ka = 30 #Harmonic bond strength
# kd = 5 # Harmonic angle potential
r0 = math.pow(2.,1./6.) ##Equilibrium bond length
kbT = 1
eps1= float(sys.argv[6])  ## Interaction strength of sticky potential
rmax = 0.5 #cut-off of sticky potential
rmin = 0.0 #minimum distance to which force is calculated
theta0 = numpy.pi

sidebead=2
N=n_polymer*(N_length+sidebead*f_sticker)
re_size = n_polymer*(N_length-1+sidebead*f_sticker)
# re_size_angle = n_polymer*(N_length-2)

seed3 = random.randint(1,60000)


# Define particle positions, types, and bonds for the core and arms

typeid = numpy.zeros(N,dtype=int)
positions = numpy.zeros((N,3),dtype=float)
diameter =  numpy.zeros(N,dtype=float)
mass =  numpy.zeros(N,dtype=float)
bondsgroup = numpy.zeros((re_size,2),dtype=int)
bondstypeid = numpy.zeros(re_size,dtype=int)

# anglesgroup = numpy.zeros((re_size_angle,3),dtype=int)
# anglestypeid = numpy.zeros(re_size_angle,dtype=int)


# Generate arm positions and set the end of each arm to a unique sticker type
spacer_types=0
sticker_types = 1  # Different types for each arm end
sticker_A_types = 2  # Different types for each arm end

# Set up 3D grid spacing for initial placement to minimize overlaps

spacing = 5.0
Mono_space = r0

grid_size = int(numpy.ceil(numpy.sqrt(n_polymer)))
Lx = 2 * spacing + (grid_size - 1) * spacing
Ly = 2 * spacing + (grid_size - 1) * spacing
Lz = (N_length - 1) * Mono_space + 2 * spacing
    
x_min = -Lx / 2 + spacing
y_min = -Ly / 2 + spacing
z_min = -Lz / 2 + spacing


index=0
index_bond=-1
index_angle=-1
for poly_idx in range(n_polymer):

    x = x_min + (poly_idx % grid_size) * spacing
    y = y_min + (poly_idx // grid_size) * spacing
    z_start = z_min
    
    positions[index] = numpy.array([x, y, z_start])
    diameter[index] = 1.0  #diameter of trimer monomers
    mass[index]=1.0  #mass of the trimer monomer
    typeid[index]=spacer_types  # Core particle type

    for j in range(1, N_length):
        index+=1
        index_bond+=1
        monomer_position=numpy.array([x, y, z_start + j * Mono_space])
        positions[index]=monomer_position
        if (j-(N_length+real_l-N_length%(real_l+1)))%(real_l+1)==0 :
            typeid[index]=sticker_types
            diameter[index] = 1.0
        else:
            typeid[index]=spacer_types
            diameter[index] = 1.0
        mass[index]=1.0

        bondsgroup[index_bond,0] = index-1
        bondsgroup[index_bond,1] = index
        if (j-(N_length+real_l-N_length%(real_l+1)))%(real_l+1)==0 or j%(real_l+1)==0:
            bondstypeid[index_bond]=1
        else:
            bondstypeid[index_bond]=0
        
        # if j>1:
        #     index_angle+=1
        #     anglesgroup[index_angle,0] = index-2
        #     anglesgroup[index_angle,1] = index-1
        #     anglesgroup[index_angle,2] = index
        #     anglestypeid[index_angle]=0

    index+=1


index=index-1
for poly_idx in range(n_polymer):
    for sticker_idx in range(f_sticker):
        core_idx=poly_idx*N_length+(real_l+1)*(sticker_idx+1)-1
        for siii in range(sidebead):
            angle0 = 2 * numpy.pi * siii / sidebead
            arm_direction = numpy.array([numpy.cos(angle0), numpy.sin(angle0), 0.0])
            positions[index+1]=positions[core_idx]+arm_direction
            bondsgroup[index_bond+1,0] = core_idx
            bondsgroup[index_bond+1,1] = index+1
            bondstypeid[index_bond+1]=1
            diameter[index+1] = 1.0
            typeid[index+1]=sticker_A_types
            mass[index+1]=1.0
            index+=1
            index_bond+=1


class Status():

    def __init__(self, sim):
        self.sim = sim
    
    @property
    def seconds_remaining(self):
        try:
            return (self.sim.final_timestep - self.sim.timestep) / self.sim.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))



gpu = hoomd.device.GPU()
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=gpu,seed=seed3)


type_= ['A','B','C']
snap = gsd.hoomd.Snapshot()
snap.particles.N = N
snap.particles.position = positions
snap.particles.types = type_
snap.particles.typeid = typeid
snap.particles.diameter = diameter
snap.particles.mass = mass
snap.configuration.box = [1.*Lx, 1.*Ly, 1.*Lz, 0, 0, 0]
snap.bonds.N=re_size
snap.bonds.types = ['polymer0','polymer1']
snap.bonds.group = bondsgroup
snap.bonds.typeid = bondstypeid

# snap.angles.N=re_size_angle
# snap.angles.types = ['polymer']
# snap.angles.group = anglesgroup
# snap.angles.typeid = anglestypeid


sim.create_state_from_snapshot(snap)


integrator = hoomd.md.Integrator(dt=dt,integrate_rotational_dof=True)
ntree = hoomd.md.nlist.Tree(buffer=.5,exclusions=['bond','body','special_pair'])



# Define interactions (e.g., Lennard-Jones potential)
wca = hoomd.md.pair.LJ(nlist =ntree,default_r_cut = rcut*sigma,mode = "shift")
wca.params[('A','A')] = {'sigma':sigma,'epsilon': eps}
wca.params[('A','B')] = {'sigma':(sigma+sigmas)/2,'epsilon': eps}
wca.params[('A','C')] = {'sigma':sigma,'epsilon': eps}
wca.params[('B','B')] = {'sigma':sigmas,'epsilon': eps}
wca.params[('B','C')] = {'sigma':(sigma+sigmas)/2,'epsilon': eps}
wca.params[('C','C')] = {'sigma':sigma,'epsilon': eps}

wca.r_cut[("A", "A")] = 2**(1/6)*sigma
wca.r_cut[("A", "B")] = 2**(1/6)*((sigma+sigmas)/2)
wca.r_cut[("A", "C")] = 2**(1/6)*sigma
wca.r_cut[("B", "B")] = 2**(1/6)*sigmas
wca.r_cut[("B", "C")] = 2**(1/6)*((sigma+sigmas)/2)
wca.r_cut[("C", "C")] = 2**(1/6)*sigma



###Defining sticky potential#####
def sticky( rmin, rmax, epsilon, sigma):
    r=numpy.linspace(rmin,rmax,100000)
    V = -epsilon*(numpy.cos(2*numpy.pi*r/(sigma))+1)
    F = -2*numpy.pi*epsilon*(numpy.sin(2*numpy.pi*r/(sigma)))/(sigma)
    return (V, F)

V,F=sticky(rmin*sigmas,rmax*sigmas,eps1,sigmas)
U0 = numpy.zeros(len(V),dtype=float)
F0 = numpy.zeros(len(F),dtype=float)

tablep=hoomd.md.pair.Table(ntree, default_r_cut=0.5)
tablep.params[('A','A')] = dict(r_min=0, U=U0, F=F0)
tablep.r_cut[('A','A')] = 0
tablep.params[('A','B')] = dict(r_min=0, U=U0, F=F0)
tablep.r_cut[('A','B')] = 0
tablep.params[('A','C')] = dict(r_min=0, U=U0, F=F0)
tablep.r_cut[('A','C')] = 0
tablep.params[('B','B')] = dict(r_min=0, U=V, F=F)
tablep.r_cut[('B','B')] = 0.5*sigmas
tablep.params[('B','C')] = dict(r_min=0, U=U0, F=F0)
tablep.r_cut[('B','C')] = 0
tablep.params[('C','C')] = dict(r_min=0, U=U0, F=F0)
tablep.r_cut[('C','C')] = 0


fenewca = hoomd.md.bond.FENEWCA()
fenewca.params['polymer0'] = dict(k=ka/(sigma*sigma), r0= 1.5*sigma, epsilon=eps, sigma=sigma, delta=0.0)
fenewca.params['polymer1'] = dict(k=ka/(((sigma+sigmas)/2)*((sigma+sigmas)/2)), r0= 1.5*(sigma+sigmas)/2, epsilon=eps, sigma=(sigma+sigmas)/2, delta=0.0)

# harmonic_angle = hoomd.md.angle.Harmonic()
# harmonic_angle.params['polymer'] = dict(k=kd, t0=theta0)

# Run a brief simulation to relax the structure
integrator = hoomd.md.Integrator(dt=dt,integrate_rotational_dof=False) ##Defining instance of integrator
ntree = hoomd.md.nlist.Tree(buffer=.51,exclusions=['bond','body','special_pair'])## neighborlist creation
groupall = hoomd.filter.All()  ###Selecting group
integrator.forces.append(wca) ##Adding excluded volume interaction
integrator.forces.append(fenewca) ##Adding fenewca bonds
# integrator.forces.append(harmonic_angle)

nvt = hoomd.md.methods.Langevin(kT = kbT, filter = groupall)  ##Adding intergrator method


########Operator definition#####################################################
###############################################################################
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter= groupall)

logger = hoomd.logging.Logger(categories=['scalar', 'string','scalar'])
logger.add(sim, quantities=['timestep', 'tps'])
logger.add(thermodynamic_properties,quantities=['kinetic_temperature'])

status = Status(sim)
logger[('Status', 'etr')] = (status, 'etr', 'string')

table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=500),logger=logger)

#### Adding the operators and then the integrator to simulation#################
################################################################################
sim.operations.writers.append(table)

sim.operations.computes.append(thermodynamic_properties)

integrator.methods.append(nvt)

sim.operations.integrator = integrator


sim.run(1000000)


ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=1000000)
initial_box = sim.state.box
final_box = [1.*L_final, 1.*L_final, 1.*L_final, 0, 0, 0]  # make a copy of initial_box
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box,box2=final_box,variant=ramp,trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)
sim.run(1000000)
sim.operations.updaters.remove(box_resize)


#thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter= groupall)
log1 = hoomd.logging.Logger(categories=['scalar', 'scalar','scalar','scalar','scalar'])
log1.add(sim, quantities=['timestep'])
log1.add(thermodynamic_properties,quantities=['kinetic_energy','potential_energy','kinetic_temperature','pressure'])

filename1="log_cnf_"+str(sample)+"_epsilon1_"+str(eps1)+"_polymer_"+str(int(n_polymer))+"_length_"+str(N_length)+"_sticker_"+str(f_sticker)+"_sigmas_"+str(sigmas)+"_boxsize_"+str(L_final)+".txt"

file1 = open(filename1, mode='w', newline='\n')
table_file = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=1000000),logger=log1, output=file1)
sim.operations.writers.append(table_file)

################################################################################
filename2="trajectory_cnf_"+str(sample)+"_epsilon1_"+str(eps1)+"_polymer_"+str(int(n_polymer))+"_length_"+str(N_length)+"_sticker_"+str(f_sticker)+"_sigmas_"+str(sigmas)+"_boxsize_"+str(L_final)+".gsd"


gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(2000000), filename=filename2,mode='wb',dynamic=['momentum','property'])

sim.operations.writers.append(gsd_oper)

###############################################################################
wca.params[('B','B')] = {'sigma':sigmas,'epsilon': 0} ##Change interaction for 'B'-'B'
wca.params[('B','C')] = {'sigma':(sigma+sigmas)/2,'epsilon': 0} ##Change interaction for 'B'-'C'

integrator.forces.append(tablep) ##Add sticky potetial

sim.run(300000000)

