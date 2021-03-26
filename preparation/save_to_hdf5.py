import h5py
import numpy as np
import csv
import os
import sys
from ase.atom import Atom
from ase import Atoms

def get_mcsh_order_group(mcsh_order):
    group_dict = {0:[1],\
                  1:[1],\
                  2:[1,2],\
                  3:[1,2,3],\
                  4:[1,2,3,4]}
    return group_dict[mcsh_order]

def get_feature_list_legendre(max_mcsh_order, max_legendre_order, max_r):
    result = []
    for mcsh_order in range(max_mcsh_order+1):
        for group_num in get_mcsh_order_group(mcsh_order):
            for legendre_order in range(max_legendre_order+1):
                result.append("{}_{}_{:.6f}_Legendre_{}".format(mcsh_order, group_num, max_r, legendre_order))
    return result

def read_feature_file(filename, grid_dimensions):
    result = np.zeros(grid_dimensions)
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvfile:
            temp = row.split(',')
            #print(float(temp[3]))
            result[int(temp[0]), int(temp[1]), int(temp[2])] = float(temp[3])
    return result

def read_coordinate_file(filename, grid_dimensions):
    x = np.zeros(grid_dimensions)
    y = np.zeros(grid_dimensions)
    z = np.zeros(grid_dimensions)
    rho = np.zeros(grid_dimensions)
    Nx = grid_dimensions[0]
    Ny = grid_dimensions[1]
    Nz = grid_dimensions[2]
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for index, row in enumerate(csvfile):
            # i = index % (Nx)
            # j = (index // (Nx)) % Ny
            # k = index // (Nx*Ny)
            temp = row.split(',')
            i = int(temp[0])
            j = int(temp[1])
            k = int(temp[2])
            x[i,j,k] = float(temp[3])
            y[i,j,k] = float(temp[4])
            z[i,j,k] = float(temp[5])
            rho[i,j,k] = float(temp[6])
    return x,y,z,rho

def read_converged_exc_file(filename, grid_dimensions):
    exc = np.zeros(grid_dimensions)
    Nx = grid_dimensions[0]
    Ny = grid_dimensions[1]
    Nz = grid_dimensions[2]
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for index, row in enumerate(csvfile):
            temp = row.split(',')
            i = int(temp[0])
            j = int(temp[1])
            k = int(temp[2])
            exc[i,j,k] = float(temp[3])
    return exc

def read_ref_energy(system_name, filename = "energy"):
    with open(filename) as fp: 
        for line in fp: 
            # temp = line.strip().split()
            if line.strip().startswith("Energy at 0K"):
                CCSDT_0K_energy = float(line.strip().split()[-1]) 
    return CCSDT_0K_energy

def read_ref_coords(system_name, filename = "coords"):

    skip = True
    list_atoms = []
    with open(filename) as fp: 
        for line in fp: 
            if skip:
                skip = False
                continue
            temp = line.strip().split()
            list_atoms.append(Atom(temp[0], (float(temp[1]), float(temp[2]), float(temp[3]))))

    system = Atoms(list_atoms)

    return system.get_atomic_numbers(), system.get_positions()

def print_hdf5_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.iteritems():
        print("    %s: %s" % (key, val))




def process_one_functional(system_name, functional, hdf5_filename):

    current_out_file = './{}/{}/sprc-calc.out'.format(system_name, functional)

    output_file = []
    with open(current_out_file) as fp:
        for cnt, line in enumerate(fp):
            temp = line.split()
            if len(temp) > 0:
                output_file.append(line.split())

    U = []
    cell = np.zeros(3)
    FD_GRID = [0,0,0]
    MCSH_RADIAL_TYPE = 0
    MCSH_RADIAL_MAX_ORDER = 0
    MCSH_MAX_ORDER = 0
    MCSH_MAX_R = 0.0
    for index, line in enumerate(output_file):
        #print(line)
        if line[0] == 'LATVEC:':
            U.append(list(map(float,output_file[index + 1])))
            U.append(list(map(float,output_file[index + 2])))
            U.append(list(map(float,output_file[index + 3])))
        elif line[0] == "CELL:":
            cell[0] = float(line[1])
            cell[1] = float(line[2])
            cell[2] = float(line[3])
        elif line[0] == "FD_GRID:":
            FD_GRID[0] = int(line[1])
            FD_GRID[1] = int(line[2])
            FD_GRID[2] = int(line[3])
        elif line[0] == "MCSH_RADIAL_TYPE:":
            MCSH_RADIAL_TYPE = int(line[1])
        elif line[0] == "MCSH_MAX_ORDER:":
            MCSH_MAX_ORDER = int(line[1])
        elif line[0] == "MCSH_RADIAL_MAX_ORDER:":
            MCSH_RADIAL_MAX_ORDER = int(line[1])
        elif line[0] == "MCSH_MAX_R:":
            MCSH_MAX_R = float(line[1])
        elif line[0] == "MCSH_R_STEPSIZE:":
            MCSH_R_STEPSIZE = float(line[1])
    U = np.array(U)
    cell = np.array(cell)

    if MCSH_RADIAL_TYPE == 2:
        feature_list = get_feature_list_legendre(MCSH_MAX_ORDER, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_R)
        # result_filename = "{}-MCSH_feature_Legendre_{}_{:.6f}_{}.h5"\
        #                     .format(system_name,MCSH_MAX_ORDER, MCSH_MAX_R, MCSH_RADIAL_MAX_ORDER)
        # print(result_filename)
        with h5py.File(hdf5_filename,'a') as data:
            if "functional_database" in data:
                functional_database_group = data["functional_database"]
            else:
                functional_database_group = data.create_group("functional_database")
            functional_group = functional_database_group.create_group(functional)
            metadata_grp = functional_group.create_group('metadata')
            feature_grp  = functional_group.create_group('feature')
            for feature in feature_list:
                # print("reading feature: {}".format(feature))
                feature_filename = "./{}/{}/MCSH_feature_{}.csv".format(system_name, functional,feature)
                temp_data = read_feature_file(feature_filename, FD_GRID)
                feature_grp.create_dataset(feature,data=temp_data)
            
            # print("reading coordinate file")
            coordinate_filename = "./{}/{}/Mesh_Coordinates.csv".format(system_name, functional)
            x,y,z,rho = read_coordinate_file(coordinate_filename, FD_GRID)
            feature_grp.create_dataset("x",data=x)
            feature_grp.create_dataset("y",data=y)
            feature_grp.create_dataset("z",data=z)
            feature_grp.create_dataset("rho",data=rho)

            converged_exc_filename = "./{}/{}/Converged_exc_density.csv".format(system_name, functional)
            exc = read_converged_exc_file(converged_exc_filename, FD_GRID)
            feature_grp.create_dataset("exc",data=exc)
            
            metadata_grp.create_dataset("FD_GRID", data=FD_GRID)
            metadata_grp.create_dataset("CELL", data=cell)
            metadata_grp.create_dataset("LATVEC", data=U)

    return


def process_system(system_name, list_of_functionals = ["GGA_PBE","GGA_PBEsol","GGA_RPBE"], MCSH_RADIAL_MAX_ORDER = 5, MCSH_MAX_ORDER = 3, MCSH_MAX_R = 3.0):
    print("\n==========\nstart system: {}".format(system_name))

    system_folder = "./{}/".format(system_name)
    hdf5_filename = "./raw_data_files/{}_MCSHLegendre_{}_{:.6f}_{}.h5"\
                         .format(system_name,MCSH_MAX_ORDER, MCSH_MAX_R, MCSH_RADIAL_MAX_ORDER)

    # system_folder = "./{}/".format(system_name)
    CCSDT_0K_energy = read_ref_energy(system_name, filename = system_folder + "energy")
    atomic_numbers, coords = read_ref_coords(system_name, filename = system_folder + "coords")
    with h5py.File(hdf5_filename,'w') as data:
        metadata_grp = data.create_group("metadata")
        metadata_grp.create_dataset("CCSDT_0K_energy", data=[CCSDT_0K_energy])
        metadata_grp.create_dataset("atomic_numbers", data=atomic_numbers)
        metadata_grp.create_dataset("atomic_coords", data=coords)

    for functional in list_of_functionals:
        process_one_functional(system_name, functional, hdf5_filename)

    print("finish system: {}".format(system_name))

    return

for system in os.listdir('./sample_files'):
    if os.path.exists("./sample_files/{}/energy".format(system)):
        try:
            process_system(system, list_of_functionals = ["GGA_PBE","GGA_RPBE"])
        except:
            print("processing failed for system: {}".format(system))