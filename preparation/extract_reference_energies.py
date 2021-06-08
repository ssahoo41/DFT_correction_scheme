#from NNSubsampling import subsampling
import numpy as np
import h5py
import pickle
import glob
import os
import csv
from sklearn.linear_model import LinearRegression

def load_system_ref(hdf5_filename):
    system_1 = hdf5_filename.split("_MCSH")[0] #this is not right (it saves raw as system name)
    system = system_1.split("raw_data_files/")[1]
    atoms_count = np.zeros(4,dtype=int)# hydrogen, carbon, nitrogen and oxygen 
    print(hdf5_filename)
    with h5py.File(hdf5_filename,'r') as data:
        #print(list(data.keys()))
        metadata_grp = data["metadata"]
        #system = metadata_grp["System_name"]
        functional_database_group = data["functional_database"] #SPARC energy is stored in the metadata of functional database 
        functional_group = functional_database_group["GGA_PBE"]
        functional_meta = functional_group["metadata"]
        ref_energy = np.array(metadata_grp["CCSDT_0K_energy"])[0]
        print(ref_energy)
        atomic_numbers = np.array(metadata_grp["atomic_numbers"])
        sparc_energy = np.array(functional_meta["Total_free_energy"])#[0]
    
    for atomic_number in atomic_numbers:
        if atomic_number == 1:
            atoms_count[0] += 1
        if atomic_number == 6:
            atoms_count[1] += 1
        if atomic_number == 7:
            atoms_count[2] += 1
        if atomic_number == 8:
            atoms_count[3] += 1

    
    return system, atoms_count, ref_energy, sparc_energy

atoms_count_list = []
system_list = []
ref_energy_list = []
sparc_energy_list = [] #we are only taking GGA_PBE energies for test 

for hdf5_filename in glob.glob("./raw_data_files/*.h5"): #will go through all the files in the folder
    
    print("start system: {}".format(hdf5_filename))
    try:
        system, atoms_count, ref_energy, sparc_energy = load_system_ref(hdf5_filename)
        system_list.append(system)
        atoms_count_list.append(atoms_count)
        ref_energy_list.append(ref_energy)
        sparc_energy_list.append(sparc_energy)
    except:
        print("error, file not exist?")
    print("end")

print(system_list)
print(atoms_count_list)
print(ref_energy_list)
print(sparc_energy_list)

#regression for reference energy 
reg = LinearRegression(fit_intercept = False).fit(np.array(atoms_count_list), np.array(ref_energy_list))
coef = reg.coef_
#prediction = reg.predict(np.array(atoms_count_list))
#error = np.array(ref_energy_list) - prediction

atom_list = ['H','C','N','O']
with open('ref_energy_regression.csv', 'w', newline='') as csvfile:
    writer2 = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(atom_list)):
        writer2.writerow([atom_list[i], coef[i]])

prediction = reg.predict(np.array(atoms_count_list))
error = np.array(ref_energy_list) - prediction
with open('ref_energy_regression_error.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['H','C','N','O','energy','predicted_energy','error','system'])
    for i, energy in enumerate(ref_energy_list):
        temp = atoms_count_list[i].tolist() + [energy, prediction[i],error[i], system_list[i]]
        writer.writerow(temp)

#regression for SPARC energy 
reg_SP = LinearRegression(fit_intercept = False).fit(np.array(atoms_count_list), np.array(sparc_energy_list))
coef_SP = reg_SP.coef_
#print(coef_SP)
#prediction_SP = reg_SP.predict(np.array(atoms_count_list))
#print(prediction_SP)
#error = np.array(sparc_energy_list) - prediction_SP
#print(error)

with open('GGA_PBE_energy_regression.csv', 'w', newline='') as csvfile:
    writer3 = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(atom_list)):
        writer3.writerow([atom_list[i], coef_SP[i]]) #coefficient corresponding to each atom 
        
prediction_SP = reg_SP.predict(np.array(atoms_count_list))
error = np.array(sparc_energy_list) - prediction_SP
with open('GGA_PBE_energy_regression_error.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['H','C','N','O','energy','predicted_energy','error','system'])
    for i, energy in enumerate(sparc_energy_list):
        temp = atoms_count_list[i].tolist() + [energy, prediction_SP[i],error[i], system_list[i]]
        writer.writerow(temp) 
