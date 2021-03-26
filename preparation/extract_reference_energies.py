from NNSubsampling import subsampling
import numpy as np
import h5py
import pickle
import glob
import os
import csv
from sklearn.linear_model import LinearRegression

def load_system_ref(hdf5_filename):
    system = hdf5_filename.split("_")[0]
    atoms_count = np.zeros(4,dtype=int)
    with h5py.File(hdf5_filename,'r') as data:
        print(list(data.keys()))
        metadata_grp = data["metadata"]
        ref_energy = np.array(metadata_grp["CCSDT_0K_energy"])[0]
        atomic_numbers = np.array(metadata_grp["atomic_numbers"])
    
    for atomic_number in atomic_numbers:
        if atomic_number == 1:
            atoms_count[0] += 1
        if atomic_number == 6:
            atoms_count[1] += 1
        if atomic_number == 7:
            atoms_count[2] += 1
        if atomic_number == 8:
            atoms_count[3] += 1

    
    return system, atoms_count, ref_energy

atoms_count_list = []
system_list = []
ref_energy_list = []

for hdf5_filename in glob.glob("./raw_data_files/*.h5"):
    print("start syste: {}".format(hdf5_filename))
    try:
        system, atoms_count, ref_energy = load_system_ref(hdf5_filename)
        system_list.append(system)
        atoms_count_list.append(atoms_count)
        ref_energy_list.append(ref_energy)
    except:
        print("error, file not exist?")
    print("end")


reg = LinearRegression(fit_intercept = False).fit(np.array(atoms_count_list), np.array(ref_energy_list))
coef = reg.coef_
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