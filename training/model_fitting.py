import numpy as np
import pickle
from sklearn.decomposition import PCA
import h5py
from pykdtree.kdtree import KDTree
from sklearn.linear_model import LinearRegression
import csv


def get_mcsh_order_group(mcsh_order):
    group_dict = {0:[1],\
                  1:[1],\
                  2:[1,2],\
                  3:[1,2,3],\
                  4:[1,2,3,4],\
                  5:[1,2,3,4,5]}
    return group_dict[mcsh_order]

def get_feature_list_legendre(max_mcsh_order, max_legendre_order, max_r):
    result = []
    for mcsh_order in range(max_mcsh_order+1):
        for group_num in get_mcsh_order_group(mcsh_order):
            for legendre_order in range(max_legendre_order+1):
                result.append("{}_{}_{:.6f}_Legendre_{}".format(mcsh_order, group_num, max_r, legendre_order))
    return result


def load_system(system, functional, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_ORDER, MCSH_MAX_R, directory = "/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/DFT_correction_scheme/preparation/raw_data_files/"):


    hdf5_filename = directory + "{}_MCSHLegendre_{}_{:.6f}_{}.h5"\
                             .format(system,MCSH_MAX_ORDER, MCSH_MAX_R, MCSH_RADIAL_MAX_ORDER)



    feature_list = get_feature_list_legendre(MCSH_MAX_ORDER, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_R)
    num_features = len(feature_list) + 1


    with h5py.File(hdf5_filename,'r') as data:
        functional_database_group = data["functional_database"]
        functional_group = functional_database_group[functional]

        feature_grp  = functional_group['feature']
        metadata_grp = functional_group['metadata']
        grid_pts = np.array(metadata_grp["FD_GRID"])
        num_grid_pts = grid_pts[0] * grid_pts[1] * grid_pts[2]
        feature_arr = np.zeros((num_grid_pts, num_features))

        rho = np.array(feature_grp["rho"])
        feature_arr[:,0] = rho.flatten()
        for i, feature in enumerate(feature_list):
            temp_feature = np.array(feature_grp[feature])
            feature_arr[:,i+1] = temp_feature.flatten()
    return feature_arr

def partition(data, refdata): 
    kd_tree = KDTree(refdata,leafsize=6)   
    distances, indices = kd_tree.query(data, k=1)
    
    indices, counts = np.unique(indices, return_counts=True)
    count_arr = np.zeros(len(refdata))
    
    for i, index in enumerate(indices):
        count_arr[index] = counts[i] #how many grid points have a particular environment 

    max_distance = np.max(distances)
    print("max distance")
    print(max_distance)
    
    return count_arr, max_distance

def load_system_info(filename): #dictionary to store the energy for each system
    # system_list = []
    energy_dict = {}
    skip = True
    with open(filename) as fp: 
        Lines = fp.readlines() 
        for line in Lines: 
            if skip:
                skip = False
                continue
            temp = line.strip().split()
            system = temp[-1]
            #system = temp[-1]split("_files/")[1]
            energy = float(temp[-2]) #error is stored in energy 
            energy_dict[system] = energy
    return energy_dict

#model_setup is a dictionary with keys and values
model_setup = {"refdata_filename": "Overall_subsampled_1.0_True.pickle", \
                "functional": "GGA_PBE", \
                "MCSH_RADIAL_MAX_ORDER": 5, \
                "MCSH_MAX_ORDER" :3, \
                "MCSH_MAX_R":3.0, \
                "PCA": False,\
                "PCA_n_components":15}

model = {"model_setup": model_setup}

functional = model_setup["functional"] #value of functional 

ref_energy_dict = load_system_info("/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/DFT_correction_scheme/preparation/ref_energy_regression_error.csv")
print("reference energy dictionary\n")
print(ref_energy_dict)
functional_energy_dict = load_system_info("/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/DFT_correction_scheme/preparation/{}_energy_regression_error.csv".format(functional))
print("functional energy dictionary\n")
print(functional_energy_dict)
systems = ref_energy_dict.keys()
print("systems\n")
print(systems)
subsampled_filename = model_setup["refdata_filename"]
refdata = pickle.load(open(subsampled_filename, "rb" )) #subsampled file
refdata = np.vstack((refdata, np.zeros(len(refdata[0]))))
print("refdata\n")
print(refdata)



if model_setup["PCA"]:
    pca = PCA(n_components=model_setup["PCA_n_components"])
    pca.fit(refdata)
    refdata_transformed= pca.transform(refdata)
    model["PCA_model"] = pca
else:
    refdata_transformed = refdata

model["refdata_transformed"] = refdata_transformed

count_arr = np.zeros((len(systems), len(refdata)))
target = np.zeros(len(systems))
max_distance = 0

for i, system in enumerate(systems):
    # try:
    print("start processing system {}".format(system))
    temp_feature_arr = load_system( system, \
                                    functional = functional, \
                                    MCSH_RADIAL_MAX_ORDER = model_setup["MCSH_RADIAL_MAX_ORDER"], \
                                    MCSH_MAX_ORDER = model_setup["MCSH_MAX_ORDER"], \
                                    MCSH_MAX_R = model_setup["MCSH_MAX_R"])
    if model_setup["PCA"]:
        feature_arr_transformed = model["PCA_model"].transform(temp_feature_arr)
    else:
        feature_arr_transformed = temp_feature_arr
    count_arr[i], temp_max_distance = partition(feature_arr_transformed, refdata_transformed)
    max_distance = max(max_distance, temp_max_distance)
    target[i] = ref_energy_dict[system] - functional_energy_dict[system] #target is the difference between CCSDT(T) energy and SPARC energy
    # except:
    #     print("ERROR: system {} not processed".format(system))

model["max_distance"] = max_distance

with open('{}_count_array.csv'.format(functional), 'w', newline='') as csvfile:#count_array consists of how many points are there in each environment 
    writer2 = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i, system in enumerate(systems):
        writer2.writerow([i, system] + count_arr[i].tolist() + [target[i]])




reg = LinearRegression(fit_intercept = False).fit(count_arr, target)
coef = reg.coef_

model["regression_model"] = reg

with open('{}_ccsdt_correction_result.csv'.format(functional), 'w', newline='') as csvfile:
    writer2 = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(refdata)):
        writer2.writerow([i, coef[i]]) #each row will have correction corresponding to each environment from overall subsampled file

#actual correction is GGA_PBE_ccsdt_correction_result 

prediction = reg.predict(count_arr)
error = target - prediction 
with open('{}_ccsdt_correction_result_error.csv'.format(functional), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['energy','predicted_energy','error','system'])
    for i, system in enumerate(systems):
        temp = [target[i], prediction[i],error[i], system]
        writer.writerow(temp)

with open('{}_correction_amount.csv'.format(functional), 'w', newline='') as csvfile:
    writer2 = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i, system in enumerate(systems):
        writer2.writerow([i, system] + np.multiply(count_arr[i], coef).tolist() + [prediction[i]])

print("end")
print(model)
pickle.dump( model, open( "{}_functional_based_model.pickle".format(functional), "wb" ) )