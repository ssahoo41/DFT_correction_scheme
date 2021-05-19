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


def load_system(system, functional, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_ORDER, MCSH_MAX_R, directory = "/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/results_folder/prediction_results/raw_data_files/"):#path to hdf5 file for the 3 systems 


    hdf5_filename = directory + "{}_MCSHLegendre_{}_{:.6f}_{}.h5"\
                             .format(system,MCSH_MAX_ORDER, MCSH_MAX_R, MCSH_RADIAL_MAX_ORDER)



    feature_list = get_feature_list_legendre(MCSH_MAX_ORDER, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_R)
    num_features = len(feature_list) + 1 #all the features and electron density 


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

def partition(data, refdata, max_distance):
    kd_tree = KDTree(refdata,leafsize=6)
    temp_distances, temp_indices = kd_tree.query(data, k=1)


    indices = []
    for i, distance in enumerate(temp_distances): #temp distances is going to contain all grid points for each system, what index does each grid point belong to and its distance from that index 
        if distance[0]< max_distance: #if distance is less than max_distance, distances which are greater than max_distances are the environments that have not been trained on
            indices.append(temp_indices[i])
        else:
            indices.append(-1) #the environments that have not been seen 
    indices = np.array(indices)

    
    indices_reduced, counts = np.unique(indices, return_counts=True)#count of each index and only unique indices
    count_arr = np.zeros(len(refdata)+1) #let's say reference data has 10 unique environments, it will add one more for the environment that training has not come across

    #[-1 -1 0 4 3 3 -1 2 1 1] #this is indices
    #[-1 0 1 2 3 4] #this is indices_reduced
    #[3 1 2 1 2 1] #count of each environment 
    
    for i, index in enumerate(indices_reduced):
        count_arr[index] = counts[i] #the number of points corresponding to each representative environment, last number in count_arr will correspond to the environment which has not been seen

    #[-1,-1,1,2,3,4,1,1,-1]
    #count_array will be 
    return count_arr, indices, indices_reduced #will give all indices in increasing order and count of each index 



model_filename = "/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/results_folder/training_results/GGA_PBE_functional_based_model.pickle" #model file name
system = "CO_box"
#system = "CO_on_Pt_relax"
#system = "Pt_relax"

model = pickle.load( open( model_filename, "wb" ) )
model_setup = model["model_setup"]
max_distance = model["max_distance"]
reg_model = model["regression_model"]
refdata_transformed = model["refdata_transformed"]

#model will give coefficient corresponding to each environment (the count)
#[e1 , e2, e3, ...... en] + [0] = [e1, e2, e3, ......,en, 0] #adding 0 correction corresponding to environment that has not been seen 

temp_feature_arr = load_system( system, \
                                functional = model_setup["functional"], \
                                MCSH_RADIAL_MAX_ORDER = model_setup["MCSH_RADIAL_MAX_ORDER"], \
                                MCSH_MAX_ORDER = model_setup["MCSH_MAX_ORDER"], \
                                MCSH_MAX_R = model_setup["MCSH_MAX_R"])
if model_setup["PCA"]:
    feature_arr_transformed = model["PCA_model"].transform(temp_feature_arr)
else:
    feature_arr_transformed = temp_feature_arr
indices, count_arr, indices_reduced = partition(feature_arr_transformed, refdata_transformed) #will get the count and unique indices in the list 



correction = reg_model.predict(count_arr[:-1]) #correction corresponding to each environment except the last one based on regression model
print("calculated correction is: {}".format(correction))

correction_amount_per_env = [0] + reg_model.coef_ 

correction_amount_per_gridpoint = np.zeros(feature_arr_transformed)

for i, index in enumerate(indices):
    for j in range(len(list(indices_reduced))):
        if index == indices_reduced[j]:
            correction_amount_per_gridpoint[i] = correction_amount_per_env[j]


with open('{}_correction_per_gridpoint.csv'.format(system), 'w', newline='') as csvfile:
    writer2 = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(data)):
        writer2.writerow([i, correction_amount_per_gridpoint[i]]) 