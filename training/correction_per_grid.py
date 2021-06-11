import numpy as np
import pickle
from sklearn.decomposition import PCA
import h5py
from pykdtree.kdtree import KDTree
from sklearn.linear_model import LinearRegression
import csv
import pandas as pd

#function details
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

def load_system(system, functional, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_ORDER, MCSH_MAX_R, directory = "/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/results_folder/preparation_results/prep_final_results/raw_data_files/"):


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
    distances, indices = kd_tree.query(data, k=1) #temp distances will have length of number of gridpoints in the system
    #har data point ka index save hoga and then we will find the unique indices
    
    indices_reduced, counts = np.unique(indices, return_counts=True) #indices reduced will only save unique indices
    count_arr = np.zeros(len(refdata))
    
    for i, index in enumerate(indices_reduced):
        count_arr[index] = counts[i] #how many grid points have a particular environment 

    max_distance = np.max(distances)
    print("max distance")
    print(max_distance)
    return count_arr, max_distance, indices, indices_reduced #indices in increasing order and count of each index

#model details
#model_filename = "/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/results_folder/training_results/cutoff_sig_overall_3.5/GGA_PBE_functional_based_model.pickle"

#model = pickle.load(open( model_filename, "rb" ) )
#model_setup = model["model_setup"]
#max_distance = model["max_distance"]
#reg_model = model["regression_model"]
#refdata_transformed = model["refdata_transformed"]
#print(reg_model.coef_)
#print(reg_model.coef_.shape)

correction_file = "/storage/coda1/p-amedford6/0/shared/rich_project_chbe-medford/medford-share/scratch/correction_scheme_testing/prelim_result/cut_off_sig_3.5_CV/GGA_PBE_1.0_ccsdt_correction_result.csv"
system = "H2CO3"
df = pd.read_csv(correction_file, header=None)

coefs = df[1].tolist()

subsampled_filename = "/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/results_folder/training_results/prelim_training_train_test/overall_subsample_files/Overall_subsampled_3.5_True.pickle"
refdata = pickle.load(open(subsampled_filename, "rb" )) #subsampled file
refdata = np.vstack((refdata, np.zeros(len(refdata[0]))))

refdata_transformed = refdata 

#temp_feature_arr = load_system( system, \
#                                functional = model_setup["functional"], \
#                                MCSH_RADIAL_MAX_ORDER = model_setup["MCSH_RADIAL_MAX_ORDER"], \
#                                MCSH_MAX_ORDER = model_setup["MCSH_MAX_ORDER"], \
#                                MCSH_MAX_R = model_setup["MCSH_MAX_R"])

temp_feature_arr = load_system( system, \
                                functional = "GGA_PBE", \
                                MCSH_RADIAL_MAX_ORDER = 5, \
                                MCSH_MAX_ORDER = 3, \
                                MCSH_MAX_R = 3.0)

#if model_setup["PCA"]:
    #feature_arr_transformed = model["PCA_model"].transform(temp_feature_arr)
#else:
feature_arr_transformed = temp_feature_arr
count_arr, max_distance, indices, indices_reduced = partition(feature_arr_transformed, refdata_transformed) #will get the count and unique indices in the list 

count_arr = count_arr.reshape(-1,1) #number of time each environment appears as compared to reference data

#correction_amount_per_env = reg_model.coef_
correction_amount_per_env = coefs 
correction_amount_per_gridpoint = np.zeros(1000000)

for i, index in enumerate(indices):
    for j in range(len(list(indices_reduced))):
        if index == indices_reduced[j]:
            correction_amount_per_gridpoint[i] = correction_amount_per_env[j]

with open('{}_correction_per_gridpoint.csv'.format(system), 'w', newline='') as csvfile:
    writer2 = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i, index in enumerate(indices):
        writer2.writerow([i, correction_amount_per_gridpoint[i]]) 
