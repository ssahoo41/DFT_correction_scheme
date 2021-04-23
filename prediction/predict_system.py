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
                  4:[1,2,3,4]}
    return group_dict[mcsh_order]

def get_feature_list_legendre(max_mcsh_order, max_legendre_order, max_r):
    result = []
    for mcsh_order in range(max_mcsh_order+1):
        for group_num in get_mcsh_order_group(mcsh_order):
            for legendre_order in range(max_legendre_order+1):
                result.append("{}_{}_{:.6f}_Legendre_{}".format(mcsh_order, group_num, max_r, legendre_order))
    return result


def load_system(system, functional, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_ORDER, MCSH_MAX_R, directory = "./"):


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

def partition(data, refdata, max_distance):
    kd_tree = KDTree(refdata,leafsize=6)
    temp_distances, temp_indices = kd_tree.query(data, k=1)


    indices = []
    for i, distance in enumerate(temp_distances):
        if distance[0]< max_distance:
            indices.append(temp_indices[i])
    indices = np.array(indices)

    
    indices, counts = np.unique(indices, return_counts=True)
    count_arr = np.zeros(len(refdata))
    
    for i, index in enumerate(indices):
        count_arr[index] = counts[i]

    
    return count_arr



model_filename = ""
system = ""

model = pickle.load( open( model_filename, "wb" ) )
model_setup = model["model_setup"]
max_distance = model["max_distance"]
reg_model = model["regression_model"]


temp_feature_arr = load_system( system, \
                                functional = model_setup["functional"], \
                                MCSH_RADIAL_MAX_ORDER = model_setup["MCSH_RADIAL_MAX_ORDER"], \
                                MCSH_MAX_ORDER = model_setup["MCSH_MAX_ORDER"], \
                                MCSH_MAX_R = model_setup["MCSH_MAX_R"])
if model_setup["PCA"]:
    feature_arr_transformed = model["PCA_model"].transform(temp_feature_arr)
else:
    feature_arr_transformed = temp_feature_arr
count_arr = partition(feature_arr_transformed, refdata_transformed)

correction = reg_model.predict(count_arr)
print("calculated correction is: {}".format(correction))