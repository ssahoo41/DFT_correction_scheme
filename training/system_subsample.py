from NNSubsampling import subsampling
import numpy as np
import h5py
import pickle
import glob
import os
import sys

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


def subsample_system(system, hdf5_filename, functional = "GGA_PBE", MCSH_RADIAL_MAX_ORDER = 5, MCSH_MAX_ORDER = 3, MCSH_MAX_R = 3.0,\
                    cutoff_sig = 0.1, standard_scale = False, num_grid_pts = 100*100*100):


    subsampled_data_filename = "./system_subsampled_files/{}_MCSHLegendre_{}_{:.6f}_{}_subsampled_{}_{}.pickle"\
                         .format(system,MCSH_MAX_ORDER, MCSH_MAX_R, MCSH_RADIAL_MAX_ORDER,cutoff_sig,standard_scale)

    # hdf5_filename = "{}_MCSHLegendre_{}_{:.6f}_{}.h5"\
    #                      .format(system,MCSH_MAX_ORDER, MCSH_MAX_R, MCSH_RADIAL_MAX_ORDER)
    
    if os.path.exists(subsampled_data_filename):
        return
    else:
        feature_list = get_feature_list_legendre(MCSH_MAX_ORDER, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_R)
        num_features = len(feature_list) + 1

        feature_arr = np.zeros((num_grid_pts, num_features))

        with h5py.File(hdf5_filename,'r') as data:
            functional_database_group = data["functional_database"]
            functional_group = functional_database_group[functional]
            feature_grp  = functional_group['feature']
            rho = np.array(feature_grp["rho"])
            feature_arr[:,0] = rho.flatten()
            for i, feature in enumerate(feature_list):
                temp_feature = np.array(feature_grp[feature])
                feature_arr[:,i+1] = temp_feature.flatten()
                
        subsampled_feature_arr = subsampling(feature_arr,cutoff_sig=cutoff_sig,rate=0.1, method = "pykdtree",\
                                             verbose = 2, standard_scale=standard_scale)
        pickle.dump( subsampled_feature_arr, open( subsampled_data_filename, "wb" ) )

system_name = sys.argv[1]
data_filepath = sys.argv[2]
functional = sys.argv[3]

try:
    subsample_system(system_name, data_filepath, functional = functional)
except:
    print("error, file not exist?")
