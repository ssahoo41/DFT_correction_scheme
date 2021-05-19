from NNSubsampling import subsampling
import numpy as np
import h5py
import pickle
import glob
import os
import sys

def log_result(log_filename, message):
    f = open(log_filename, 'a')
    f.write(message)
    f.close()
    return 

def get_mcsh_order_group(mcsh_order):
    group_dict = {0:[1],\
                  1:[1],\
                  2:[1,2],\
                  3:[1,2,3],\
                  4:[1,2,3,4],
                  5:[1,2,3,4,5]}
    return group_dict[mcsh_order]

def get_feature_list_legendre(max_mcsh_order, max_legendre_order, max_r):
    result = []
    for mcsh_order in range(max_mcsh_order+1):
        for group_num in get_mcsh_order_group(mcsh_order):
            for legendre_order in range(max_legendre_order+1):
                result.append("{}_{}_{:.6f}_Legendre_{}".format(mcsh_order, group_num, max_r, legendre_order))
    return result

def subsample_system(system, hdf5_filename, functional = "GGA_PBE", MCSH_RADIAL_MAX_ORDER = 5, MCSH_MAX_ORDER = 3, MCSH_MAX_R = 3.0,\
                    cutoff_sig = 0.1, standard_scale = False, num_grid_pts = 100*100*100): #can be used with standard_scale = True also


    subsampled_data_filename = "/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/DFT_correction_scheme/training/system_subsampled_files/{}_MCSHLegendre_{}_{:.6f}_{}_subsampled_{}_{}.pickle"\
                         .format(system,MCSH_MAX_ORDER, MCSH_MAX_R, MCSH_RADIAL_MAX_ORDER,cutoff_sig,standard_scale)#save the subsampled files as this
    #print(subsampled_data_filename)
    # hdf5_filename = "{}_MCSHLegendre_{}_{:.6f}_{}.h5"\
    #                      .format(system,MCSH_MAX_ORDER, MCSH_MAX_R, MCSH_RADIAL_MAX_ORDER)
    
    if os.path.exists(subsampled_data_filename):
        #print(os.path.exists(subsampled_data_filename))
        return
    else: #else make the file
        feature_list = get_feature_list_legendre(MCSH_MAX_ORDER, MCSH_RADIAL_MAX_ORDER, MCSH_MAX_R)
        num_features = len(feature_list) + 1 #for density

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
            #print(feature_arr)
        subsampled_feature_arr = subsampling(data=feature_arr,cutoff_sig=cutoff_sig,rate=0.1, method = "pykdtree",\
                                             verbose = 2, standard_scale=False)# can turn off standard scale here 
        len_sub = len(subsampled_feature_arr)
        pickle.dump( subsampled_feature_arr, open( subsampled_data_filename, "wb" ) )
    return len_sub

#How the subsampling is run (import sys)- we give arguments while running the code for subsampling 
#system_name = sys.argv[1] #name of the system
data_filepath = sys.argv[1] #this is the file path to hdf5 file
functional = sys.argv[2] #what functional data needs to be subsampled (right now it is just GGA_PBE)

system = data_filepath.split("_MCSH")[0]
system_name = system.split("raw_data_files/")[1]
print(system_name)

filename = "./logged_data/{}_subsampled_length.dat".format(system_name)
for i in range(2,20,2):
    print(i)
    print(np.log(i))
    len_sub_feature = subsample_system(system_name, data_filepath, cutoff_sig = np.log(i), functional = functional)
    message = "{}\t{}\n".format(np.log(i), len_sub_feature)
    log_result(filename, message)

