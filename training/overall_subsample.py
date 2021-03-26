from NNSubsampling import subsampling
import numpy as np
import pickle
import glob
import os

data_list = []
for file in glob.glob("system_subsampled_files/*.pickle"):
    system_name = file.split("_")[0]
    print("start syste: {}".format(system_name))
    try:
        temp = pickle.load(open(file, "rb" ))
        data_list.append(temp)
    except:
        print("error, file not exist?")
    print("end")
    
overall_data = np.vstack(data_list)
    
cutoff_sig = 1.0
standard_scale = False

subsampled_data_filename = "Overall_subsampled_{}_{}.pickle"\
                         .format(cutoff_sig,standard_scale)

subsampled_feature_arr = subsampling(overall_data,cutoff_sig=cutoff_sig,rate=0.1, method = "pykdtree",\
                                             verbose = 2, standard_scale=standard_scale)

pickle.dump( subsampled_feature_arr, open( subsampled_data_filename, "wb" ) )