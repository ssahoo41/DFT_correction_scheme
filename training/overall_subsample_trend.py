from NNSubsampling import subsampling
import numpy as np
import pickle
import glob
import os
import time

start = time.time()

data_list = []
for file in glob.glob("/storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/results_folder/training_results/system_subsample/system_subsampled_files_0.1/*.pickle"): #all the files in the subsampled folder 
    #system_name = file.split("_")[0]
    print(file)
    system_name = file.split("_MCSH")[0]
    print("start system: {}".format(system_name))
    try:
        temp = pickle.load(open(file, "rb" ))
        data_list.append(temp)
    except:
        print("error, file not exist?")
    print("end")
    
overall_data = np.vstack(data_list) #all the data in data_list

arr = np.arange(1.0001, 100, 5)
standard_scale = True #default is true 

for i in arr:
    cutoff_sig = np.log(i)
    start = time.time()
    subsampled_data_filename = "Overall_subsampled_{}_{}.pickle"\
                         .format(cutoff_sig,standard_scale)
    subsampled_feature_arr = subsampling(overall_data,cutoff_sig=cutoff_sig,rate=0.1, method = "pykdtree",\
                                             verbose = 2, standard_scale=standard_scale)#increased the rate from 0.1 to 0.5 to check if it works
    pickle.dump( subsampled_feature_arr, open( subsampled_data_filename, "wb" ) )
    end = time.time()
    print("Time elapsed for cutoff_sig {}:".format(cutoff_sig), end - start )

print(len(subsampled_feature_arr)) #gives a list 
