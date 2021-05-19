for SYSTEM in /storage/home/hcoda1/0/ssahoo41/data/testflight_data/SPARC_test_mcsh/results_folder/preparation_results/raw_data_files/*
do
	echo $SYSTEM
        qsub -v SYSTEM=$SYSTEM run_subsample.sh 
done
