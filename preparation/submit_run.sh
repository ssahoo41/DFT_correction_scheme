for FILEPATH in /storage/coda1/p-amedford6/0/ssahoo41/testflight_data/SPARC_test_mcsh/sparc_cccbdb_data/training_data/*
do
    echo $FILEPATH
    qsub -v FILEPATH=$FILEPATH run.sh
done
