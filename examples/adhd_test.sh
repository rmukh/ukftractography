start=`date +%s`

SRC="../UKFTractography"

#logFile Name
logFile="log.txt"

# BINARY
BINARY='../build/UKFTractography-build/UKFTractography/bin/UKFTractography'

# VOLUME
dwi_path="/home/rinat/Desktop/ukftests/dwi-Aligned-Ed-Bet-Merged-unring-Epi.nhdr"

# MASK
mask_path="/home/rinat/Desktop/ukftests/epi_corrected_tensormask.nrrd"

# SEEDS
seeds_path="/home/rinat/Desktop/ukftests/Segmentation-label_363x.nrrd"

# OUTPUT FIBER
output_path='/home/rinat/Desktop/ukftests/adhd363/seeds_tc_363x_NO_init_multi.vtk'

# --seedsFile $seeds_path \

eval $BINARY \
 --dwiFile $dwi_path \
 --maskFile $mask_path \
 --tracts $output_path \
 --seedsPerVoxel 1 \
 --diffusionPropagator \
 --minRTOP 20 \
 --recordNMSE \
 --recordWeights \
 --recordRTOP \
 --seedsFile $seeds_path \
 --numTensor 3 | tee $logFile
 end=`date +%s`

runtime=$((end - start))
echo "Output file name $output_path" | tee -a $logFile
echo "CPU Runtime was $runtime sec."  | tee -a $logFile

#diff $output_path $SRC/ukf/Data/Baseline/seeds_tc.vtk
