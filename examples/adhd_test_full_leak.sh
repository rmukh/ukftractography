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

# OUTPUT FIBER
output_path='/home/rinat/Desktop/ukftests/seeds_tc_full_multi.vtk'

# --seedsFile $seeds_path \
valgrind -v --show-leak-kinds=all --track-origins=yes --leak-check=full ../build/UKFTractography-build/UKFTractography/bin/UKFTractography \
 --dwiFile $dwi_path \
 --maskFile $mask_path \
 --tracts $output_path \
 --seedsPerVoxel 1 \
 --diffusionPropagator \
 --minRTOP 35 \
 --recordNMSE \
 --recordWeights \
 --recordRTOP \
 --seedingThreshold 0.5 \
 --numTensor 3 | tee $logFile

