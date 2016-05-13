#python classify_gl.py -cf sample_ActiveMMStatus_spectralFeature_CDR_raw_nofilt_10_full_apr22.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus
#rm ./ActiveMM_symm_28.txt
#python classify_gl.py -cf head2.txt -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus |tee ./ActiveMM_symm_28.txt
#python classify_gl.py -cf sample_ActiveMMStatus_spectralFeature_CDR_raw_nofilt_100_1.0_apr28.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus |tee ./ActiveMM_symm_28.txt
#rm ./ActiveMM_asymm_28.txt

base_dir="//data//pk_anon//preds//"
#logs/log_with_interaction_spectralFeature_CDR_raw_nofilt_50_1.0_apr28_cleaned.csv
#infiles="spectralFeature_CDR_numCalls_nofilt_asym_50_1.0_apr28_cleaned.csv"

infiles="spectralFeature_CDR_numCalls_nofilt_asym_100_1.0_apr28_cleaned.csv"

#infiles="spectralFeature_CDR_raw_nofilt_100_1.0_apr28_cleaned.csv spectralFeature_CDR_numCalls_nofilt_asym_100_1.0_apr28_cleaned.csv spectralFeature_CDR_raw_nofilt_50_1.0_apr28_cleaned.csv spectralFeature_CDR_numCalls_nofilt_100_1.0_apr28_cleaned.csv  spectralFeature_CDR_numCalls_nofilt_asym_50_1.0_apr28_cleaned.csv spectralFeature_CDR_numCalls_nofilt_50_1.0_apr28_cleaned.csv spectralFeature_CDR_raw_nofilt_100_1.0_apr28_cleaned.csv spectralFeature_CDR_numCalls_nofilt_asym_100_1.0_apr28_cleaned.csv spectralFeature_CDR_raw_nofilt_50_1.0_apr28_cleaned.csv" 
#infiles="spectralFeature_CDR_numCalls_nofilt_100_1.0_apr28_cleaned.csv spectralFeature_CDR_numCalls_nofilt_asym_50_1.0_apr28_cleaned.csv spectralFeature_CDR_numCalls_nofilt_50_1.0_apr28_cleaned.csv spectralFeature_CDR_raw_nofilt_100_1.0_apr28_cleaned.csv spectralFeature_CDR_numCalls_nofilt_asym_100_1.0_apr28_cleaned.csv spectralFeature_CDR_raw_nofilt_50_1.0_apr28_cleaned.csv" 

#spectralFeature_CDR_numCalls_nofilt_50_1.0_apr28.csv spectralFeature_CDR_numCalls_nofilt_asym_100_1.0_apr28.csv spectralFeature_CDR_numCalls_nofilt_asym_50_1.0_apr28.csv spectralFeature_CDR_raw_nofilt_100_1.0_apr28.csv preds/spectralFeature_CDR_raw_nofilt_50_1.0_apr28.csv spectralFeature_CDR_raw_nofilt_aysm_100_1.0_apr28.csv spectralFeature_CDR_raw_nofilt_aysm__50_1.0_apr28.csv"

for each in $infiles
	do 
		echo $base_dir$each
	
		#python classify_gl.py -cf $base_dir$each -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 0 -ex MMStatus  |tee ./logs/log_no_interaction$each
		#python classify_gl.py -cf $base_dir$each -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus  |tee ./logs/log_with_interaction$each
		#python classify_gl.py -cf $base_dir$each -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 0 -ex MMStatus -norm sqrt |tee ./logs/log_no_interaction_sqrt$each
		python classify_gl.py -cf $base_dir$each -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus -norm sqrt |tee ./logs/log_with_interaction_sqrt$each
		#python classify_gl.py -cf $base_dir$each -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 0 -ex MMStatus -norm normal |tee ./logs/log_no_interaction_normal$each
		#python classify_gl.py -cf $base_dir$each -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus -norm normal |tee ./logs/log_with_interaction_normal$each
	done
