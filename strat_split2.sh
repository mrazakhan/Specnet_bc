#spectralFeature_CDR_out_30_1.0_apr19.csv

# base dir /data/pk_anon/preds/  
# files spectralFeature_CDR_raw_nofilt_50_1.0_apr28.csv spectralFeature_CDR_raw_nofilt_aysm__50_1.0_apr28.csv  spectralFeature_CDR_raw_nofilt_100_1.0_apr28.csv 

base_dir=./
for each in spectralFeature_CDR_raw_nofilt_aysm__50_1.0_apr28.csv
	do
		echo $base_dir$each
		#python stratified_sample.py $each MobileMoneyStatus.csv ActiveMMStatus&
		#python stratified_sample.py $each MobileMoneyStatus.csv MMStatus&
		python stratified_sample.py $each UserDemographics.csv gend&
	done



#python stratified_sample.py spectralFeature_CDR_raw_nofilt_10_full_apr22.csv MobileMoneyStatus.csv ActiveMMStatus
#python stratified_sample.py spectralFeature_CDR_raw_nofilt_10_full_apr22.csv MobileMoneyStatus.csv MMStatus
#python stratified_sample.py spectralFeature_CDR_raw_nofilt_10_full_apr22.csv UserDemographics.csv gend

#python stratified_sample.py SpectralBaseline_OutcallsperUser.csv  UserDemographics.csv gend
#python stratified_sample.py SpectralBaseline_OutcallsperUser.csv  MobileMoneyStatus.csv ActiveMMStatus
#python stratified_sample.py SpectralBaseline_OutcallsperUser.csv  MobileMoneyStatus.csv MMStatus

#python stratified_sample.py SpectralBaseline_OutDegreeperUser.csv  UserDemographics.csv gend
#python stratified_sample.py SpectralBaseline_OutDegreeperUser.csv  MobileMoneyStatus.csv ActiveMMStatus
#python stratified_sample.py SpectralBaseline_OutDegreeperUser.csv  MobileMoneyStatus.csv MMStatus
