#python classify_gl.py -cf sample_ActiveMMStatus_spectralFeature_CDR_raw_nofilt_10_full_apr22.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus
#rm ./ActiveMM_symm_28.txt
#python classify_gl.py -cf head2.txt -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus |tee ./ActiveMM_symm_28.txt
#python classify_gl.py -cf sample_ActiveMMStatus_spectralFeature_CDR_raw_nofilt_100_1.0_apr28.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus |tee ./ActiveMM_symm_28.txt
rm ./ActiveMM_asymm_28.txt
python classify_gl.py -cf sample_ActiveMMStatus_spectralFeature_CDR_raw_nofilt_aysm__50_1.0_apr28.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus |tee ./ActiveMM_asymm_28.txt
