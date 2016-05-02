rm gender_log.txt
#python classify_gl.py -cf head2.txt -lf UserDemographics.csv -of quad_gend.csv -cfk orig_cid2 -lfk orig_cid2 -lfv gend   -j inner -i 1 -ex MMStatus | tee gender_log.txt
python classify_gl.py -cf sample_gend_spectralFeature_CDR_raw_nofilt_aysm__50_1.0_apr28.csv -lf UserDemographics.csv -of quad_gend.csv -cfk orig_cid2 -lfk orig_cid2 -lfv gend   -j inner -i 1 -ex MMStatus | tee gender_log.txt
