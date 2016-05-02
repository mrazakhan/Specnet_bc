rm mobilemoney_log.txt
python classify_gl.py -cf sample_MMStatus_spectralFeature_CDR_raw_nofilt_aysm__50_1.0_apr28.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv MMStatus   -j inner -i 1 -ex ActiveMMStatus |tee mobilemoney_log.txt
