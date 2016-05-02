
echo '************* Volume ****************************'
#python classify_gl2.py -cf sample_MMStatus_SpectralBaseline_OutcallsperUser.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv MMStatus   -j inner -i 1 -ex ActiveMMStatus>./mm_vol_baseline.txt

echo '************* Degree ****************************'
#python classify_gl2.py -cf sample_MMStatus_SpectralBaseline_OutDegreeperUser.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv MMStatus   -j inner -i 1 -ex ActiveMMStatus>./mm_degree_baseline.txt

#sample_ActiveMMStatus_pak_icml_final_df.csv
rm mm_dfa_baseline.txt
echo '************* DFA ****************************'
python classify_gl.py -cf sample_MMStatus_pak_icml_final_df2.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv MMStatus   -j inner -i 1 -ex ActiveMMStatus|tee ./mm_dfa_baseline.txt

