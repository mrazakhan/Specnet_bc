echo '****************** Volume****************'

python classify_gl2.py -cf sample_gend_SpectralBaseline_OutcallsperUser.csv -lf UserDemographics.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv gend   -j inner -i 1 -ex MMStatus >./gend_mm_vol_baseline.txt
echo '****************** Degree****************'
#python classify_gl2.py -cf sample_gend_SpectralBaseline_OutDegreeperUser.csv -lf UserDemographics.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv gend   -j inner -i 1 -ex MMStatus>./gend_mm_degree_baseline.txt

#sample_gend_pak_icml_final_df.csv
echo '****************** DFA****************'
#python classify_gl2.py -cf sample_gend_pak_icml_final_df.csv -lf UserDemographics.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv gend   -j inner -i 1 -ex MMStatus >./gend_mm_dfa_baseline.txt

