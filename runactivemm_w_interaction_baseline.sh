
echo '************************* Volume *********************************'
#rm ./amm_vol_baseline.txt
#python classify_gl2.py -cf sample_ActiveMM_SpectralBaseline_OutcallsperUser2.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus >./amm_vol_baseline.txt


echo '************************* Degree *********************************'
#rm ./amm_degree_baseline.txt
#python classify_gl2.py -cf sample_ActiveMM_SpectralBaseline_OutDegreeperUser2.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus >./amm_degree_baseline.txt

echo '**********************DFA****************************************'
#rm ./amm_pak_icml.txt
python classify_gl.py -cf prop_sample_ActiveMMStatus_pak_icml_final_df2.csv -lf MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus |tee ./amm_pak_icml2.txt

