
#echo '************************* Volume *********************************'
rm ./logs/amm_vol_baseline.txt
python classify_gl_baseline.py -cf data/sample_ActiveMM_SpectralBaseline_OutcallsperUser2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus -b 1|tee ./logs/amm_vol_baseline.txt

#rm ./logs/amm_vol_baseline_wo_interaction.txt
#python classify_gl.py -cf data/sample_ActiveMM_SpectralBaseline_OutcallsperUser2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 0 -ex MMStatus -b 1|tee ./logs/amm_vol_baseline_wo_interaction.txt

#echo '************************* Degree *********************************'
#rm ./logs/amm_degree_baseline.txt
#python classify_gl.py -cf data/sample_ActiveMM_SpectralBaseline_OutDegreeperUser2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus -b 1|tee ./logs/amm_degree_baseline.txt

#rm ./logs/amm_degree_baseline_wo_interaction.txt
#python classify_gl.py -cf data/sample_ActiveMM_SpectralBaseline_OutDegreeperUser2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus -b 1|tee ./logs/amm_degree_baseline_wo_interaction.txt

#echo '**********************DFA****************************************'
#rm ./logs/amm_pak_icml_std.txt
#python classify_gl_baseline.py -cf data/prop_sample_ActiveMMStatus_pak_icml_final_df2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus -b 1 -norm standard|tee ./logs/amm_pak_icml_std.txt 

#rm ./logs/amm_pak_icml_wo_interaction_std.txt
#python classify_gl_baseline.py -cf data/prop_sample_ActiveMMStatus_pak_icml_final_df2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 0 -ex MMStatus -b 1 -norm standard |tee ./logs/amm_pak_icml2_wo_interaction_std.txt 

#rm ./logs/amm_pak_icml_sqrt.txt
#python classify_gl_baseline.py -cf data/prop_sample_ActiveMMStatus_pak_icml_final_df2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus -b 1 -norm sqrt|tee ./logs/amm_pak_icml_sqrt.txt 

#rm ./logs/amm_pak_icml_wo_interaction_sqrt.txt
#python classify_gl_baseline.py -cf data/prop_sample_ActiveMMStatus_pak_icml_final_df2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 0 -ex MMStatus -b 1 -norm sqrt |tee ./logs/amm_pak_icml2_wo_interaction_sqrt.txt 
#rm ./logs/amm_pak_icml_norm.txt
#python classify_gl_baseline.py -cf data/prop_sample_ActiveMMStatus_pak_icml_final_df2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 1 -ex MMStatus -b 1 -norm normal|tee ./logs/amm_pak_icml_norm.txt 

#rm ./logs/amm_pak_icml_wo_interaction_norm.txt
#python classify_gl_baseline.py -cf data/prop_sample_ActiveMMStatus_pak_icml_final_df2.csv -lf data/MobileMoneyStatus.csv -of quad_mm.csv -cfk orig_cid2 -lfk orig_cid2 -lfv ActiveMMStatus   -j inner -i 0 -ex MMStatus -b 1 -norm normal |tee ./logs/amm_pak_icml2_wo_interaction_norm.txt 
