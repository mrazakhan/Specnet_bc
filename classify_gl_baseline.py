import numpy as np
import re
import argparse
import graphlab as gl
import csv
import sys
import math
import numpy
from graphlab import feature_engineering as fe
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from util import Util
from evals import Eval
from spectral_training import SpectralTraining
DEBUG=1




if __name__=='__main__':
	gl.set_runtime_config('GRAPHLAB_CACHE_FILE_LOCATIONS','/home/mraza/tmp/')	
	gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY', 200*1024*1024*1024)
	gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 100*1024*1024*1024)
	gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 20)

	parser=argparse.ArgumentParser(description='Spectral Features Preprocessing')
	parser.add_argument('-cf','--clusteral_features',help='Input File containing clusteral features', required=True)
	parser.add_argument('-lf','--labels_file', help='Ground Truth labels file', required=True)
	parser.add_argument('-of','--output_file', help='Output file', required=True)
	parser.add_argument('-cfk','--clusteral_key_column', required=True)
	parser.add_argument('-lfk','--labels_key_column', required=True)
	parser.add_argument('-lfv','--labels_value_column', required=True)
	parser.add_argument('-i','--interaction', required=True)
	parser.add_argument('-j','--join_type', required=False)
	parser.add_argument('-e','--encode', required=False)
	parser.add_argument('-ex','--exclude', required=False)
	parser.add_argument('-b','--baseline', required=False)
	parser.add_argument('-norm','--normalization',help='standard, normal or sqrt', required=False)

	args=parser.parse_args()
	

	join_type='inner'
	encode=False
	baseline=False
	normalization='standard'
	if args.join_type is not None:
		join_type=args.join_type
	if args.encode is not None:
		if args.encode=='1':
			encode=True
	if args.baseline is not None:
		if args.baseline=='1':
			baseline=True
	if args.normalization is not None:
		normalization=args.normalization

	print args

	interaction=int(args.interaction)

	trainer=SpectralTraining(args.clusteral_features, args.labels_file, args.output_file, args.clusteral_key_column, args.labels_key_column, args.labels_value_column, interaction=0, join_type='inner',encode=False,baseline=baseline, normalization=normalization)

	trainer.run()	
	
