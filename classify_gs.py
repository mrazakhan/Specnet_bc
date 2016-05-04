import argparse
import graphlab as gl
import csv
import sys
import math
import numpy
from graphlab import feature_engineering as fe
from sklearn.metrics import roc_auc_score
from graphlab import model_parameter_search
DEBUG=1

def bin_encode(x):
	if x==None:
		return 0
	elif x>=4:
		return 1
	else:
		return 0


def custom_auc(roc_curve):
	fpr_vs_tpr = roc_curve['roc_curve'].sort('fpr',ascending=True)[['fpr', 'tpr']]
	area = 0.0
	for i in range(0,len(fpr_vs_tpr)-1):
	    cur_row = fpr_vs_tpr[i]
	    next_row = fpr_vs_tpr[i+1]
	    # build a trapezoid
	    base = next_row['fpr'] - cur_row['fpr']
	    left_side = cur_row['tpr']
	    right_side = next_row['tpr']
	    area += base * (left_side + right_side) / 2.0
	print 'Custom AUCC = %.3f' % area
	return area


def preprocess(clusteral_features, ground_truth, output_file, clusteral_key_col, labels_key_col, labels_value_col, interaction=0, join_type='inner', encode=False):
	lf=gl.SFrame.read_csv(ground_truth)[[labels_key_col,labels_value_col]]
	print 'Shape of the labels file is ', lf.shape
	cf=gl.SFrame.read_csv(clusteral_features).rename({clusteral_key_col:labels_key_col})
	print 'Shape of the Clusteral file is ', cf.shape

	cf=cf[[each for each in cf.column_names() if each not in ['old_cid','orig_cid','gend'] and 'alter' not in each] ]
	
	for col in cf.column_names():
		cf=cf.fillna(col,0)
	
	merged_sf=lf.join(cf , on=labels_key_col, how=join_type).fillna(labels_value_col,0)
	print 'Shape of the merged file is ', merged_sf.shape
	print merged_sf[labels_value_col].sketch_summary()
	
	if encode:
		merged_sf[labels_value_col]=merged_sf[labels_value_col].apply(lambda x:bin_encode(x))
		merged_sf[labels_value_col].head(2)

	print merged_sf[labels_value_col].sketch_summary()

	interaction_columns=[each for each in cf.column_names() if labels_key_col not in each and clusteral_key_col not in each and labels_value_col not in each]
	if DEBUG:
		print 'Interaction.column_names()'
		print interaction_columns
		#sys.exit(0)
	if interaction:
		quad = fe.create(merged_sf, fe.QuadraticFeatures(features=interaction_columns))	
		print 'Applying Quadratic Transformation'
		merged_sf=quad.transform(merged_sf)
		#print 'Flattening the quadratic features'
		#merged_sf=merged_sf.unpack('quadratic_features')
	
	return merged_sf

def get_auc(model, test, col):
	pred=model.predict(test, output_type='class')
	ret=gl.evaluation.auc(test[col], pred)
	roc_curve = gl.evaluation.roc_curve(test[col], pred)
	roc_curve['auc']=(roc_curve['tpr']-roc_curve['fpr']+1)/2.0
	print 'Max auc {}'.format(max(roc_curve['auc']))
	#ret = model.evaluate(test, metric='auc')
	print 'auc through the numpy ', roc_auc_score(numpy.asarray(test[col]), numpy.asarray(pred))
	return ret

def log_loss(model, test, col):
	'''Evaluate a trained model using Kaggle scoring.'''
	return log_loss_raw(test[col], model.predict(test, output_type='probability'))
	
def log_loss_raw(target, predicted):
	'''Calculate log_loss between target and predicted and return.'''
	p = predicted.apply(lambda x: min(0.99999, max(1e-5, x)))
	logp = p.apply(lambda x: math.log(x))
	logmp = p.apply(lambda x: (math.log(1-x)))
	return -(target * logp + (1-target) * logmp).mean()

def twoclass_stratified_sampling(sf, col, fraction, values=[0,1]):
	sf_a=sf.filter_by(values[0], col) 
	sf_b=sf.filter_by(values[1], col) 
	print 'Shape of Class 1 and Class 2 Sframes is {}, {}'.format(sf_a.shape,sf_b.shape)
	sf_a,temp=sf_a.random_split(seed=12345, fraction=fraction)
	sf_b,temp=sf_b.random_split(seed=12345, fraction=fraction)

	print 'Shape of Class 1 and Class 2 Sframes after sampling is {}, {}'.format(sf_a.shape,sf_b.shape)
	sf_a,temp=sf_a.random_split(seed=12345, fraction=fraction)
	
	return sf_a.append(sf_b)

def get_coeffs(model):
	
	coefs = model['coefficients'].sort('value',ascending=False)
	print coefs.head(20)
	return coefs

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

	args=parser.parse_args()
	

	join_type='inner'
	encode=False
	if args.join_type is not None:
		join_type=args.join_type
	if args.encode is not None:
		if args.encode=='1':
			encode=True

	print args

	interaction=int(args.interaction)
	
	quad_df=preprocess(clusteral_features=args.clusteral_features, \
		ground_truth=args.labels_file,\
		output_file=args.output_file,\
		clusteral_key_col=args.clusteral_key_column, \
		labels_key_col=args.labels_key_column,\
		labels_value_col=args.labels_value_column, interaction=interaction, join_type=join_type, encode=encode)
	
	#quad_df=gl.SFrame.read_csv(args.clusteral_features)
	#quad_df=twoclass_stratified_sampling(quad_df, col=args.labels_value_column, fraction=0.3)
	#quad_df=quad_df[[ for each in quad_df.column_names()]]
	print '****************Unbalanced Sample*****************'
	train, val = quad_df.random_split(0.75, seed=12345)
	train_features=[each for each in quad_df.column_names() \
                if args.labels_key_column not in each and args.exclude not in each and args.clusteral_key_column not in each and args.labels_value_column not in each]
	params = {'target': args.labels_value_column,'features':[train_features]}
	try:
		job = model_parameter_search.create((train, val),
                                    gl.logistic_classifier.create,
                                    params)
		results = job.get_results()

		print results
	except:
		print job.get_metrics()
	results.export_csv('Logistic_GS_Results.csv',quote_level=csv.QUOTE_NONE)
	
	print '****************Unbalanced Sample*****************'
	train, val = quad_df.random_split(0.75, seed=12345)
	train_features=[each for each in quad_df.column_names() \
                if args.labels_key_column not in each and args.exclude not in each and args.clusteral_key_column not in each and args.labels_value_column not in each]
	params = {'target': args.labels_value_column,'features':[train_features]}
	try:
		job = model_parameter_search.create((train, val),
                                    gl.linear_classifier.create,
                                    params)
		results2 = job.get_results()

		print results2
	except:
		print job.get_metrics()
	results2.export_csv('Linear_GS_Results.csv',quote_level=csv.QUOTE_NONE)
