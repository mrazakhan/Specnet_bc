import argparse
import graphlab as gl
import csv
import sys
import math
from graphlab import feature_engineering as fe
DEBUG=1

def bin_encode(x):
	if x==None:
		return 0
	elif x>=4:
		return 1
	else:
		return 0


def preprocess(clusteral_features, ground_truth, output_file, clusteral_key_col, labels_key_col, labels_value_col, interaction=0, join_type='inner', encode=False):
	lf=gl.SFrame.read_csv(ground_truth)[[labels_key_col,labels_value_col]]
	print 'Shape of the labels file is ', lf.shape
	cf=gl.SFrame.read_csv(clusteral_features).rename({clusteral_key_col:labels_key_col})
	print 'Shape of the Clusteral file is ', cf.shape

		
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

def eval_model(model, test, col):
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

if __name__=='__main__':
	gl.set_runtime_config('GRAPHLAB_CACHE_FILE_LOCATIONS','/home/mraza/tmp/')	
	gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY', 200*1024*1024*1024)
	gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 100*1024*1024*1024)

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
	
	quad_df=gl.SFrame.read_csv(args.clusteral_features)
	#quad_df=twoclass_stratified_sampling(quad_df, col=args.labels_value_column, fraction=0.3)
	#quad_df=quad_df[[ for each in quad_df.column_names()]]
	print '****************Unbalanced Sample*****************'
	train, val = quad_df.random_split(0.75, seed=12345)
	train_features=[each for each in quad_df.column_names() \
		if args.labels_key_column not in each and args.exclude not in each and args.clusteral_key_column not in each and args.labels_value_column not in each]
	model = gl.logistic_classifier.create(train, features=train_features, target=args.labels_value_column,\
		 validation_set=val, max_iterations=5)
	print "LL %0.20f" % eval_model(model, val, col=args.labels_value_column)
	results = model.evaluate(val)
	print 'Accuracy', results['accuracy']
	print 'Class Percentages in the validation data', val[args.labels_value_column].sketch_summary()
	print 'Class Percentages in the training data', train[args.labels_value_column].sketch_summary()

	print '****************Balanced Sample*****************'
	train, val = quad_df.random_split(0.75, seed=12345)
	quad_df_class1=quad_df.filter_by(1,args.labels_value_column)
	quad_df_class0=quad_df.filter_by(0,args.labels_value_column)

	# Random split for the bigger sf
	if quad_df_class0.shape[0]<quad_df_class1.shape[0]:
		print 'Case 1, Class1 {} Class 2 {}'.format(quad_df_class0.shape[0], quad_df_class1.shape[0])
		quad_df_class1,temp=quad_df_class1.random_split(quad_df_class0.shape[0]/float(quad_df_class1.shape[0]))
		quad_df=quad_df_class1.append(quad_df_class0)
	else:
		print 'Case 2, Class1 {} Class 2 {}'.format(quad_df_class0.shape[0], quad_df_class1.shape[0])
		quad_df_class0,temp=quad_df_class0.random_split(quad_df_class1.shape[0]/float(quad_df_class0.shape[0]))
		quad_df=quad_df_class0.append(quad_df_class1)
	
	print 'Shape of the class1 df for the balanced sample is', quad_df_class1.shape
	print 'Shape of the class0 df for the balanced sample is', quad_df_class0.shape
	print 'Shape of the merged quad df is ', quad_df.shape
	
	train, val = quad_df.random_split(0.75, seed=12345)

	train_features=[each for each in quad_df.column_names() \
		if args.labels_key_column not in each and args.clusteral_key_column not in each and args.labels_value_column not in each]
	model = gl.logistic_classifier.create(train, features=train_features, target=args.labels_value_column,\
		 validation_set=val, max_iterations=5)
	print "LL %0.20f" % eval_model(model, val, col=args.labels_value_column)
	results = model.evaluate(val)
	print 'Accuracy', results['accuracy']
	#print 'Class Percentages in the validation data', val[args.labels_value_column].sketch_summary()
	#print 'Class Percentages in the training data', train[args.labels_value_column].sketch_summary()

	
	if interaction:
		print 'Saving the quadratic features file'
		quad_df.export_csv(args.output_file, quote_level=csv.QUOTE_NONE)
