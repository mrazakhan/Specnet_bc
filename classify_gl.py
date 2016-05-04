import argparse
import graphlab as gl
import csv
import sys
import math
import numpy
from graphlab import feature_engineering as fe
from sklearn.metrics import roc_auc_score
DEBUG=1

class Util():
	
	def bin_encode(self,x):
		if x==None:
			return 0
		elif x>=4:
			return 1
		else:
			return 0


class Eval():

	def custom_auc(self,roc_curve):
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
	def get_auc(self,model, test, col):
		try:
			pred=model.predict(test, output_type='class')
			print 'auc through the numpy ', roc_auc_score(numpy.asarray(test[col]), numpy.asarray(pred))
			roc_curve = gl.evaluation.roc_curve(test[col], pred)
			roc_curve['auc']=(roc_curve['tpr']-roc_curve['fpr']+1)/2.0
			print 'Max auc {}'.format(max(roc_curve['auc']))
			#ret = model.evaluate(test, metric='auc')
			ret=gl.evaluation.auc(test[col], pred)
		except:
			
			pred=model.predict(test)
			ret= roc_auc_score(numpy.asarray(test[col]), numpy.asarray(pred))
			#roc_curve = gl.evaluation.roc_curve(test[col], pred)
			#roc_curve['auc']=(roc_curve['tpr']-roc_curve['fpr']+1)/2.0
			#print 'Max auc {}'.format(max(roc_curve['auc']))
			#ret = model.evaluate(test, metric='auc')
			print 'auc through the numpy ', ret
			pass
		return ret

	def log_loss(self,model, test, col):
		'''Evaluate a trained model using Kaggle scoring.'''
		try:
			ret= self.log_loss_raw(test[col], model.predict(test, output_type='probability'))
		except:
	
			ret= self.log_loss_raw(test[col], model.predict(test))
			pass
		return ret
		
	def log_loss_raw(self,target, predicted):
		'''Calculate log_loss between target and predicted and return.'''
		p = predicted.apply(lambda x: min(0.99999, max(1e-5, x)))
		logp = p.apply(lambda x: math.log(x))
		logmp = p.apply(lambda x: (math.log(1-x)))
		return -(target * logp + (1-target) * logmp).mean()

class spectral_training():
	def __init__(self, clusteral_features, ground_truth, output_file, clusteral_key_col, labels_key_col, labels_value_col, interaction=0, join_type='inner',encode=False):
		self.util=Util()
		self.evaluator=Eval()	
		self.clusteral_features=clusteral_features
		self.ground_truth=ground_truth
		self.output_file=output_file
		self.clusteral_key_col=clusteral_key_col
		self.labels_key_col=labels_key_col
		self.labels_value_col=labels_value_col
		self.interaction=interaction
		self.join_type=join_type
		self.encode=encode
		

	def preprocess(self):
		ground_truth=self.ground_truth
		labels_key_col=self.labels_key_col
		labels_value_col=self.labels_value_col
		clusteral_features=self.clusteral_features
		clusteral_key_col=self.clusteral_key_col		


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
		
		self.merged_sf=merged_sf


	def twoclass_stratified_sampling(self,sf, col, fraction, values=[0,1]):
		sf_a=sf.filter_by(values[0], col) 
		sf_b=sf.filter_by(values[1], col) 
		print 'Shape of Class 1 and Class 2 Sframes is {}, {}'.format(sf_a.shape,sf_b.shape)
		sf_a,temp=sf_a.random_split(seed=12345, fraction=fraction)
		sf_b,temp=sf_b.random_split(seed=12345, fraction=fraction)

		print 'Shape of Class 1 and Class 2 Sframes after sampling is {}, {}'.format(sf_a.shape,sf_b.shape)
		sf_a,temp=sf_a.random_split(seed=12345, fraction=fraction)
		
		return sf_a.append(sf_b)

	def get_coeffs(self,model):
		
		coefs = model['coefficients'].sort('value',ascending=False)
		print coefs.head(20)
		return coefs

	def fit_and_predict(self,classifier, train_features,mode='bal'):
		if mode=='bal':
			train=self.train_bal
			val=self.val_bal
		elif mode=='unbal':
			train=self.train_unbal
			val=self.val_unbal
		model = classifier.create(train, features=train_features, target=args.labels_value_column,\
			l1_penalty=0.01,validation_set=val, max_iterations=3, convergence_threshold=0.001)#,metric='auc')
		#model = gl.boosted_trees_classifier.create(train, features=train_features, target=args.labels_value_column,\
		#        validation_set=val, metric='auc')
		print "LL %0.20f" % self.evaluator.log_loss(model, val, col=args.labels_value_column)
		print "AUC %0.20f" % self.evaluator.get_auc(model, val, col=args.labels_value_column)
		try:
			roc_curve = model.evaluate(val, metric='roc_curve')
			area=self.evaluator.custom_auc(roc_curve)
		except:
			pass
		results = model.evaluate(val)
		try:
			print 'Accuracy', results['accuracy']
		except:

			print 'Accuracy', results
			pass
		print 'Class Percentages in the validation data', val[args.labels_value_column].sketch_summary()
		print 'Class Percentages in the training data', train[args.labels_value_column].sketch_summary()


	def run(self):
		self.preprocess()
		print '****************Unbalanced Sample*****************'
		self.train_bal, self.val_bal = self.merged_sf.random_split(0.75, seed=12345)
		train_features=[each for each in self.merged_sf.column_names() \
			if args.labels_key_column not in each and args.exclude not in each and args.clusteral_key_column not in each and args.labels_value_column not in each]
		print '***************Logistic****************************'
		self.fit_and_predict(gl.logistic_classifier, train_features)
		print '***************Linear****************************'
		self.fit_and_predict(gl.linear_regression, train_features)


		print '****************Balanced Sample*****************'
		self.merged_sfclass1=self.merged_sf.filter_by(1,args.labels_value_column)
		self.merged_sfclass0=self.merged_sf.filter_by(0,args.labels_value_column)

		# Random split for the bigger sf
		if self.merged_sfclass0.shape[0]<self.merged_sfclass1.shape[0]:
			print 'Case 1, Class1 {} Class 2 {}'.format(self.merged_sfclass0.shape[0], self.merged_sfclass1.shape[0])
			self.merged_sfclass1,temp=self.merged_sfclass1.random_split(self.merged_sfclass0.shape[0]/float(self.merged_sfclass1.shape[0]))
			self.merged_sf=self.merged_sfclass1.append(self.merged_sfclass0)
		else:
			print 'Case 2, Class1 {} Class 2 {}'.format(self.merged_sfclass0.shape[0], self.merged_sfclass1.shape[0])
			self.merged_sfclass0,temp=self.merged_sfclass0.random_split(self.merged_sfclass1.shape[0]/float(self.merged_sfclass0.shape[0]))
			self.merged_sf=self.merged_sfclass0.append(self.merged_sfclass1)
		
		print 'Shape of the class1 df for the balanced sample is', self.merged_sfclass1.shape
		print 'Shape of the class0 df for the balanced sample is', self.merged_sfclass0.shape
		print 'Shape of the merged quad df is ', self.merged_sf.shape
		

		train_features=[each for each in self.merged_sf.column_names() \
			if args.labels_key_column not in each and args.clusteral_key_column not in each and args.labels_value_column not in each]

		self.train_unbal, self.val_unbal = self.merged_sf.random_split(0.75, seed=12345)
		
		print '***************Logistic****************************'
		self.fit_and_predict(gl.logistic_classifier, train_features,mode='unbal')
		print '***************Linear****************************'
		self.fit_and_predict(gl.linear_regression, train_features,mode='unbal')

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

	trainer=spectral_training(args.clusteral_features, args.labels_file, args.output_file, args.clusteral_key_column, args.labels_key_column, args.labels_value_column, interaction=0, join_type='inner',encode=False)

	trainer.run()	
	
