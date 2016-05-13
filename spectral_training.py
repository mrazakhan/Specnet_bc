import numpy as np
import re
import argparse
import graphlab as gl
import csv
import sys
import math
from math import sqrt
import numpy
from graphlab import feature_engineering as fe
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from util import Util
from evals import Eval
DEBUG=1



class SpectralTraining():
	def __init__(self, clusteral_features, ground_truth, output_file, clusteral_key_col, labels_key_col, labels_value_col, interaction=0, join_type='inner',encode=False, baseline=False,exclude='NA', normalization='standard'):
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
		self.baseline=baseline
		self.max_iterations=30
		self.ntype=normalization
		self.exclude=exclude
		if self.baseline==1:
			self.max_iterations=120
	def signed_sqrt(self, x):
		factor=1
		if x<0:
			factor=-1
		return factor*sqrt(abs(x))

	def normalize(self,cf,ntype):
		cols=[each for each in cf.column_names() if self.labels_key_col not in each and self.labels_value_col not in each]
		
		for col in cols:
			sketch=gl.Sketch(cf[col])
			mean_val=sketch.mean()
			std_val=sketch.std()
                        cf[col]=cf[col].apply(lambda x:((x-mean_val)/std_val))
		if ntype=='normal':#0,1
						
			for col in cols:
				sketch=gl.Sketch(cf[col])
				min_val=sketch.min()
				max_val=sketch.max()
				
                        	cf[col]=cf[col].apply(lambda x:((x-min_val)/(max_val-min_val)))
		elif ntype=='sqrt':
			for col in cols:
				sketch=gl.Sketch(cf[col])
				
                        	cf[col]=cf[col].apply(lambda x:(self.signed_sqrt(x)))
				
		return cf	


	def preprocess(self, join_type='inner', encode=False, interaction=0):
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
	
		print 'Filling NAs'	
		for col in cf.column_names():
			
			cf=cf.fillna(col,gl.Sketch(cf[col]).quantile(0.5))
	
	
		#cf=self.normalize(cf,self.ntype)
		self.merged_sf=lf.join(cf , on=labels_key_col, how=join_type).fillna(labels_value_col,0)
		print 'Shape of the merged file is ', self.merged_sf.shape
		print self.merged_sf[labels_value_col].sketch_summary()
		
		if encode:
			self.merged_sf[labels_value_col]=self.merged_sf[labels_value_col].apply(lambda x:self.util.bin_encode(x))
			self.merged_sf[labels_value_col].head(2)

		print self.merged_sf[labels_value_col].sketch_summary()

		if interaction:
			if self.baseline==0:
				regexp=re.compile('\d{2,}')
				# Filtering out the spectal features with two digits in it. As the spectral digits are in reverse order of importance, so first of all I want to filter the feature with two or more digits then I want to reverse sort them	
				ic_norm=[each for each in cf.column_names() if labels_key_col not in each and clusteral_key_col not in each and labels_value_col not in each and 'norm' in each]
				ic_norm_in=sorted([each for each in ic_norm  if 'in' in each and re.search(regexp,each)], reverse=True)[:5]
				ic_norm_out=sorted([each for each in ic_norm if 'out' in each and re.search(regexp,each)], reverse=True)[:5]
				
				ic_unnorm=[each for each in cf.column_names() if labels_key_col not in each and clusteral_key_col not in each and labels_value_col not in each and 'norm' not in each]
				ic_unnorm_in=sorted([each for each in ic_unnorm  if 'in' in each and re.search(regexp,each)], reverse=True)[:5]
				ic_unnorm_out=sorted([each for each in ic_unnorm if 'out' in each and re.search(regexp,each)],reverse=True)[:5]
				
				quad_norm = fe.create(self.merged_sf, fe.QuadraticFeatures(features=ic_norm_in+ic_norm_out))	
				print 'Applying Quadratic Transformation on normalized'
				self.merged_sf=quad_norm.transform(self.merged_sf)
				quad_unnorm= fe.create(self.merged_sf, fe.QuadraticFeatures(features=ic_unnorm_in+ic_unnorm_out))
				print 'Applying Quadratic Transformation on unnormalized columns'
				self.merged_sf=quad_unnorm.transform(self.merged_sf)
			else:
				print 'Applying feature transformation in the case when the shape of the sf is low', self.merged_sf.shape
				feats=[each for each in cf.column_names() if labels_key_col not in each and clusteral_key_col not     in each and labels_value_col not in each]
				quad_transform = fe.create(self.merged_sf, fe.QuadraticFeatures(features=feats))
				self.merged_sf=quad_transform.transform(self.merged_sf)
		#self.merged_sf=self.normalize(self.merged_sf, self.ntype)
		print 'Preprocessing complete'
		#cf=self.normalize(cf,self.ntype)


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
		#train_matrix=train[train_features].to_numpy()
		#l2_penalty = 1.0/(np.sqrt(train_matrix.dot(train_matrix.T)))
		l2_penalty=10
		if classifier==gl.logistic_classifier:
			print 'train.shape', train.shape
			print 'len(train_features', len(train_features)
			model = classifier.create(train, features=train_features, target=self.labels_value_col,\
				solver='fista',l2_penalty=l2_penalty,validation_set=val, class_weights='auto',\
				max_iterations=5, convergence_threshold=0.005)#,metric='auc')
		elif classifier==gl.linear_regression:
			model = classifier.create(train, features=train_features, target=self.labels_value_col,\
				solver='fista',l2_penalty=l2_penalty,validation_set=val,\
				max_iterations=5, convergence_threshold=0.005)#,metric='auc')
			
		#model = gl.boosted_trees_classifier.create(train, features=train_features, target=self.labels_value_col,\
		#        validation_set=val, metric='auc')
		print "LL %0.20f" % self.evaluator.log_loss(model, val, col=self.labels_value_col)
		'''
		if max(predictions)>1 or min(predictions)<0:
			scaler=MinMaxScaler()
			predictions=gl.SArray(scaler.fit_transform(predictions), float)
			print 'Post Transform Predictions min: {} , max {}'.format(min(predictions), max(predictions))
		'''
		results = model.evaluate(val)
		if classifier==gl.logistic_classifier:
			predictions=model.predict(val, output_type='probability')
			print 'Predictions min: {} , max {}'.format(min(predictions), max(predictions))
			np_y=numpy.asarray(val[self.labels_value_col])
			max_accuracy, thresh=self.evaluator.calc_max_accuracy(val[self.labels_value_col],predictions)
			#auc = gl.evaluation.auc(val[self.labels_value_col],predictions)
			#print "AUC %0.20f" %auc
			#print 'Accuracy 1', results['accuracy']
			#auc = self.evaluator.custom_auc(np_y,predictions)
                        #print "Custom AUC %0.20f" %auc 
			
			auc = self.evaluator.custom_auc(np_y,predictions)
			#auc = self.evaluator.custom_auc(np_y,[0 if each <thresh else 1 for each in predictions])
                        print "Custom AUC %0.20f" %auc 
			print 'roc_auc_score_sklearn', metrics.roc_auc_score(np_y,numpy.asarray(predictions))
			print 'Max Accuracy 2', self.evaluator.calc_max_accuracy(val[self.labels_value_col],predictions)
		elif classifier==gl.linear_regression:
			predictions=model.predict(val)
			print 'Predictions min: {} , max {}'.format(min(predictions), max(predictions))
			np_y=numpy.asarray(val[self.labels_value_col])
			max_accuracy, thresh=self.evaluator.calc_max_accuracy(val[self.labels_value_col],predictions)
			fpr,tpr,_=metrics.roc_curve(np_y,predictions)
			auc=metrics.auc(fpr, tpr)
			auc = self.evaluator.custom_auc(np_y,predictions)
			#auc = self.evaluator.custom_auc(np_y,[0 if each <thresh else 1 for each in predictions])
                        print "Custom AUC %0.20f" %auc 
			print 'roc_auc_score_sklearn', metrics.roc_auc_score(np_y,numpy.asarray(predictions))
			print 'Max Accuracy 2', self.evaluator.calc_max_accuracy(val[self.labels_value_col],predictions)
			
		print 'Class Percentages in the validation data', val[self.labels_value_col].sketch_summary()
		print 'Class Percentages in the training data', train[self.labels_value_col].sketch_summary()


	def run(self):
		self.preprocess(self.join_type, self.encode, self.interaction)
		print '****************Unbalanced Sample*****************'
		self.train_bal, self.val_bal = self.merged_sf.random_split(0.75, seed=12345)
		train_features=[each for each in self.merged_sf.column_names() \
			if self.labels_key_col not in each and self.exclude not in each and self.clusteral_key_col not in each and self.labels_value_col not in each]
		print '***************Logistic****************************'
		self.fit_and_predict(gl.logistic_classifier, train_features)
		print '***************Linear****************************'
		self.fit_and_predict(gl.linear_regression, train_features)


		print '****************Balanced Sample*****************'
		self.merged_sfclass1=self.merged_sf.filter_by(1,self.labels_value_col)
		self.merged_sfclass0=self.merged_sf.filter_by(0,self.labels_value_col)

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
			if self.labels_key_col not in each and self.clusteral_key_col not in each and self.labels_value_col not in each]

		self.train_unbal, self.val_unbal = self.merged_sf.random_split(0.75, seed=12345)
		
		print '***************Logistic****************************'
		self.fit_and_predict(gl.logistic_classifier, train_features,mode='unbal')
		print '***************Linear****************************'
		self.fit_and_predict(gl.linear_regression, train_features,mode='unbal')

