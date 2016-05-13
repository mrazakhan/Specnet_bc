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

DEBUG=1


class Eval():
#https://raw.githubusercontent.com/benhamner/Metrics/master/Python/ml_metrics/auc.py
	"""
	Computes the tied rank of elements in x.

	This function computes the tied rank of elements in x.

	Parameters
	----------
	x : list of numbers, numpy array

	Returns
	-------
	score : list of numbers
	    The tied rank f each element in x
	"""
	def tied_rank(self,x):
		sorted_x = sorted(zip(x,range(len(x))))
		r = [0 for k in x]
		cur_val = sorted_x[0][0]
		last_rank = 0
		for i in range(len(sorted_x)):
			if cur_val != sorted_x[i][0]:
				cur_val = sorted_x[i][0]
				for j in range(last_rank, i): 
					r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
				last_rank = i
			if i==len(sorted_x)-1:
				for j in range(last_rank, i+1): 
					r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
		return r

	def custom_auc(self,actual, posterior):
		"""
		Computes the area under the receiver-operater characteristic (AUC)

		This function computes the AUC error metric for binary classification.

		Parameters
		----------
		actual : list of binary numbers, numpy array
		     The ground truth value
		posterior : same type as actual
			Defines a ranking on the binary numbers, from most likely to
			be positive to least likely to be positive.

		Returns
		-------
		score : double
		    The mean squared error between actual and posterior

		"""
		#actual=[1 if each <0.5 else 0 for each in actual]
		r = self.tied_rank(posterior)
		num_positive = len([0 for x in actual if x==1])
		num_negative = len(actual)-num_positive
		sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
		auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
		   (num_negative*num_positive))
		return auc


	
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

	def calc_max_accuracy(self,labels, predictions_probs):
		max_accuracy=0
		best_thresh=0
		predictions=[]
		for thres in np.arange(min(predictions_probs),max(predictions_probs)+0.1,0.005):
			predictions=predictions_probs.apply(lambda x:0 if x<thres else 1)
			acc=gl.evaluation.accuracy(labels, predictions)
			if acc >max_accuracy:
				max_accuracy=acc
				best_thresh=thres
		print 'Best threshold for accuracy', best_thresh
		return max_accuracy,best_thresh

