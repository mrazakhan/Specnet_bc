import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import sys
import argparse
def accuracy(y_true, y_pred):
	return np.mean(y_true == y_pred)

if __name__=='__main__':
	df=pd.read_csv(sys.argv[1])
	label=sys.argv[2]
	print '  shuffling'
	df=df.sample(df.shape[0])#.head(100000)
	#print 'Gender profile after shuffling', df.gend.head()
	print 'Truth Distribution Overall', df[label].value_counts()
	y=df[label]
	try:
		df=df.drop(['orig_cid2','old_cid','msisdn','gend'], axis=1)
	except:
		pass
		#print df.columns
	X = df.as_matrix().astype(np.float)
	X_Train,X_Test,y_train,y_test=train_test_split(X,y)
	print 'Labels Distribution Test', y_test.value_counts()
	#clf=LogisticRegressionCV()
	clf=LogisticRegressionCV(solver='liblinear',penalty='l1', cv=3)
	#clf=LogisticRegressionCV(penalty='l2', cv=5, n_jobs=-1)
	clf.fit(X_Train,y_train)
	y_pred=clf.predict(X_Test)
	#print 'Truth Distribution Test Prediction', np.bincount(y_pred), sum(y_pred)
	print "%s Test Accuracy %.4f" % ('CVLogistic',accuracy(y_test, y_pred))
	print 'ROC: %f', roc_auc_score(y_test, y_pred)
	print 'LogLoss: %f', log_loss(y_test, y_pred)
