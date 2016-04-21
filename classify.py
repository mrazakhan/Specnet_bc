import pandas as pd
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import sys
import argparse
def accuracy(y_true, y_pred):
	return np.mean(y_true == y_pred)

if __name__=='__main__':
	df=pd.read_csv(sys.argv[1])
	print 'Gender profile before shuffling', df.gend.head()
	df=df.sample(df.shape[0])
	print 'Gender profile after shuffling', df.gend.head()
	print 'Gend Distribution Overall', df['gend'].value_counts()
	y=df.gend
	df=df.drop(['old_cid','msisdn','gend'], axis=1)
	print df.columns
	X = df.as_matrix().astype(np.float)
	X_Train,X_Test,y_train,y_test=train_test_split(X,y)
	print 'Gend Distribution Test', y_test.value_counts()
	#clf=LogisticRegressionCV()
	clf=LogisticRegressionCV(solver='liblinear',penalty='l1',n_jobs=-1, cv=5)
	clf.fit(X_Train,y_train)
	y_pred=clf.predict(X_Test)
	print 'Gend Distribution Test Prediction', np.bincount(y_pred), sum(y_pred)
	print "%s Test %.4f" % ('CVLogistic',accuracy(y_test, y_pred))
