import argparse
import graphlab as gl
import csv

class MMFreq:

	def __init__(self, fileName, sep, userIdIndex, dateIndex, cdrType,start_date, end_date, outputFileName, filter_dates ):
		self.fileName=fileName
		self.sep=sep
		self.userIdIndex=userIdIndex
		self.dateIndex=dateIndex
		self.cdrType=cdrType # Pak, Ghana or Zambia
		self.start_date=start_date
		self.end_date=end_date
		self.outputFileName=outputFileName
		self.filter_dates=filter_dates		



	def processPakistanDate(self,filter_dates):
		self.sf['str_date']=self.sf[self.header[self.dateIndex]].apply(lambda x:str(x).split(' ')[0]) # Getting the date only
		if filter_dates:
			print 'Shape before filtering the dates', self.sf.shape
			self.sf=self.sf[self.sf['str_date'].apply(lambda x:(x<=self.end_date and x>=self.start_date))]
			print 'Shape after filtering the dates', self.sf.shape
		self.sf[self.sf.column_names()[self.dateIndex]]=self.sf['str_date'].apply(lambda x:str(x)[:7])
		return self.sf

	def run(self):
		self.sf=gl.SFrame.read_csv(self.fileName, delimiter=self.sep)
		self.header=header=self.sf.column_names()
		to_keep_inds=[self.userIdIndex, self.dateIndex]
		to_keep_cols=[header[i] for i in range(len(header)) if i in to_keep_inds]
		self.sf=self.sf[to_keep_cols]
		
		if self.cdrType=='Pakistan':
			self.sf=self.processPakistanDate(self.filter_dates)
		print self.sf.head()
		self.sf=self.sf.groupby(key_columns=header[self.userIdIndex],operations={'NoOfMonths':gl.aggregate.COUNT_DISTINCT('str_date')})

		print 'Shape of SFrame before removing non subscriber ids', self.sf.shape
		self.sf=self.sf[self.sf[header[self.userIdIndex]].apply(lambda x:str(x).isdigit())]
		print 'Shape of SFrame after removing non subscriber ids', self.sf.shape
		#print self.sf.head()
		self.sf[header[self.userIdIndex]]=self.sf[header[self.userIdIndex]].apply(lambda x:str(int(x)).zfill(9))		

		self.sf.rename({header[self.userIdIndex]:'UserId'})
		print self.sf['NoOfMonths'].sketch_summary()
		self.sf.export_csv(self.outputFileName, quote_level= csv.QUOTE_NONE)

def main():
	parser = argparse.ArgumentParser(description='MobileMoneyFrequency')
	parser.add_argument('-if', '--input_file_name', help='mobile money file', required=True)
	parser.add_argument('-ui', '--user_index', help='UserId Index', required=True)
	parser.add_argument('-di', '--date_index', help='Date Index', required=True)
	parser.add_argument('-ct', '--cdr_type', help='cdr type Ghana, Zambia or Pakistan', required=True)
	parser.add_argument('-of', '--output_file_name', help='output file', required=True)
	parser.add_argument('-st_date', '--start_date', help='start date  format', required=True)
	parser.add_argument('-fd', '--filter_dates', help='Filter dates or not', required=True)
	parser.add_argument('-end_date', '--end_date', help='end date', required=True)

	args = parser.parse_args()

	mmf=MMFreq(fileName=args.input_file_name,sep='\t',userIdIndex=int(args.user_index), dateIndex=int(args.date_index), cdrType=args.cdr_type, outputFileName=args.output_file_name, start_date=args.start_date, end_date=args.end_date,filter_dates=int(args.filter_dates))
	mmf.run()

if __name__=='__main__':
	main()
		
				
		
