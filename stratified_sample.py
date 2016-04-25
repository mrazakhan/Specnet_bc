import graphlab as gl
import sys
import csv

def twoclass_stratified_sampling(sf, col, fraction, values=[0,1]):
        sf_a=sf.filter_by(values[0], col)
        sf_b=sf.filter_by(values[1], col)
        print 'Shape of Class 1 and Class 2 Sframes is {}, {}'.format(sf_a.shape,sf_b.shape)
        sf_a,temp=sf_a.random_split(seed=12345, fraction=fraction)
        sf_b,temp=sf_b.random_split(seed=12345, fraction=fraction)

        print 'Shape of Class 1 and Class 2 Sframes after sampling is {}, {}'.format(sf_a.shape,sf_b.shape)

        merged_sf=sf_a.append(sf_b)
	print 'Shape of the merged_sf is ', merged_sf.shape
	return merged_sf

if __name__=='__main__':
	fn=sys.argv[1] # Spectral file
	mm=sys.argv[2]# MM Status
	col=sys.argv[3]
	sf=gl.SFrame.read_csv(fn)

	mf=gl.SFrame.read_csv(mm)[['orig_cid2',col]]
	sf=sf.join(mf, on='orig_cid2')
	sf2=twoclass_stratified_sampling(sf,col, fraction=0.3)
	sf2.export_csv('sample_'+col+'_'+fn,quote_level=csv.QUOTE_NONE)
