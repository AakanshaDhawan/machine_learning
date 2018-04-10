import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math


df = pd.read_csv('/home/aakansha/Desktop/ml_algo/linear_regression/data.csv', header = None, names = ['x','y'])
x = np.array(df.x)
y = np.array(df.y)
theta = np.zeros((2,1))
  
print (x)
print (y)

x.reshape(-1,1)
y.reshape(-1,1)

print (x)
print (y)

def scatterPlot(x,y,yp,savePng=0):
	plt.xlabel('Population of City in 10,000s')
	plt.ylabel('Profit in $10,000s')
	plt.scatter(x, y, marker='x')
	if yp != None:
		plt.plot(x,yp)
	if savePng==0:
		plt.show()
	else:
		name = input('Name Figure File: ')
		plt.savefig(name+'.png')

print (scatterPlot(x,y,None))

#linear regression implementation using libraries
(m,b) = np.polyfit(x,y,1)
print ('Slope is ' + str(m))
print ('Y intercept is ' + str(b))
yp = np.polyval([m,b],x)
#scatterPlot(x,y,yp)