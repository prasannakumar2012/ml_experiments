#http://researchcomputing.github.io/RMACC_2014_python/spark/04_kmeans.html
#http://matplotlib.org/users/pyplot_tutorial.html
import os
import sys


# Path for spark source folder


os.environ['SPARK_HOME']="/Users/prasanna/mike/spark/"

# Append pyspark  to Python Path
sys.path.append("/Users/prasanna/mike/spark/python/")

import pyspark
from pyspark import SparkContext, SparkConf


import os
import numpy as np
N = 5
# N= 2
num_points = 100

set1 = np.reshape(1 + np.random.randn(num_points*N), (num_points,N))
set2 = np.reshape(5 + np.random.randn(num_points*N), (num_points,N))
set3 = np.reshape(10 + np.random.randn(num_points*N), (num_points,N))

x = np.concatenate([set1, set2, set3])

np.savetxt('points2.txt', x, delimiter=',', fmt='%10.5f')


from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

data = np.loadtxt('points2.txt', delimiter=',')
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(data)
print "data"
# print data
print len(data)
print data[0]
print data[1]
print k_means.cluster_centers_
print len(k_means.cluster_centers_)

#%matplotlib inline
import matplotlib.pyplot as plt

def plot(centers):
    for i in range(3):

        plt.plot(centers[i][0])

    # for i in range(len(data)):
    #     plt.plot(data[i])

    plt.show()


# plot(data)

plot(k_means.cluster_centers_)