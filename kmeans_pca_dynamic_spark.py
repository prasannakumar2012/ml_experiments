#http://researchcomputing.github.io/RMACC_2014_python/spark/04_kmeans.html
#http://matplotlib.org/users/pyplot_tutorial.html
#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

import os
import sys


# Path for spark source folder


os.environ['SPARK_HOME']="/Users/prasanna/mike/spark/"

# Append pyspark  to Python Path
sys.path.append("/Users/prasanna/mike/spark/python/")

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext,HiveContext

import os
import numpy as np
N = 5
# N= 2
num_points = 100

set1 = np.reshape(1 + np.random.randn(num_points*N), (num_points,N))
set2 = np.reshape(5 + np.random.randn(num_points*N), (num_points,N))
set3 = np.reshape(10 + np.random.randn(num_points*N), (num_points,N))

x = np.concatenate([set1, set2, set3])

# np.savetxt('points2.txt', x, delimiter=',', fmt='%10.5f')


# from sklearn.cluster import KMeans
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.feature import PCA
from sklearn.metrics.pairwise import euclidean_distances


def parseVector(line):
    arr = []
    for x in line.split(','):
        y = x.split(',')
        for item in y:
            arr.append(float(str(item).strip()))
        # return np.array([float(str(x).strip()) for x in line.split(',')])

    return np.array(arr)

sc = SparkContext()

lines = sc.textFile("file:///Users/prasanna/scrapy_projects/points2.txt")
# print lines.take(1)
# print lines.take(5)
data = lines.map(parseVector)
print data.collect()
# data = np.loadtxt('points2.txt', delimiter=',')

# print data[0]



if sys.argv[1] == "All":
    print "All"
    distance = []
    num_clusters = [1,2,3,4,5]
    for n_clus in num_clusters :
        # k_means = KMeans(init='k-means++', n_clusters=n_clus, n_init=10)
        model = KMeans.train(data, n_clus)
        # k_means.fit(data)
        import matplotlib.pyplot as plt
        import numpy as np
        # from sklearn.decomposition import PCA
        # print "data"
        # print data.take(1)
        # pca = PCA(2)
        # pca.fit(data)
        # pca = PCA(n_components=2)
        # pca.fit(data)
        distance.append(model.computeCost(data))

    plt.plot(num_clusters,distance, '-o' )
    plt.show()


else:
    n_clus = int(str(sys.argv[1]))
    model = KMeans.train(data, n_clus)
    k_means = model
    # k_means = KMeans(init='k-means++', n_clusters=n_clus, n_init=10)
    # k_means.fit(data)
    import matplotlib.pyplot as plt
    import numpy as np
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # pca.fit(data)
    from pyspark.mllib.linalg import Vectors
    sqlContext = SQLContext(sc)
    pca_data = []
    for item in data.collect() :
        pca_data.append((Vectors.dense(item),),)
    df = sqlContext.createDataFrame(pca_data,["features"])
    pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(df)
    result = model.transform(df)



    for item in result.collect():
        twod = item.pcaFeatures
        # twod = pca.transform(item)
        item = item.features

        # print "item"
        # print item
        # print k_means.predict(item)
        # print k_means.predict(item).collect()

        if (k_means.predict(item))==0:
            plt.plot(twod[0], twod[1], 'ro' , color = 'green')
        if (k_means.predict(item))==1:
            plt.plot(twod[0], twod[1], 'ro' , color = 'black')
        if (k_means.predict(item))==2:
            plt.plot(twod[0], twod[1], 'ro' , color = 'pink')

        if (k_means.predict(item))==3:
            plt.plot(twod[0], twod[1], 'ro' , color = 'yellow')

        if (k_means.predict(item))==4:
            plt.plot(twod[0], twod[1], 'ro' , color = 'blue')

        if (k_means.predict(item))==5:
            plt.plot(twod[0], twod[1], 'ro' , color = 'brown')

    pca_center = []

    for item in k_means.clusterCenters:
        pca_center.append((Vectors.dense(item),),)
    df = sqlContext.createDataFrame(pca_center,["features"])
    # pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
    # model = pca.fit(df)
    result = model.transform(df)
    for item in result.collect():
        twod = item.pcaFeatures
        plt.plot(twod[0], twod[1], 'ro' , color = 'red')

plt.show()
