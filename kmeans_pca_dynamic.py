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

distance = []
num_clusters = [1,2,3,4,5]

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

data = np.loadtxt('points2.txt', delimiter=',')

print data[0]



if sys.argv[1] == "All":
    print "All"
    distance = []
    num_clusters = [1,2,3,4,5]
    for n_clus in num_clusters :
        k_means = KMeans(init='k-means++', n_clusters=n_clus, n_init=10)
        k_means.fit(data)
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(data)
        distance.append(k_means.inertia_)

    plt.plot(num_clusters,distance, '-o' )
    plt.show()


else:
    n_clus = int(str(sys.argv[1]))
    k_means = KMeans(init='k-means++', n_clusters=n_clus, n_init=10)
    k_means.fit(data)
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(data)


    for item in data:
        twod = pca.transform(item)
        if (k_means.predict(item))[0]==0:
            plt.plot(twod[0][0], twod[0][1], 'ro' , color = 'green')
        if (k_means.predict(item))[0]==1:
            plt.plot(twod[0][0], twod[0][1], 'ro' , color = 'black')
        if (k_means.predict(item))[0]==2:
            plt.plot(twod[0][0], twod[0][1], 'ro' , color = 'pink')

        if (k_means.predict(item))[0]==3:
            plt.plot(twod[0][0], twod[0][1], 'ro' , color = 'yellow')

        if (k_means.predict(item))[0]==4:
            plt.plot(twod[0][0], twod[0][1], 'ro' , color = 'blue')

        if (k_means.predict(item))[0]==5:
            plt.plot(twod[0][0], twod[0][1], 'ro' , color = 'brown')

    for item in k_means.cluster_centers_:
        twod = pca.transform(item)
        plt.plot(twod[0][0], twod[0][1], 'ro' , color = 'red')

plt.show()
