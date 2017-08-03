from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext,HiveContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.feature import PCA
# import matplotlib.pyplot as plt
import numpy as np
import json
from pyspark.mllib.linalg import Vectors

from datetime import datetime

sc = SparkContext()

with open('kmeans_pca_conf.json') as conf_file:
    conf = json.load(conf_file)


def parsePoint(line):
    values = [str(s) for s in line.split(',')]
    values_arr = []
    try:
        values_arr.append(float(values[0]))
    except:
        values_arr.append(0)
    try:
        start = datetime.strptime(str(values[1]), "%Y-%m-%d")

        diff = (datetime.now() - datetime.combine(start, datetime.min.time()) ).days
        values_arr.append(float(diff))
    except:
        values_arr.append(0)
    try:
        if str(values[2]) == "2G":
            values_arr.append(0)
        if str(values[2]) == "3G":
            values_arr.append(1)
        if str(values[2]) == "4G":
            values_arr.append(1)
    except:
        values_arr.append(0)
    try:
        values_arr.append(float(values[3]))
    except:
        values_arr.append(0)
    try:
        values_arr.append(float(values[4]))
    except:
        values_arr.append(0)
    try:
        values_arr.append(float(values[5]))
    except:
        values_arr.append(0)
    final_array = []

    for i in range(len(values_arr)):
        if str(i) in index_of_features_to_take_array:
            final_array.append(float(str(values_arr[i])))


    return np.array(final_array)



"""
Msisdn, Aon, Handset,  Data Usage, Maximum recharge amount,Data Arpu

"""

input_file_path = conf["input_file_path"]
input_separated_by = conf["input_separated_by"]
index_of_features_to_take = conf["index_of_features_to_take"]
index_of_features_to_take_array = str(index_of_features_to_take).split(",")
# print index_of_features_to_take_array
num_clusters = conf["num_clusters"]
output_file_path = conf["output_file_path"]
output_separated_by = conf["num_clusters"]
# output_plot_path = conf["output_plot_path"]

# lines = sc.textFile("file:///Users/prasanna/scrapy_projects/points2.txt")


data = sc.textFile(input_file_path)
points = data.map(parsePoint)
data = points


if num_clusters == "All":
    distance = []
    num_clusters = [1,2,3,4,5]
    out_file = open(output_file_path,"w")
    str_out = "num_clusters,distance" + "\n"
    out_file.write(str_out)

    for n_clus in num_clusters :
        model = KMeans.train(data, n_clus)
        distance.append(model.computeCost(data))
        str_out = str(n_clus) + "," + str(distance) + "\n"
        out_file.write(str_out)
    # plt.plot(num_clusters,distance, '-o' )
    # plt.savefig(output_plot_path)
    # plt.show()

else:
    n_clus = int(str(num_clusters))
    model = KMeans.train(data, n_clus)
    k_means = model
    sqlContext = SQLContext(sc)
    pca_data = []
    for item in data.collect() :
        pca_data.append((Vectors.dense(item),),)
    df = sqlContext.createDataFrame(pca_data,["features"])
    pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(df)
    result = model.transform(df)

    with open(output_file_path,"w") as out_file:
        index_of_features_to_take_array.append("cluster_number")
        str_out = ",".join(index_of_features_to_take_array)
        out_file.write(str_out)
        out_file.write("\n")

    for item in result.collect():
        twod = item.pcaFeatures
        item = item.features
        pred_item = k_means.predict(item)
        item_write = item

        item_write = [str(x) for x in item_write]
        item_write.append(str(pred_item))
        with open(output_file_path,"a") as out_file:
            str_out = ",".join(item_write)
            out_file.write(str_out)
            out_file.write("\n")

        # if (pred_item)==0:
        #     plt.plot(twod[0], twod[1], 'ro' , color = 'green')
        # if (pred_item)==1:
        #     plt.plot(twod[0], twod[1], 'ro' , color = 'black')
        # if (pred_item)==2:
        #     plt.plot(twod[0], twod[1], 'ro' , color = 'pink')
        # if (pred_item)==3:
        #     plt.plot(twod[0], twod[1], 'ro' , color = 'yellow')
        # if (pred_item)==4:
        #     plt.plot(twod[0], twod[1], 'ro' , color = 'blue')
        # if (pred_item)==5:
        #     plt.plot(twod[0], twod[1], 'ro' , color = 'brown')

    pca_center = []
    for item in k_means.clusterCenters:
        pca_center.append((Vectors.dense(item),),)
    df = sqlContext.createDataFrame(pca_center,["features"])
    result = model.transform(df)
    i = 1
    for item in result.collect():
        if i % 3 == 0:
            factor = 0
        if i % 3 == 1:
            factor = 1
        if i % 3 == 2:
            factor = -1
        i = i+1
        twod = item.pcaFeatures
        actual = item.features
        actual = [float("{0:.2f}".format(x)) for x in actual]
        # plt.annotate(str(actual), xy=(-10, -1), xytext=(twod[0], twod[1]+factor))
        # plt.plot(twod[0], twod[1], 'ro' , color = 'red')

# plt.savefig(output_plot_path)
# plt.show()