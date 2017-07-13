"""

    for item in data.collect():
        final_array = []
        pred_item = k_means.predict(item)
        group_count_arr[pred_item] +=1
        if item < group_min_arr[pred_item]:
            group_min_arr[pred_item] = item
        if item > group_max_arr[pred_item]:
            group_max_arr[pred_item] = item


    print group_count_arr
    print group_min_arr
    print group_max_arr
    str_min_max_arr = []
    str_min_mx_arr_ord = []
    group_count_arr_ord = [0]*n_clus
    group_min_arr_ord = [0]*n_clus
    group_max_arr_ord = [0]*n_clus
    min_rank = ss.rankdata(group_min_arr)
    count_ord = 0
    for item in min_rank:
        group_count_arr_ord[int(item)-1] = group_count_arr[count_ord]
        group_min_arr_ord[int(item)-1] = group_min_arr[count_ord]
        group_max_arr_ord[int(item)-1] = group_max_arr[count_ord]
        count_ord += 1

    print group_count_arr_ord
    print group_min_arr_ord
    print group_max_arr_ord
    print "before min max array"
    print n_clus
    print str_min_max_arr
    print str_min_mx_arr_ord
    print n_clus
    for i in xrange(0,n_clus):
        print "first"
        str_min_max_arr.append(str(group_min_arr[i][0])+"-"+str(group_max_arr[i][0]))
    print "before second"
    for i in xrange(0,n_clus):
        print "second"
        str_min_mx_arr_ord.append(str(group_min_arr_ord[i][0])+"-"+str(group_max_arr_ord[i][0]))
    print "after min max array"
    print str_min_max_arr
    print str_min_mx_arr_ord
    print datetime.now()


    print "start plotting"
    print datetime.now()




"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext,HiveContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from datetime import date, datetime
from pyspark.sql import Row
import numpy as np
from collections import OrderedDict
from operator import itemgetter
import sys
import scipy.stats as ss
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mpld3
from mpld3 import plugins, utils


sc = SparkContext("spark://master-ip:7077", "appname")
hc = HiveContext(sc)
def make_array(line):
    return np.array([float(line[0])])

kpi = "aon"
passed_param = str(kpi)


print "start"
print datetime.now()

hive_query = " select " + passed_param + " from db.table where pdate = '2016-02-17' and " + passed_param + " is not null limit 1000"

result = hc.sql(hive_query)

print result.take(5)
points = result.rdd.map(make_array)
print points.take(5)
print "points done"
data = points

n_clus = 4

model = KMeans.train(data, n_clus)

k_means = model

bv = sc.broadcast(k_means)

def map_model(line):
    a = bv.value
    return a.predict(line)


# group_count_arr = [0]*n_clus
# group_min_arr = [0]*n_clus
# group_max_arr = [0]*n_clus
#

pred_item_rdd = data.map(map_model)
item_pred_item_rdd = pred_item_rdd.zip(data)
# item_pred_item_rdd = data.zip(pred_item_rdd)
# print "After zip"
# print item_pred_item_rdd.take(1)
# print item_pred_item_rdd.take(10)
# print datetime.now()

count = item_pred_item_rdd.countByKey()
max = item_pred_item_rdd.reduceByKey(lambda a,b:max(a,b))
min = item_pred_item_rdd.reduceByKey(lambda a,b:min(a,b))

print count.collect()
print max.collect()
print min.collect()


print "End"

print datetime.now()

