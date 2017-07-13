

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext,HiveContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from datetime import date, datetime, timedelta
from pyspark.sql import Row
from collections import OrderedDict
from operator import itemgetter
import numpy as np
str_arr=[]
arr=[]
int_arr=[]
str_arr=[]

def make_array(line):
    return np.array([float(line[0])])

def optimal_clusters(data):
    total_count = float(data.count())
    print "total_count"
    print total_count
    #num_clusters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    num_clusters = [1,2,3,4,5,6,7,8,9,10]
    dict_cost = {}
    distance = []
    for n_clus in num_clusters :
       model = KMeans.train(data, n_clus)
       cost = model.computeCost(data)
       cost = cost/total_count
       distance.append(cost)
       percent_dec = 1.0
       if n_clus!=1:
               print "old_cost,cost"
               print old_cost,cost
               try:
                percent_dec = (old_cost - cost)*100/old_cost
               except:
                   percent_dec = 0.0
               print "percent_dec"
               print percent_dec
               if percent_dec <= 0.0:
                   percent_dec = 100.0
               dict_cost[n_clus] = percent_dec
       old_cost = cost
       print "n_clus,old_cost,cost"
       print n_clus,old_cost,cost
       str_out = str(n_clus) + "," + format(cost,'.3f') + "," + str(percent_dec) + "\n"
    dict_cost_sort = OrderedDict(sorted(dict_cost.items(), key=itemgetter(1)))
    # n_clus = [key for (key, value) in sorted(dict_cost.items())][0] - 1
    n_clus = dict_cost_sort.items()[0][0] - 1
    return n_clus

def min_max_count(item_pred_item_rdd,n_clus):
    count_g = item_pred_item_rdd.countByKey()
    min_g = item_pred_item_rdd.reduceByKey(lambda a,b:a if a<b else b).collect()
    max_g = item_pred_item_rdd.reduceByKey(lambda a,b:a if a>b else b).collect()
    group_count_arr = [0]*n_clus
    for key in count_g.keys():
        group_count_arr[key] = count_g[key]
    group_min_arr = [0]*n_clus
    for item in min_g:
        group_min_arr[item[0]] =  item[1]
    group_max_arr = [0]*n_clus
    for item in max_g:
        group_max_arr[item[0]] = item[1]
    return group_count_arr, group_min_arr, group_max_arr

def sort_min_max_count(group_count_arr, group_min_arr, group_max_arr,n_clus):
    print group_count_arr
    print group_min_arr
    print "group_max_arr"
    print group_max_arr
    str_min_max_arr = []
    str_min_mx_arr_ord = []
    group_count_arr_ord = [0]*n_clus
    group_min_arr_ord = [0]*n_clus
    group_max_arr_ord = [0]*n_clus
    ttemp_arr=[x[0] for x in group_min_arr]
    array = np.array(ttemp_arr)
    print "ttemp_arr"
    print ttemp_arr
    print "array"
    print array
    temp = array.argsort()
    min_rank = np.empty(len(array), int)
    min_rank[temp] = np.arange(len(array))
    min_rank=[x+1 for x in min_rank]
    print min_rank
    count_ord = 0
    for item in min_rank:
        group_count_arr_ord[int(item)-1] = group_count_arr[count_ord]
        group_min_arr_ord[int(item)-1] = group_min_arr[count_ord]
        group_max_arr_ord[int(item)-1] = group_max_arr[count_ord]
        count_ord += 1
    for i in xrange(0,n_clus):
        str_min_max_arr.append(str(group_min_arr[i][0])+"-"+str(group_max_arr[i][0]))
    for i in xrange(0,n_clus):
        str_min_mx_arr_ord.append(str(group_min_arr_ord[i][0])+"-"+str(group_max_arr_ord[i][0]))
    return str_min_mx_arr_ord,group_count_arr_ord


for num_custom_col_name in int_arr:

    # hive_query = " select * from db.custom_col_analyse_group where user_id is not null and other_col is not null and ooo_col is not null "

    hive_query = " select "+num_custom_col_name+" from db.table where col is not null and "+num_custom_col_name+" is not null "
    # hive_query = " select " + passed_param + " from cdr.custom_col_cdr where pdate = '" + today_date + "' and " + passed_param + " is not null order by aon desc limit 1000"
    print hive_query
    sc=SparkContext()
    hc=HiveContext(sc)
    result = hc.sql(hive_query).rdd.map(make_array)
    data=result
    # all_rech=result.map(lambda x:x[1]).distinct().collect()
    # for rec_item in all_rech:
    #     d=result.filter(lambda x:x[1]==rec_item).map(lambda x:x[2])
    #     print d.take(5)
    #     points = d.map(make_array)
    #     data=points
    n_clus = optimal_clusters(data)
    model = KMeans.train(data, n_clus)
    k_means = model
    print n_clus
    bv = sc.broadcast(k_means)
    def map_model(line):
        a = bv.value
        return a.predict(line)
    pred_item_rdd = data.map(map_model)
    pred_item_rdd.take(5)
    item_pred_item_rdd = pred_item_rdd.zip(data)
    outer_count_array, outer_min_array, outer_max_array = min_max_count(item_pred_item_rdd,n_clus)
    str_min_mx_arr_ord,group_count_arr_ord = sort_min_max_count(outer_count_array, outer_min_array, outer_max_array, n_clus)
    print str_min_mx_arr_ord
    print group_count_arr_ord


    from datetime import datetime
    custom_col_date= (datetime.now()).strftime("%Y-%m-%d")
    for i in xrange(0,n_clus):
        group_count = group_count_arr_ord[i]
        min_max=str_min_mx_arr_ord[i]
        min_table,max_table=min_max.split("-")
        query = "insert into db.custom_col_cluster partition (pdate = '"+custom_col_date+"') values('"+num_custom_col_name+"',"+min_table+","+max_table+",null,"+group_count+")"
        print query
        out=hc.sql(query)
        print out
        #
        # line = ""
        # line += str_min_mx_arr_ord[i] + "," + str(group_count_arr_ord[i]) + "\n"
        # print "line"
        # print line
        # str_csv += line
    # print str_csv
    # file_name = "/home/hadoop/spark-1.6.0/cluster_out2/" + "cluster_"+str(rec_item)+".csv"
    # f_open = open(file_name,"w")
    # f_open.write(str_csv)

for str_col_name in str_arr:
    query = "insert into db.table partition (pdate = '"+custom_col_date+"') select '"+str_col_name+"' null, null,"+str_col_name+",count(*) from custom_col_final group by "+str_col_name+" "
    print query
    out=hc.sql(query)
    print out
