
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
data = pd.read_csv("/Users/prasanna/Downloads/temp_late_ontime.csv")
data.hist()
plt.show()
data['STATUS'].value_counts().plot(kind='bar')
plt.show()
sns.set() #rescue matplotlib's styles from the early '90s

"""
"FULFILLMENT_DAYS,PRODUCT_QTY
"""
#for numerical columns
data.hist(by='STATUS',column = 'PRODUCT_QTY')
data.hist(by='STATUS',column = 'FULFILLMENT_DAYS')
plt.show()


"""
          ORDERSERIALNO ORDER_COMPLEXITY     PROD_BASE_SKU  PRODUCT_QTY  \
0  001015434433-9000024           SIMPLE  FRMPRNT_11X14_CC            1
1  005081934313-9000050          COMPLEX    DESIGNERCARD55           25
2  005089548353-7000036           SIMPLE  GLITTER5X7GOLD10           15
3  001064570673-9000024           SIMPLE     6X8FLATCARD06           25
4  007012634278-7000274          COMPLEX   5X7FOLDEDCARD06            1

  PRODUCT_QTY_CATEGORY        ORDER_DATE_TIME  FULFILLMENT_DAYS  \
0                  LOW  2016-11-03 13:49:48.0                 1
1                 HIGH  2016-11-13 23:07:07.0                 0
2               MEDIUM  2016-11-26 12:55:30.0                 2
3                 HIGH  2016-12-11 18:52:03.0                 4
4                  LOW  2016-11-13 19:05:13.0                 5

          SHIP_DATE_TIME     EXPECTED_SHIP_DATE  STATUS       ...        \
0  2016-11-04 10:12:00.0  2016-11-10 00:00:00.0  ONTIME       ...
1  2016-11-14 14:00:53.0  2016-11-18 00:00:00.0  ONTIME       ...
2  2016-11-30 16:35:41.0  2016-12-09 00:00:00.0  ONTIME       ...
3  2016-12-16 15:05:51.0  2016-12-14 00:00:00.0    LATE       ...
4  2016-11-21 14:37:37.0  2016-11-21 00:00:00.0    LATE       ...

  FULFILLER_CATEGORY FULFILLER_DESCRIPTION FULFILLER_COMPLEXITY  \
0           EXTERNAL                  FUJI               SIMPLE
1           INTERNAL        UNITY_SHAKOPEE               SIMPLE
2           EXTERNAL                VISION               SIMPLE
3           INTERNAL        UNITY_SHAKOPEE               SIMPLE
4           INTERNAL        UNITY_FORTMILL              COMPLEX

  PRODUCT_CATEGORY_COMPLEXITY RECIPIENT_COMPLEXITY SHIPMENT_COMPLEXITY  \
0                      SIMPLE               SIMPLE     SINGLE SHIPMENT
1                      SIMPLE               SIMPLE     SINGLE SHIPMENT
2                      SIMPLE               SIMPLE     SINGLE SHIPMENT
3                      SIMPLE               SIMPLE     SINGLE SHIPMENT
4                     COMPLEX               SIMPLE  MULTIPLE SHIPMENTS

                    SPLIT_LOGIC             SPLIT_LOGIC_DESC PRAFLAG  \
0   SIMPLE ORDER & ONE SHIPMENT  SIMPLE ORDER & ONE SHIPMENT   NOPRA
1  COMPLEX ORDER & ONE SHIPMENT           FULL CONSOLIDATION   NOPRA
2   SIMPLE ORDER & ONE SHIPMENT  SIMPLE ORDER & ONE SHIPMENT   NOPRA
3   SIMPLE ORDER & ONE SHIPMENT  SIMPLE ORDER & ONE SHIPMENT   NOPRA
4         COMPLEX ORDER & SPLIT      FULL SPLIT BY FULFILLER   NOPRA

  MEMORABILIAFLAG
0    BOOKNOPOCKET
1    BOOKNOPOCKET
2    BOOKNOPOCKET
3    BOOKNOPOCKET
4    BOOKNOPOCKET
"""

#for categorical columns
ag = data.groupby('STATUS')["PRODUCT_CATEGORY_COMPLEXITY"].value_counts().sort_index()
print ag
ag.unstack()
ag.unstack().plot(kind='bar', subplots=True, layout=(2,2))
plt.show()


ag.unstack().plot(kind='bar', subplots=True, layout=(2,2))

#data.plot.scatter('STATUS', 'PRODUCT_CATEGORY_COMPLEXITY')




df = data

mylist = list(df.select_dtypes(include=['object']).columns)
exclusion_list = ['ORDERSERIALNO','ORDER_DATE_TIME','SHIP_DATE_TIME','EXPECTED_SHIP_DATE']
for i in mylist:
    if i not in exclusion_list:
        df[i] = df[i].astype('category')

newlist = list(df.select_dtypes(include=['category']).columns)
#print newlist
#print list(df.select_dtypes(include=['object']).columns)

print df.dtypes

df[newlist] = df[newlist].apply(lambda x: x.cat.codes)
print df

data.plot.scatter('STATUS', 'PRODUCT_CATEGORY_COMPLEXITY')
plt.show()
data.plot.scatter('STATUS', 'FULFILLMENT_DAYS')
plt.show()


import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
data = pd.read_csv("/Users/prasanna/Downloads/temp_late_ontime.csv")
df = data
for item in df.columns:
    if item != 'STATUS':
        grouped = df.groupby(['STATUS',item]).agg({'ORDERSERIALNO': 'count'})
        grouped_per = grouped.groupby(level=1).apply(lambda x:100 * x / float(x.sum()))
        print grouped_per


import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
data = pd.read_csv("/Users/prasanna/Downloads/temp_late_ontime.csv")
df = data
fig = plt.figure()
total_count = 2
count = 1
for item in df.columns:
    if item in ['PRODUCT_CATEGORY_COMPLEXITY','SHIPMENT_COMPLEXITY']:
        grouped = df.groupby(['STATUS',item]).agg({'ORDERSERIALNO': 'count'})
        grouped_per = grouped.groupby(level=1).apply(lambda x:100 * x / float(x.sum()))
        print grouped_per
        int_str = int("1"+str(total_count)+str(count))
        try:
            ax1 = fig.add_subplot(int_str,sharey=share_ax)
        except:
            ax1 = fig.add_subplot(int_str)
        if count==1:
            share_ax = ax1
        fil = grouped_per['ORDERSERIALNO'].filter(like='LATE', axis=0)
        # fil.plot.line()
        fil.plot(style='o')
        count+=1


plt.show()



import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
data = pd.read_csv("/Users/prasanna/Downloads/temp_late_ontime.csv")
df = data
fig = plt.figure()
all_columns_arr = ['PRODUCT_CATEGORY_COMPLEXITY','SHIPMENT_COMPLEXITY']
total_count = len(all_columns_arr)
count = 1
for item in df.columns:
    if item in all_columns_arr:
        grouped = df.groupby(['STATUS',item]).agg({'ORDERSERIALNO': 'count'})
        grouped_per = grouped.groupby(level=1).apply(lambda x:1 * x / float(x.sum()))
        print grouped_per
        int_str = int("1"+str(total_count)+str(count))
        ax1 = fig.add_subplot(int_str)
        ax1.set_ylim(0, 1)
        fil = grouped_per['ORDERSERIALNO'].filter(like='LATE', axis=0)
        # fil.plot.line()
        fil.plot(style='o')
        # for iter in len(fil):
        #     ax1.annotate(str(iter), xy=(i, ))
        count+=1


plt.show()




import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
data = pd.read_csv("/Users/prasanna/Downloads/temp_late_ontime.csv")
df = data
fig = plt.figure()
all_columns_arr = ['PRODUCT_CATEGORY_COMPLEXITY','SHIPMENT_COMPLEXITY']
all_columns_arr = ['PRODUCT_QTY','SHIPMENT_COMPLEXITY']
total_count = len(all_columns_arr)
count = 1
for item in df.columns:
    if item in all_columns_arr:
        grouped = df.groupby(['STATUS',item]).agg({'ORDERSERIALNO': 'count'})
        grouped_per = grouped.groupby(level=1).apply(lambda x:1 * x / float(x.sum()))
        print grouped_per
        int_str = int("2"+str(total_count)+str(count))
        ax1 = fig.add_subplot(int_str)
        ax1.set_ylim(0, 1)
        fil = grouped_per['ORDERSERIALNO'].filter(like='LATE', axis=0)
        # fil2 = grouped_per['ORDERSERIALNO'].filter(like='ONTIME', axis=0)
        # fil.plot.line()
        fil.plot(style='o')
        # fil2.plot(style='--',color='b')
        # for iter in len(fil):
        #     ax1.annotate(str(iter), xy=(i, ))
        count+=1

for item in df.columns:
    if item in all_columns_arr:
        grouped = df.groupby(['STATUS', item]).agg({'ORDERSERIALNO': 'count'})
        grouped_per = grouped.groupby(level=1).apply(lambda x: 1 * x / float(x.sum()))
        print grouped_per
        int_str = int("2" + str(total_count) + str(count))
        ax1 = fig.add_subplot(int_str)
        ax1.set_ylim(0, 1)
        fil = grouped_per['ORDERSERIALNO'].filter(like='ONTIME', axis=0)
        # fil2 = grouped_per['ORDERSERIALNO'].filter(like='ONTIME', axis=0)
        # fil.plot.line()
        fil.plot(style='o')
        # fil2.plot(style='--',color='b')
        # for iter in len(fil):
        #     ax1.annotate(str(iter), xy=(i, ))
        count += 1

plt.show()




import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
np.random.seed(0)
x, y = np.random.normal(size=(2, 200))
color, size = np.random.random((2, 200))

ax.scatter(x, y, c=color, s=500 * size, alpha=0.3)
ax.grid(color='lightgray', alpha=0.7)

import mpld3
mpld3.show()


