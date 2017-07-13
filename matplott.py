import numpy as np, matplotlib.pyplot as plt, mpld3, seaborn as sns
# fig, ax = plt.subplots()
list = [['0-50',4],['50-100',11],['100-150',73],['150-200',46]]
n_groups = len(list)
index = np.arange(n_groups)

bar_width = 0.9
opacity = 0.4

number = []
ranges = []
a = plt.figure(figsize=(20,13))
for item in list:
    number.append(item[1])
    ranges.append(item[0])

rects1 = plt.bar(index, number, bar_width,
                 alpha=opacity,
                 color='b')

plt.xlabel('Number')
plt.ylabel('Range')
# plt.xticks(index + bar_width/2, (ranges[0],ranges[1],ranges[2],ranges[3]))

plt.xticks(index + bar_width/2, ranges)
plt.show()
# mpld3.show()
# fig, ax = plt.subplots()
# print mpld3.display()
# mpld3.save_html(a,'/Users/prasanna/from_server/mpldhtml4.html')