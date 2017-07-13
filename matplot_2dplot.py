import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import numpy


x = [i for i in range(10)]
y = [i**2 for i in range(10)]
z = [i**3 for i in range(10)]

# z=[0.15, 0.3, 0.45, 0.6, 0.75]
# y=[2.56422, 3.77284,3.52623,3.51468,3.02199]

# n=[58,651,393,203,123]

corr1 = pearsonr(x,x)[0]
corr2 = pearsonr(x,y)[0]
corr3 = pearsonr(x,z)[0]

corr1  = float("{0:.2f}".format(corr1))
corr2  = float("{0:.2f}".format(corr2))
corr3  = float("{0:.2f}".format(corr3))
# z_mean = numpy.mean(z)
# y_mean = numpy.mean(y)

# fig, ax = plt.subplots()

# ax.scatter(z, y)

x = [1,2,3,4,5,6,7,8,9,10,11,12]
y = [1138936.995,354284.931,159093.569,96895.811,59584.507,41185.969,30874.905,23366.439,18787.987,15604.755,12841.863,11400.533]
plt.plot(x,y, '-o' )
# plt.plot(x,x, '-o' )
# plt.plot(x,y, '-o' )
# plt.plot(x,z, '-o' )
# plt.annotate("Correlation = "+str(corr), xy=(2, 1), xytext=(3, 1.5),
#              arrowprops=dict(facecolor='black', shrink=0.05))

# plt.annotate("Correlation = "+str(corr), xy=(2, 1), xytext=(3, 1.5))
plt.annotate("Corr = "+str(corr1), xy=(2, 100), xytext=(3, 150))
plt.annotate("Corr = "+str(corr2), xy=(3, 200), xytext=(5, 250))
plt.annotate("Corr = "+str(corr3), xy=(4, 300), xytext=(8, 400))
plt.xlabel('Number')
plt.ylabel('range')
# ax.annotate("Correlation="+str(corr),(z_mean,y_mean))
plt.show()

# a = [1,1.5,20]
# # a = [1,2,3]
# b = [1,2,3]
# print pearsonr(a,b)[0]
# import numpy
# print numpy.corrcoef(a,b)[0][0]

# for i, txt in enumerate(n):
#     ax.annotate("haha", (z[i],y[i]))

# plt.show()