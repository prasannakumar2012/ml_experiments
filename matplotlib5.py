# import matplotlib.pyplot as plt
#
# y=[2.56422, 3.77284,3.52623,3.51468,3.02199]
# z=[0.15, 0.3, 0.45, 0.6, 0.75]
# n=[58,651,393,203,123]
#
# fig, ax = plt.subplots()
# ax.scatter(z, y)
#
# for i, txt in enumerate(n):
#     ax.annotate("haha", (z[i],y[i]))
#
# plt.show()



import matplotlib.pyplot as plt
import numpy as np


# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.ylim(-2,2)
fig = plt.gcf()
plt.show()
