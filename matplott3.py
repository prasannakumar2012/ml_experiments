# import matplotlib.pyplot as plt
# from numpy.random import normal
# gaussian_numbers = [1,2,3,4,5,6,7,8,9,10,2,2,2,2,8,8,8,8,8,8,8,8]
# # gaussian_numbers = normal(size=1000)
# # print gaussian_numbers
# # print "0"
# # print gaussian_numbers[0]
# # print "1"
# # print gaussian_numbers[1]
# # print type(gaussian_numbers)
# plt.hist(gaussian_numbers)
# plt.title("Gaussian Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

#
# import numpy as np
# import pandas as pd
#
# d = {'one' : np.random.rand(10),
#      'two' : np.random.rand(10)}
#
# df = pd.DataFrame(d)
#
# print df.plot(style=['o','rx'])


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = {'one' : np.random.rand(10),
     'two' : np.random.rand(10)}

df = pd.DataFrame(d)
plt.scatter(df['one'], df['two'])
plt.show()

#df.toPandas().save('mycsv.csv')