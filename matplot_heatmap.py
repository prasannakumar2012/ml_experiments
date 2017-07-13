# """
# Annotated heatmaps
# ==================
#
# """
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
ax = plt.axes()
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5)
plt.show()

# sns.show()


# import numpy as np
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# data = np.random.randn(10,12)
# print data
# ax = plt.axes()
# sns.heatmap(data, ax = ax,annot=True,fmt="g")
#
# ax.set_title('Sample Title')
# plt.show()


# import numpy as np; np.random.seed(0)
# import seaborn as sns; sns.set()
# uniform_data = np.random.rand(10, 12)
# ax = sns.heatmap(uniform_data)


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import seaborn.matrix as smatrix
#
# sns.set()
#
# flights_long = sns.load_dataset("flights")
# flights = flights_long.pivot("month", "year", "passengers")
# flights = flights.reindex(flights_long.iloc[:12].month)
#
# columns = [1953,1955]
# myflights = flights.copy()
# mask = myflights.columns.isin(columns)
# myflights.loc[:, ~mask] = 0
# arr = flights.values
# vmin, vmax = arr.min(), arr.max()
# sns.heatmap(myflights, annot=True, fmt="d", vmin=vmin, vmax=vmax)
# plt.show()


# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
#
# data = np.random.randint(-1, 2, (10,10)) # Random [-1, 0, 1] data
# sns.heatmap(data, cmap=ListedColormap(['green', 'yellow', 'red']), annot=True)
# plt.show()


# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.set()
# fig, ax = plt.subplots(1,2)
# data = np.array([[0.000000,0.000000],[-0.231049,0.000000],[-0.231049,0.000000]])
# sns.heatmap(data, vmin=-0.231049, vmax=0, annot=True, fmt='f', annot_kws={"size": 15}, ax=ax[0])
# sns.heatmap(data, vmin=-0.231049, vmax=0, annot=True, fmt='f', annot_kws={"size": 10}, ax=ax[1]);
# plt.show()