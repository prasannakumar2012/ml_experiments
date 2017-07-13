import matplotlib.pyplot as plt
data = [[1.0,1.0],[1.1,1.1],[5.0,5.0],[5.1,5.1],[1.8,2.3],[2.5,2.1],[7.1,6.9],[6,7]]
radius = []
area = []

for i in range(len(data)):
    radius.append(data[i][0])
    area.append(data[i][1])
# radius = [1.0, 1.1, 1.3,  4.9, 5.0, 5.1]
# area = [1.0, 1.1, 1.3 , 4.9, 5.0, 5.1]
# area = [3.14159, 12.56636, 28.27431, 50.26544, 78.53975, 113.09724]
# plot(x, y, color='green', linestyle='dashed', marker='o')
# plt.plot(radius, area, 'r--')
# plt.plot(radius, area, color='red', linestyle='dashed', marker='o')

from sklearn.cluster import KMeans
k_means = KMeans(init='k-means++', n_clusters=2)
k_means.fit(data)
print k_means.cluster_centers_

# plt.plot(k_means.cluster_centers_[0], k_means.cluster_centers_[1], 'ro', color = 'red')
# plt.plot(radius, area, 'ro', color = 'red')
plt.plot(radius, area, 'ro' ,color = 'green')

plt.plot(k_means.cluster_centers_[0][0], k_means.cluster_centers_[0][1], 'ro' , color = 'red')
plt.plot(k_means.cluster_centers_[1][0], k_means.cluster_centers_[1][1], 'ro' , color = 'red' )
# plt.plot(3,3,'ro', color = 'green')
# plt.plot(radius, area, 'ro')

plt.axis([0, 8, 0, 8])
plt.show()