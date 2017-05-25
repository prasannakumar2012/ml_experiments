"""
Name: Prasanna Kumar
Team Name : Data_hack
Mobile Number : 9871965377
Email : prasannakumar2012@gmail.com
"""

from bs4 import BeautifulSoup

directory="/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/"
from collections import namedtuple
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
count=0
docs=[]
import os
filepath_arr=[]
#Getting html tags and using it as words of the sentences. Using Doc2Vec to learn vectors for the sentences.
for root, directories, files in os.walk(directory):
    for filename in files:
        count+=1
        if count <50:  #Just to run on sample data and test the results
            filepath = os.path.join(root, filename)
            filepath_arr.append(filepath)
            # print filepath
            cases_text = open(filepath)
            soup = BeautifulSoup(cases_text, 'html.parser')
            words=[]
            for tag in soup.findAll():
                words.append(tag.name)
            tags=[count-1]
            docs.append(analyzedDocument(words, tags))
        else:
            break


from gensim.models import doc2vec
model = doc2vec.Doc2Vec(docs, size = 100, window = 300, min_count = 1, workers = 4)
from sklearn.cluster import KMeans

X=[]
#Array of sentence vector is used for clustering
for item in model.docvecs:
    X.append(item.tolist())


# print km.cluster_centers_

# km.predict(model.docvecs[0].tolist())
# for item in model.docvecs:
#     km.predict(item.tolist())

x_arr=[]
y_arr=[]
for i in range(2,20):
    km_temp = KMeans(n_clusters=i, init='k-means++', max_iter=100)
    km_temp.fit(X)
    x_arr.append(i)
    y_arr.append(km_temp.inertia_)

import matplotlib.pyplot as plt
#To find optimal number of clusters by seeing the graph. Elbow Method
plt.scatter(x_arr, y_arr)
plt.show()

#Clustering based on number of optimal clusters
optimal_cluster=4
km = KMeans(n_clusters=optimal_cluster, init='k-means++', max_iter=1000)
km.fit(X)


#filepath_arr - array of filepaths of the documents. km.labels_ - array of labels for the document. So all the documents are classified into
#one of the categories. Print file path and cluster number for that file. Giving names to each of the clusters.

for iter in range(len(filepath_arr)):
    print filepath_arr[iter],km.labels_[iter]


#Get features of each html page and build a classification model using random forrest. Print the tree formed. Trying to
#classify the clusters using tree formed and also manually looking at samples from each of the clusters
#count_of_different_types_of_tags_in_each_cluster - from collections import Counter
#Counter()
#Features for classification : count_of_tags,count_of_different_tags,count_of_images,count_of_url,table_tag_ratio(TTR),text_of_title



"""
Output


0 - One product page and details about the product
1 - Product catalog page from different website - different language
2 - Catalog page - Having many products
3 - One product page and details about the product(hiving lesser images), reviews page


/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00080d0404a8a81746f530ecdb9b442f 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/000f38b3a5e44b735bef1548f8523ba6 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/001d8525d90e1225a1f0186fe899b9ca 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00242bca4584d49ca66a0615e5ea1dc7 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00268b66f6eb8e4fcab132ba9fe7d203 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/002f4ffb333f18af2fd90d27848b26f0 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00349531709c8d8ff39940fbcd844c33 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0037c9087fd8eda3da077d565a480de6 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0039e501fdf7713a3d8f84e50d39d7aa 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0039e8e3d339e9d78e9124c10bffd7f3 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/005f975c3f836186b0d92fe1ad6359e7 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0064ffff1d8bad645fca8735d1694634 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00722dae7212cf93191d5c24212890d9 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00729864692568aec2528d074d36c526 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0080d8b0cd00c8a484af0d71a6df510b 0
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00818d71bde16b831305e72f115f04f8 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0088e0d42c1eb94133bdccf389851373 0
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/008ac104c0cfe5e8ae55b98b007caa48 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00966461ed1b95b7dcdea92246cd9322 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00a5bbf4e31068e9dea9bbc3bd7cb639 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00a767dbf32fb2e42d425044c32fa006 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00ae87b1d892b11c1367196393b2c8b9 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00bd0b0b0fee4b5ec31c570cc6f4555a 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00c3a64fc5186a96ca6fa70e7ffb16bc 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00c505390356017990316c8ba29ee312 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00efe1f33bc0de119bb4070aaa96c3b5 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00f52feb5aa77cfebfbc0e72ff443595 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00f65cb89194a5b24f4686a5babb4ad3 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00f916336c8da0e0a92657569e20e0b9 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00fb0f33153f75864270ea7bdec070f6 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/00fc1df11fa83f69a0e994a3570707bd 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/01000b43c985bc0f678794966ac5e5f6 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0102d0904947d07ffae0a592332f3e90 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/01059867ac70011d65660d5ccf44e8d2 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/010a0e6969f1e769c78aeb4bd82c0eb0 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0110d449dbfcc98cd260117c9acfc969 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0113323693071c25f8dede3883e60b8a 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/012a333c371fb160ce4de6efa4beb291 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/012b2c816223bdd296cb32dfc6873eb7 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0132f812546719923ad2e7ee850cbff3 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/013381eded1248c5d869399842a3168e 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/01372337a754bf70a5dca6932faae7ae 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/015d442f4b1adbbaaf156f96431fec69 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/015ea15b3a251810ea50659140b35d27 1
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/015fa19386011b11b1e5b242a166c5ca 2
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/0160b90e96479622b48c6c296dc4abeb 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/01682132fce2842678573360226d4f71 3
/Users/prasanna/Desktop/Datasets/HTMLPage-Classifier-Dataset/html_sample_dataset/sample_data/016a0e606823776989d5e2e915a54638 2


"""