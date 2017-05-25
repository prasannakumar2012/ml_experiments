"""

pip install --user git+https://github.com/jpmml/sklearn2pmml.git
"""

from sklearn import datasets, tree
iris = datasets.load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

from sklearn_pandas import DataFrameMapper
default_mapper = DataFrameMapper([(i, None) for i in iris.feature_names + ['Species']])

from sklearn2pmml import sklearn2pmml

sklearn2pmml(estimator=clf,
             mapper=default_mapper,
             pmml="IrisClassificationTree.pmml")

