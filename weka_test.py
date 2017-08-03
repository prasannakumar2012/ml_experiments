"""
weka starter
http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/

saving models in weka
http://machinelearningmastery.com/save-machine-learning-model-make-predictions-weka/

weka python guide
http://pythonhosted.org//python-weka-wrapper/api.html#serialization

weka python installation
http://pythonhosted.org//python-weka-wrapper/install.html
brew install pkg-config and brew install graphviz
pip install javabridge
pip install python-weka-wrapper

java
https://stackoverflow.com/questions/14209510/loding-a-weka-model-to-a-java-code
https://stackoverflow.com/questions/20017957/how-to-reuse-saved-classifier-created-from-explorerin-weka-in-eclipse-java
weka.jar
java weka.classifiers.trees.J48 -T /Users/prasanna/Desktop/weka-3-8-1/data/iris.arff -l /Users/prasanna/Desktop/weka-3-8-1/treej48.model -p 0



"""

import weka.core.serialization as serialization
from weka.classifiers import Classifier
objects = serialization.read_all("/Users/prasanna/Desktop/weka-3-8-1/treej48.model")
classifier = Classifier(jobject=objects[0])

from weka.core.dataset import Instances
data2 = Instances(jobject=objects[1])
print data2

import weka.core.converters as converters
data = converters.load_any_file("/Users/prasanna/Desktop/weka-3-8-1/data/iris.arff")
data.class_is_last()

for index, inst in enumerate(data):
    pred = classifier.classify_instance(inst)
    dist = classifier.distribution_for_instance(inst)
    print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))