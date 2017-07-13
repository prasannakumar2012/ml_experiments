
import os, commands, sys, json, uuid
from pyspark import SparkContext,SparkConf
from flask import Flask, request, jsonify
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, VectorAssembler, Binarizer
import pyspark.sql.functions as func
from pyspark.sql.functions import col, count, sum, min, avg, when, max
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.feature import PCA
from pyspark.ml.feature import PolynomialExpansion
import os, commands, sys, json, uuid
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, VectorAssembler, Binarizer
import pyspark.sql.functions as func
from pyspark.sql.functions import col, count, sum, min, avg, when
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor, LinearRegression, GeneralizedLinearRegression
from pyspark.ml.feature import PCA
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.evaluation import RegressionEvaluator



from pyspark.sql.functions import col, count, sum, min, avg, when, max
# Count number of not null values in each column
training_data.agg(*[count(c).alias(c) for c in training_data.columns]).show()
training_data.agg(*[count(c).alias(c) for c in training_data.columns]).collect()
query_label = "select *, cast(" + label + " as double ) as label from df_temp "


print training_data.groupBy('label').agg(func.count("label")).show()
input_col = ['a','b','c']
assembler = VectorAssembler(inputCols=input_col, outputCol="features")
training_data_ass = assembler.transform(training_data)

# Feature extraction and selection
#pca
pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(training_data_ass)
print pca_model.explainedVariance
training_data_ass_pca = pca_model.transform(training_data_ass)

# PolynomialExpansion
px = PolynomialExpansion(degree=2, inputCol="features", outputCol="polyFeatures")
training_data_ass_pca_ply = px.transform(training_data_ass)


sindexer = StringIndexer(inputCol="label", outputCol="label_ind")
indexer = sindexer.fit(training_data_ass)
indexed = indexer.transform(training_data_ass)
indexed.cache()
(trainingData, testData) = indexed.randomSplit([0.75, 0.25])


assembler = VectorAssembler(inputCols=input_col, outputCol="features")
training_data_ass = assembler.transform(df)
sindexer = StringIndexer(inputCol="label", outputCol="label_ind")
indexer = sindexer.fit(training_data_ass)
indexed = indexer.transform(training_data_ass)
indexed.cache()
(trainingData, testData) = indexed.randomSplit([0.75, 0.25])


def kfold_split(df, n_fold):
    return df.randomSplit([1 / float(n_fold)] * n_fold)


training_data_ass_arr = kfold_split(training_data_ass, 5)

dt = DecisionTreeRegressor(featuresCol="features", labelCol="label")
rf = RandomForestRegressor(featuresCol="features", labelCol="label")
gbt = GBTRegressor(featuresCol="features", labelCol="label")
lr = LinearRegression(featuresCol="features", labelCol="label")
glr = GeneralizedLinearRegression(family="gaussian", link="identity")

def calculate_accuracy_reg_kfolds(model, training_data_ass_arr):
    n_folds = len(training_data_ass_arr)
    for i in range(0, n_folds):
        j = (i + n_folds - 1) % n_folds
        testData = training_data_ass_arr[j]
        count = 0
        for k in range(0, n_folds):
            if k != j:
                if count == 0:
                    trainingData = training_data_ass_arr[k]
                    count += 1
                else:
                    trainingData = trainingData.unionAll(training_data_ass_arr[k])
                    count += 1
        model = model.fit(trainingData)
        predictions = model.transform(testData)
        total_count = predictions.count()
        rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(
            predictions)
        mse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse").evaluate(
            predictions)
        r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(
            predictions)
        mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(
            predictions)
        return total_count, rmse, mse, r2, mae


dt = DecisionTreeClassifier(labelCol="label_ind", featuresCol="features")
rf = RandomForestClassifier(labelCol="label_ind", featuresCol="features")
gbt = GBTClassifier(labelCol="label_ind", featuresCol="features")
lr = LogisticRegression(labelCol="label_ind", featuresCol="features")
layers = [3, 4, 3, 2]
per = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

def calculate_accuracy_class(model, trainingData, testData):
    model = model.fit(trainingData)
    predictions = model.transform(testData)
    one_prediction = predictions.take(1)[0]
    switch = one_prediction.label != one_prediction.label_ind
    predictions.select('label_ind', 'prediction').registerTempTable('temp_predictions')
    if switch:
        measure_count = sqlContext.sql(
            "select count(if(label_ind=1.0 and prediction==1.0,'',null)) as a0p0,count(if(label_ind=1.0 and prediction==0.0,'',null)) as a0p1, count(if(label_ind=0.0 and prediction==1.0,'',null)) as a1p0, count(if(label_ind=0.0 and prediction==0.0,'',null)) as a1p1   from temp_predictions").collect()[
            0]
    else:
        measure_count = sqlContext.sql(
            "select count(if(label_ind=0.0 and prediction==0.0,'',null)) as a0p0,count(if(label_ind=0.0 and prediction==1.0,'',null)) as a0p1, count(if(label_ind=1.0 and prediction==0.0,'',null)) as a1p0, count(if(label_ind=1.0 and prediction==1.0,'',null)) as a1p1   from temp_predictions").collect()[
            0]
    total_count = measure_count.a0p0 + measure_count.a0p1 + measure_count.a1p0 + measure_count.a1p1
    accuracy = (measure_count.a0p0 + measure_count.a1p1) / float(total_count)
    recall = tp = measure_count.a1p1 / float(measure_count.a1p0 + measure_count.a1p1)
    fp = measure_count.a0p1 / float(measure_count.a0p0 + measure_count.a0p1)
    tn = measure_count.a0p0 / float(measure_count.a0p0 + measure_count.a0p1)
    fn = measure_count.a1p0 / float(measure_count.a1p0 + measure_count.a1p1)
    precision = measure_count.a1p1 / float(measure_count.a0p1 + measure_count.a1p1)
    return total_count, accuracy, recall, tp, fp, tn, fn, precision

