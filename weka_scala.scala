// Databricks notebook source
 import weka.core.Instances;
 import weka.experiment.InstanceQuery;

// COMMAND ----------

import weka.classifiers.trees.j48
import weka.classifiers.trees.J48

// COMMAND ----------

val abc = new J48();
abc.setUnpruned(true);
// val options = new String[1];
// options[0] = "-U";  
// abc.setOptions(options);

// COMMAND ----------

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
val reader = new BufferedReader(new FileReader("/dbfs/FileStore/tables/iauus1fy1505668418548/iris.arff"));
val data = new Instances(reader);

// COMMAND ----------

import weka.classifiers.Evaluation;
import java.util.Random;
abc.buildClassifier(data);
// evaluate classifier and print some statistics
val eval = new Evaluation(data);
eval.evaluateModel(abc, data);
// System.out.println(eval.toSummaryString("\nResults\n======\n", false));

// COMMAND ----------


