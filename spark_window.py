"""

http://xinhstechblog.blogspot.in/2016/04/spark-window-functions-for-dataframes.html
"""

"""
// Building the customer DataFrame. All examples are written in Scala with Spark 1.6.1, but the same can be done in Python or SQL.
val customers = sc.parallelize(List(("Alice", "2016-05-01", 50.00),
                                    ("Alice", "2016-05-03", 45.00),
                                    ("Alice", "2016-05-04", 55.00),
                                    ("Bob", "2016-05-01", 25.00),
                                    ("Bob", "2016-05-04", 29.00),
                                    ("Bob", "2016-05-06", 27.00))).toDF("name", "date", "amountSpent")

// Import the window functions.
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

// Create a window spec.
val wSpec1 = Window.partitionBy("name").orderBy("date").rowsBetween(-1, 1)

In this window spec, the data is partitioned by customer. Each customer’s data is ordered by date. And, the window frame is defined as starting from -1 (one row before the current row) and ending at 1 (one row after the current row), for a total of 3 rows in the sliding window.

// Calculate the moving average
customers.withColumn( "movingAvg",avg(customers("amountSpent")).over(wSpec1)).show()

This code adds a new column, “movingAvg”, by applying the avg function on the sliding window defined in the window spec:

val w = org.apache.spark.sql.expressions.Window.orderBy("date") //some spec

val leadDf = inputDSAAcolonly.withColumn("df1Rank", rank().over(w))



"""