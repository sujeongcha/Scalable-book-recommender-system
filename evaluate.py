#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''training
Usage:
    $ module load python/gnu/3.6.5
    $ module load spark/2.4.0
    $ spark-submit evaluate.py hdfs:/user/hj1399/test.parquet hdfs:/user/hj1399/ds_best_pipeline
'''

import sys
from pyspark.sql import SparkSession

#spark.ml packages
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, expr
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
import datetime

def main(spark, val_pq, model_file_path):
    '''
    Args
    -------
    val_pq:
        validation data
    
    model_file_path:
        path to the pipeline(stringIndexers + als) model
    '''

    # Read data
    val = spark.read.parquet(val_pq)

    print('load trained model')
    # Load the trained pipeline model
    model = PipelineModel.load(model_file_path)

    # evaluation

    print("Run prediction")
    # Run the model to create prediction against a validation set
    preds = model.transform(val)
    
    print("Run evaluation")

    # model evaluation using rmse on val data
    print("Start evaluation using rmse")
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(preds)

    # Generate top 500 book recommendations for each user in validation data.
    # Returns a DataFrame of (userCol, recommendations), 
    # where recommendations are stored as an array of (itemCol, rating) Rows.
    #user_id = preds.select("user_id_idx").distinct()
    #res = model.stages[-1].recommendForUserSubset(user_id, 500)

    print("generate top 500 book recommendations for val users")
    res = model.stages[-1].recommendForAllUsers(500)
    preds_per_user = res.selectExpr("user_id_idx", "recommendations.book_id_idx as preds_books")
    # preds_pe_user.show(5)

    true_per_user = preds.select("user_id_idx","book_id_idx").filter("rating>=3")\
                        .groupBy("user_id_idx")\
                        .agg(expr("collect_set(book_id_idx) as books"))
    # true_per_user.show(5)

    
    print("Start join")
    # true_per_user: an RDD of (predicted ranking, ground
    # truth set) pairs
    # true_vs_preds_per_user = preds_per_user.join(true_per_user, ["userId"]).rdd\
    #                 .map(lambda row: (row.items_pred, row.items)).cache()

    true_vs_preds_per_user = preds_per_user.join(true_per_user, ["user_id_idx"])\
                    .select("preds_books","books").rdd

    # print(*true_vs_preds_per_user.take(5),sep="\n")

    # Evaluate using RMSE
    #evaluator = RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="??")
    #rmse = evaluator.evaluate(preds)
    #print(f'The out-of-sample RMSE of the current model is: {rmse:.2f}')

    # Evaluate using MAP
    print("Start evaluation using MAP")
    metrics = RankingMetrics(true_vs_preds_per_user)
    map_ = metrics.meanAveragePrecision

    #Evaluate using ndcg
    print("Start evaluation using ndcg")
    ndcg = metrics.ndcgAt(500)
    
    #Evaluate using precision
    mpa = metrics.precisionAt(500)

    print('rmse score: ', rmse, 'map score: ', map_, 'ndcg score: ', ndcg, 'mpa score: ', mpa)

 # Only enter this block if we're in main

if __name__ == "__main__":
    memory = "10g"

    # Create the spark session object
    spark = SparkSession.builder.appName('evaluate')\
        .config("spark.sql.broadcastTimeout", "36000")\
        .config('spark.executor.memory', memory)\
        .config('spark.driver.memory', memory)\
        .master('yarn')\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # validation data file
    val_pq = sys.argv[1]

    # location of the trained model
    model_file_path = sys.argv[2]

    # Call our main routine
    main(spark, val_pq, model_file_path)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    spark.stop()