#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''training
Usage:
    $ module load python/gnu/3.6.5
    $ module load spark/2.4.0
    $ spark-submit tuning.py hdfs:/user/hj1399/train.parquet hdfs:/user/hj1399/val.parquet
'''
import sys
from pyspark.sql import SparkSession
#spark.ml packages
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, expr
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
import datetime

def main(spark, train_pq, val_pq):
    '''
    Args
    -------
    val_pq:
        validation data
    
    model_file_path:
        path to the pipeline(stringIndexers + als) model
    ''' 
    import itertools

    # Read train and val data
    print("load train and validation data")
    train = spark.read.parquet(train_pq)
    val = spark.read.parquet(val_pq)

    # Increase partition size of train data to reduce task load
    #train.repartition(200)

    # Pipeline
    # StringIndexers
    print("build stringIndexer")
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_idx", handleInvalid = 'skip')
    indexer_book = StringIndexer(inputCol="book_id", outputCol="book_id_idx", handleInvalid='skip')

    # Hyper-parameter tuning
    rank_  = [10, 15, 20]
    regParam_ = [0.01, 0.05, 0.1, 0.3, 1]
    param_grid = itertools.product(rank_, regParam_)

    # ALS model hyperparameter tuning
    for i in param_grid:
        print('training for {} start'.format(i))

        als = ALS(maxIter=10, rank=i[0], regParam=i[1],\
                userCol="user_id_idx", itemCol="book_id_idx", ratingCol="rating",\
                coldStartStrategy="drop").setSeed(42)
        
        # Combine into the pipeline
        pipeline = Pipeline(stages=[indexer_user,indexer_book, als])

        model = pipeline.fit(train)
        print('training for {} complete'.format(i))

        # predition against validation data
        preds = model.transform(val)

        # model evaluation using rmse on val data
        print("Start evaluation using rmse for {}".format(i))
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(preds)

        # Make top 500 recommendations for users in validation test
        print('evaluation for {} start'.format(i))        

        res = model.stages[-1].recommendForAllUsers(500)
        
        preds_per_user = res.selectExpr("user_id_idx", "recommendations.book_id_idx as preds_books")
        true_per_user = preds.select("user_id_idx","book_id_idx").filter("rating>=3")\
                            .groupBy("user_id_idx")\
                            .agg(expr("collect_set(book_id_idx) as books"))
        
        print("Start join for {}".format(i))
        true_vs_preds_per_user = preds_per_user.join(true_per_user, ["user_id_idx"])\
                        .select("preds_books","books").rdd
        
        # Evaluate using MAP
        print("Start evaluation using MAP for {}".format(i))
        metrics = RankingMetrics(true_vs_preds_per_user)
        map_ = metrics.meanAveragePrecision

        #Evaluate using ndcg
        print("Start evaluation using ndcg for {}".format(i))
        ndcg = metrics.ndcgAt(500)
        
        #Evaluate using precision
        print("Start evaluation using precisionAtK for {}".format(i))
        mpa = metrics.precisionAt(500)

        print(i, 'rmse score: ', rmse, 'map score: ', map_, 'ndcg score: ', ndcg, 'mpa score: ', mpa)

# Only enter this block if we're in main
if __name__ == "__main__":
    memory = "10g"

    # Create the spark session object
    spark = SparkSession.builder.appName('tuning')\
        .config("spark.sql.broadcastTimeout", "36000")\
        .config('spark.executor.memory', memory)\
        .config('spark.driver.memory', memory)\
        .master('yarn')\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Get the filename from the command line
    train_pq = sys.argv[1]
    val_pq = sys.argv[2]

    # Call our main routine
    main(spark, train_pq, val_pq)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    spark.stop()