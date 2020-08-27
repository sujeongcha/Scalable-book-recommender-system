#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''training
Usage:
    $ module load python/gnu/3.6.5
    $ module load spark/2.4.0
    $ spark-submit train.py hdfs:/user/hj1399/train.parquet
'''

import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
import datetime

def main(spark, train_pq):
    '''
    Args
    ---------
    train_pq(parquet):
    training data 
    '''
    #Read train data
    train = spark.read.parquet(train_pq)
    train.repartition(200)
    
    # pipeline
    # StringIndexers
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_idx", handleInvalid = 'skip')
    indexer_book = StringIndexer(inputCol="book_id", outputCol="book_id_idx", handleInvalid='skip')

    # ALS model
    #als = ALS(maxIter=10, rank= 10, regParam=0.05, alpha = 1.0,
              #userCol="user_id_idx", itemCol="book_id_idx", ratingCol="rating",
              #coldStartStrategy="drop").setSeed(42)

    #doing baseline          
    als = ALS(maxIter=10, rank = 20, regParam=0.05,\
              userCol="user_id_idx", itemCol="book_id_idx", ratingCol="rating",\
              coldStartStrategy="drop").setSeed(42)

    # Combine into the pipeline
    pipeline = Pipeline(stages=[indexer_user,indexer_book, als])

    # Training
    print("start training")
    model = pipeline.fit(train)

    # Save the trained model
    model.write().overwrite().save("hdfs:/user/hj1399/ds_best_pipeline")
    print("Trained model saved")


# Only enter this block if we're in main
if __name__ == "__main__":
    memory = "10g"

    # Create the spark session object
    spark = SparkSession.builder.appName('train')\
        .config("spark.sql.broadcastTimeout", "36000")\
        .config('spark.executor.memory', memory)\
        .config('spark.driver.memory', memory)\
        .master('yarn')\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Get the filename from the command line
    train_pq = sys.argv[1]

    # Call our main routine
    main(spark, train_pq)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    spark.stop()