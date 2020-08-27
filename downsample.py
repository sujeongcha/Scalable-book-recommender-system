#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''A simple pyspark script to downsample data in a parquet-backed dataframe
   Return train, val, test dataframe  
Usage:
    $ spark-submit downsample.py hdfs:/user/hj1399/interaction.parquet
'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark import SparkContext as sc
import math
import datetime

def split_data(df, fraction, seed=None):
    #df = (val or test)
    #fraction = 0.5
    #seed = 42
    fractions = {row['user_id']: fraction for row in df.select('user_id').distinct().collect()} # {user_id1: 0.5, user_id2: 0.5, ..., user_idn: 0.5 }
    data_sampled = df.sampleBy('user_id', fractions, seed=42)
    data_left = df.rdd.subtract(data_sampled.rdd)
    data_left_df = spark.createDataFrame(data_left, df.schema)

    # return data_sampled and data_left_df (sql dataframe)
    return data_sampled, data_left_df

def main(spark, interaction_path):
    '''
    ----------
    spark : SparkSession object
    interaction_path : string, path to the interaction parquet file to load
    user_map_path : string, path to the user map parquet file to load
    '''
    # Load the dataframe
    interaction = spark.read.parquet(interaction_path)
    #user_map = spark.read.parquet(user_map_path)

    # discard rating == 0
    interaction = interaction.filter(interaction['is_read'] == 1)
    is_read_filtered = interaction.select("user_id").distinct().count()
    print("number of distinct user ids after discarding is_read == 0: ", is_read_filtered)

    # Get dataframe with users having more than 30 interactions
    df = interaction.groupby('user_id').agg(countDistinct('book_id').alias('num_interaction'))
    df_more30 = df.where(df.num_interaction > 30)
    user_id_ls = df_more30.select('user_id').distinct()
    print("number of distinct user ids after discarding interactions < 30: ", user_id_ls.count())

    # subsample
    print("start subsampling")
    fraction = 0.01
    seed = 42
    downsample = user_id_ls\
        .sample(False, fraction=fraction, seed=seed)\
        .join(interaction, 'user_id', 'inner')

    # save downsample for possible later use
    downsample.write.format("parquet").mode("overwrite").save("hdfs:/user/hj1399/downsample.parquet")

    # split 60%, 20%, 20% of downsample based on distinct user id
    train_user, val_user, test_user = downsample\
        .select("user_id")\
        .distinct()\
        .randomSplit(weights=[0.6, 0.2, 0.2], seed=seed)

    #join splitted train_user, val_user, test_user with downsample to create training dataset
    # these are df now (!= rdd)
    train = train_user.join(downsample, 'user_id', 'inner')
    val = val_user.join(downsample, 'user_id', 'inner')
    test = test_user.join(downsample, 'user_id', 'inner')

    # take 50 % of interactions per user from val and test data and add those to train
    print("start taking half inteactions from val and test data and add them to train")
    val_sampled, val_left = split_data(val, fraction=0.5, seed=42)
    test_sampled, test_left = split_data(test, fraction=0.5, seed=42)

    train_df = train.union(val_sampled).union(test_sampled)
    
    print("number of distinct user ids of train data: ", train_df.select("user_id").distinct().count())
    print("number of distinct user ids of val data: ", val_left.select("user_id").distinct().count())
    print("number of distinct user ids of test data: ", test_left.select("user_id").distinct().count())

    train_df.write.format("parquet").mode("overwrite").save("hdfs:/user/hj1399/train.parquet")
    val_left.write.format("parquet").mode("overwrite").save("hdfs:/user/hj1399/val.parquet")
    test_left.write.format("parquet").mode("overwrite").save("hdfs:/user/hj1399/test.parquet")

# Only enter this block if we're in main
if __name__ == "__main__":
    memory = "10g"

    # Create the spark session object
    spark = SparkSession.builder.appName('downsample')\
        .config("spark.sql.broadcastTimeout", "36000")\
        .config('spark.executor.memory', memory)\
        .config('spark.driver.memory', memory)\
        .master('yarn')\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Get the filename from the command line
    interaction_path = sys.argv[1]

    # Call our main routine
    main(spark, interaction_path)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    spark.stop()
    

