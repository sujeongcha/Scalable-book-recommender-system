#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''training
Usage:
    $ module load python/gnu/3.6.5
    $ module load spark/2.4.0
    $ spark-submit nmslib_rec.py hdfs:/user/hj1399/test.parquet hdfs:/user/hj1399/ds_best_pipeline True 500
    $ spark-submit nmslib_rec.py hdfs:/user/hj1399/test.parquet hdfs:/user/hj1399/ds_best_pipeline False 500
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
import nmslib
import numpy as np

def augment_inner_product_matrix(factors):
    """
    This involves transforming each row by adding one extra dimension as suggested in the paper:
    "Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product
    Spaces" https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
    
    # Code adopted from 'implicit' repo
    # https://github.com/benfred/implicit/blob/4dba6dd90c4a470cb25ede34a930c56558ef10b2/implicit/approximate_als.py#L37
    """
    norms = np.linalg.norm(factors, axis=1)
    max_norm = norms.max()

    extra_dimension = np.sqrt(max_norm ** 2 - norms ** 2)
    return np.append(factors, extra_dimension.reshape(norms.shape[0], 1), axis=1)

def nmslib_recommend(df, model, k=500):
    # user_factors only for the users in the given df (ordered by user id)
    subset_user = df.select('user').distinct()
    whole_user = model.userFactors
    user = whole_user.join(subset_user, whole_user.id == subset_user.user).orderBy('id')
    user_factors = user.select('features')
    
    # item_factors ordered by item id
    item = model.itemFactors.orderBy('id')
    item_factors = item.select('features')
    
    # save user/item label
    user_label = [user[0] for user in user.select('id').collect()]
    item_label = [item[0] for item in item.select('id').collect()]
    print("item_label length", len(item_label))
    
    # to numpy array
    user_factors = np.array(user_factors.collect()).reshape(-1, model.rank) 
    item_factors = np.array(item_factors.collect()).reshape(-1, model.rank)
    print("feature array created")

    # Euclidean Transformation for Inner-Product Spaces
    extra = augment_inner_product_matrix(item_factors)
    print("augmented")
    
    # create nmslib index
    recommend_index = nmslib.init(method='hnsw', space='cosinesimil')
    recommend_index.addDataPointBatch(extra)
    recommend_index.createIndex({'post': 2})
    print("index created")
    
    # recommend for given users
    query = np.append(user_factors, np.zeros((user_factors.shape[0],1)), axis=1)
    results = recommend_index.knnQueryBatch(query, 5)
    print("results: ", results)
    print("results type: ", type(results))
    
    recommend = []
    for user_ in range(len(results)):
        itemlist = []
        for item_ in results[user_][0]:
            itemlist.append(item_label[item_])
        recommend.append((user_label[user_], itemlist))
        
    return recommend #[(user_id_1, [book_1, book_2,..., book_n]), (...)]

def main(spark, df_path, trained_model_path, approx, k):
    """
    This function evaluates the performance of a given model on a given dataset using Ranking Metrics,
    and returns the final performance metrics.
    Parameters
    ----------
    df: DataFrame to evaluate on
    trained_model_path: path to trained model to evaluate
    approx: boolean; use ANN(approximate nearest neighbors) when True
    k: number of recommendation 
    ----------
    """
    trained_model = PipelineModel.load(trained_model_path)
    trained_model = trained_model.stages[-1]

    df = spark.read.parquet(df_path)

    # change column names
    df = df.select(['user_id', 'book_id', 'rating']).toDF('user', 'item', 'rating')
    
    # relevant item if its centered rating > 0
    fn = F.udf(lambda x: 1.0 if x >= 3 else 0.0)
    df = df.withColumn('rating', fn(df.rating))
    relevant = df[df.rating == 1.0].groupBy('user').agg(F.collect_list('item'))
    
    # recommend k items for each user
    print("recommendation time comparison start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if approx:
        recommend = nmslib_recommend(df, trained_model, k)
        recommend = spark.createDataFrame(recommend, ["user", "recommend"])
        joined = recommend.join(relevant, on='user')
        rec_and_rel = []
        for user, rec, rel in joined.collect():
            rec_and_rel.append((rec, rel))
    else:     
        userSubset = relevant.select('user')
        recommend = trained_model.recommendForUserSubset(userSubset, 500)
        joined = recommend.join(relevant, on='user')
        rec_and_rel = []
        for user, rec, rel in joined.collect():
            predict_items = [i.item for i in rec]
            rec_and_rel.append((predict_items, rel))
    print("recommendation time comparison end: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Compute metrics
    rec_and_rel_rdd = spark.sparkContext.parallelize(rec_and_rel)
    metric_class = RankingMetrics(rec_and_rel_rdd)

    ndcg = metric_class.ndcgAt(k)
    map_ = metric_class.meanAveragePrecision
    pk = metric_class.precisionAt(k)

    return print("NDCG:", ndcg, "\nMAP:", map_, "\nPrecision:", pk)

# Only enter this block if we're in main
if __name__ == "__main__":
    memory = "10g"

    # Create the spark session object
    spark = SparkSession.builder.appName('nmslib_rec')\
        .config("spark.sql.broadcastTimeout", "36000")\
        .config('spark.executor.memory', memory)\
        .config('spark.driver.memory', memory)\
        .master('yarn')\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    print("Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Get the filename from the command line
    df_path = sys.argv[1] # test df
    trained_model_path = sys.argv[2] # pipeline path
    approx = sys.argv[3] # if true => nmslib, else => brute force
    k = sys.argv[4] # top k recommendations for each user

    # Call our main routine
    main(spark, df_path, trained_model_path, approx, k)
    print("end of everything: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    spark.stop()
