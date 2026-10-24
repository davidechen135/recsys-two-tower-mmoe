"""PySpark Feature Engineering Job for E-commerce Recommendation Pipeline.

Matches the SQL window function logic in sql/feature_engineering.sql.
Requires a Spark cluster or local Java Runtime (OpenJDK 11+).
"""
from __future__ import annotations

import sys


def run_spark_feature_engineering(input_path: str, output_path: str):
    try:
        from pyspark.sql import SparkSession, Window
        import pyspark.sql.functions as F
    except ImportError:
        print("PySpark is not installed or Java is missing. Please ensure Java 11+ is configured.")
        sys.exit(1)

    spark = (
        SparkSession.builder.appName("RecSys-FeatureEngineering")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    schema = "user_id LONG, item_id LONG, category_id INT, behavior_type STRING, timestamp LONG"
    df = spark.read.csv(input_path, schema=schema, sep=",")

    # 1. Item Level Statistics
    item_stats = df.groupBy("item_id").agg(
        F.first("category_id").alias("category_id"),
        F.count("*").alias("total_interactions"),
        F.sum(F.when(F.col("behavior_type") == "pv", 1).otherwise(0)).alias("item_pv_count"),
        F.sum(F.when(F.col("behavior_type") == "cart", 1).otherwise(0)).alias("item_cart_count"),
        F.sum(F.when(F.col("behavior_type") == "fav", 1).otherwise(0)).alias("item_fav_count"),
        F.sum(F.when(F.col("behavior_type") == "buy", 1).otherwise(0)).alias("item_buy_count"),
    )

    item_features = item_stats.withColumn(
        "item_pv_to_cart_rate",
        (F.col("item_cart_count") + 1.0) / (F.col("item_pv_count") + 20.0),
    ).withColumn(
        "item_pv_to_buy_rate",
        (F.col("item_buy_count") + 1.0) / (F.col("item_pv_count") + 50.0),
    )

    # 2. User Window Sequences
    user_time_window = Window.partitionBy("user_id").orderBy(F.col("timestamp").desc())
    user_cum_window = Window.partitionBy("user_id").orderBy(F.col("timestamp").asc())

    ranked_events = (
        df.withColumn("seq_rank_desc", F.row_number().over(user_time_window))
        .withColumn(
            "time_since_prev",
            F.col("timestamp") - F.lag("timestamp", 1).over(user_cum_window),
        )
        .withColumn("cum_actions", F.count("*").over(user_cum_window))
    )

    # Extract recent 20 interaction item IDs per user
    recent_seq = (
        ranked_events.filter(F.col("seq_rank_desc") <= 20)
        .groupBy("user_id")
        .agg(
            F.collect_list("item_id").alias("recent_item_seq"),
            F.collect_list("category_id").alias("recent_cat_seq"),
        )
    )

    # 3. User Aggregates
    user_stats = df.groupBy("user_id").agg(
        F.count("*").alias("user_total_actions"),
        F.sum(F.when(F.col("behavior_type") == "pv", 1).otherwise(0)).alias("user_pv_count"),
        F.sum(F.when(F.col("behavior_type") == "buy", 1).otherwise(0)).alias("user_buy_count"),
        F.countDistinct("category_id").alias("user_distinct_categories"),
        F.countDistinct("item_id").alias("user_distinct_items"),
    )

    user_features = user_stats.join(recent_seq, on="user_id", how="left")

    print(f"Saving features to {output_path}...")
    item_features.write.mode("overwrite").parquet(f"{output_path}/item_features.parquet")
    user_features.write.mode("overwrite").parquet(f"{output_path}/user_features.parquet")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python feature_engineering_job.py <input_csv> <output_dir>")
        sys.exit(1)
    run_spark_feature_engineering(sys.argv[1], sys.argv[2])
