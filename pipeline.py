import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("CustomerChurnAnalytics") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Create Medallion Folder Structure
os.makedirs("data/bronze", exist_ok=True)
os.makedirs("data/silver", exist_ok=True)
os.makedirs("data/gold", exist_ok=True)

# Step 3: Bronze Layer
print("--- Step 3: Writing Bronze Layer ---")
raw_df = spark.read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", header=True, inferSchema=True)
raw_df.write.mode("overwrite").parquet("data/bronze/churn_raw.parquet")

# Step 4 & 5: Silver Layer Cleaning & Transformations
print("--- Step 4 & 5: Cleaning Data & Silver Layer ---")
bronze_df = spark.read.parquet("data/bronze/churn_raw.parquet")

silver_df = bronze_df.withColumn("TotalCharges", col("TotalCharges").cast("double")) \
                     .na.fill(0.0, subset=["TotalCharges"]) \
                     .withColumn("ChurnBinary", when(col("Churn") == "Yes", 1).otherwise(0))

silver_df.write.mode("overwrite").parquet("data/silver/churn_cleaned.parquet")

# Step 6: Gold Layer (Feature Engineering)
print("--- Step 6: Engineering Features & Gold Layer ---")
categorical_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
                    "MultipleLines", "InternetService", "Contract", "PaymentMethod"]
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

indexed_df = silver_df
for c in categorical_cols:
    indexer = StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="skip")
    indexed_df = indexer.fit(indexed_df).transform(indexed_df)

indexed_cols = [f"{c}_indexed" for c in categorical_cols]
assembler = VectorAssembler(inputCols=indexed_cols + numeric_cols, outputCol="features")
gold_ml_df = assembler.transform(indexed_df).select("features", "ChurnBinary")

gold_ml_df.write.mode("overwrite").parquet("data/gold/churn_features.parquet")

# Step 7 & 8: Machine Learning & Prediction Generation
print("--- Step 7 & 8: Training Machine Learning Model ---")
train_data, test_data = gold_ml_df.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(labelCol="ChurnBinary", featuresCol="features")
model = lr.fit(train_data)

predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="ChurnBinary")
auc = evaluator.evaluate(predictions)

print(f"\n SUCCESS! Model Accuracy (AUC Score): {auc:.4f}")
print("All data layers (Bronze, Silver, Gold) and predictions saved successfully!")
