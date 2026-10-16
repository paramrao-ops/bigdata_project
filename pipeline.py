import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("TelcoChurnPipeline").getOrCreate()

# 1. Bronze Layer: Read Raw CSV
df_raw = spark.read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", header=True, inferSchema=True)
df_raw.write.mode("overwrite").parquet("data/bronze/churn_raw.parquet")

# 2. Silver Layer: Clean Data & Handle Empty Strings
silver_df = spark.read.parquet("data/bronze/churn_raw.parquet")

# Replace empty spaces in TotalCharges with 0.0 and cast types safely
silver_df = silver_df.withColumn(
    "TotalCharges", 
    F.when(F.trim(F.col("TotalCharges")) == "", 0.0)
     .otherwise(F.col("TotalCharges").cast(DoubleType()))
)
silver_df = silver_df.withColumn("tenure", F.col("tenure").cast(IntegerType()))
silver_df = silver_df.withColumn("MonthlyCharges", F.col("MonthlyCharges").cast(DoubleType()))
silver_df = silver_df.withColumn("ChurnBinary", F.when(F.col("Churn") == "Yes", 1.0).otherwise(0.0))

silver_df.write.mode("overwrite").parquet("data/silver/churn_cleaned.parquet")

# 3. Gold Layer: Feature Engineering & ML Training
gold_df = spark.read.parquet("data/silver/churn_cleaned.parquet")

# Index Categorical Columns
indexer_contract = StringIndexer(inputCol="Contract", outputCol="ContractIndex")
gold_df = indexer_contract.fit(gold_df).transform(gold_df)

indexer_internet = StringIndexer(inputCol="InternetService", outputCol="InternetServiceIndex")
gold_df = indexer_internet.fit(gold_df).transform(gold_df)

assembler = VectorAssembler(
    inputCols=["tenure", "MonthlyCharges", "TotalCharges", "ContractIndex", "InternetServiceIndex"],
    outputCol="features"
)
gold_data = assembler.transform(gold_df)
gold_data.write.mode("overwrite").parquet("data/gold/churn_features.parquet")

# MLlib Training
train_data, test_data = gold_data.randomSplit([0.8, 0.2], seed=42)
lr = LogisticRegression(featuresCol="features", labelCol="ChurnBinary")
model = lr.fit(train_data)

predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="ChurnBinary", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)

print(f"\n==========================================")
print(f"SUCCESS! Model Accuracy (AUC Score): {auc:.4f}")
print(f"==========================================\n")
