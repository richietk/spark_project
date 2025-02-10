#!/usr/bin/env python3
"""
Scalable Election and Spending Analysis with Spark

This script replaces Pandas operations with Spark so that the entire workflow is scalable.
It:
  - Loads and flattens a JSON file with election results.
  - Reads and pivots a CSV of spending data (with subcategories).
  - Merges the two datasets on municipality (using a slug) and year.
  - Builds a Spark ML pipeline to predict the winning party using a random forest.
  - Trains one-vs-rest binary classifiers for selected parties.
  - Computes FPÖ win distribution by turnout level using Spark.
"""

import json
import re

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, when, regexp_replace, udf
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

################################################################################
# 1) HELPER FUNCTIONS
################################################################################

def clean_subcat_name(name: str) -> str:
    """
    Remove special characters from a subcategory name and add a prefix.
    e.g. 'Allgemeinbildender Unterricht' -> 'Sp_Allgemeinbildender_Unterricht'
    """
    c = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    return f"Sp_{c}"

################################################################################
# 2) INITIALIZE SPARK SESSION
################################################################################

spark = SparkSession.builder \
    .appName("ScalableElectionAnalysis") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

################################################################################
# 3) LOAD & FLATTEN ELECTION RESULTS
################################################################################

# Load the merged dataset from the first script instead of election_results.json
merged_data_path = "data/merged_data.csv"
merged_df = (
    spark.read
         .option("header", True)
         .option("inferSchema", True)
         .option("sep", ",")
         .csv(merged_data_path)
         .drop("_c0")  # Drop unnecessary column if exists
)

# Ensure proper data types
merged_df = merged_df.withColumn("Year", F.col("Year").cast("integer"))

# Rename columns if necessary (to match expectations)
if "Municipality" in merged_df.columns:
    merged_df = merged_df.withColumnRenamed("Municipality", "Municipality_Name")

# Add Municipality_Lowercase if not already present
if "Municipality_Lowercase" not in merged_df.columns:
    merged_df = merged_df.withColumn(
        "Municipality_Lowercase",
        F.regexp_replace(F.lower(F.trim(col("Municipality_Name"))), r"\s+", "-")
    )

print("Loaded merged data sample:")
merged_df.show(5, truncate=False)

################################################################################
# 4) LOAD SPENDING CSV WITH SUBCATEGORIES & PIVOT THE DATA
################################################################################

csv_path = "data/Bildungsausgaben_Gemeinden_Oberösterreich_data_2007_bis_2019.csv"

# Read the CSV into a Spark DataFrame
spend_raw = spark.read.csv(csv_path, header=True, sep=",", inferSchema=True)

# Rename columns for clarity
spend_raw = spend_raw.withColumnRenamed("Gemeinde", "Municipality")\
                     .withColumnRenamed("Year", "Year")\
                     .withColumnRenamed("Abschnitt", "Subcategory")\
                     .withColumnRenamed("Betrag in Euro", "Spending")

# Add a lowercase municipality column for joining later
spend_raw = spend_raw.withColumn(
    "Municipality_Lowercase",
    F.regexp_replace(F.lower(F.trim(col("Municipality"))), r"\s+", "-")
)

# Ensure proper types
spend_raw = spend_raw.withColumn("Year", col("Year").cast("int"))\
                     .withColumn("Spending", col("Spending").cast("float"))

print("Raw subcategory spending sample:")
spend_raw.show(10, truncate=False)

# Pivot the spending data so that each subcategory becomes its own column.
pivot_spend = spend_raw.groupBy("Municipality_Lowercase", "Year") \
                       .pivot("Subcategory") \
                       .agg(F.sum("Spending"))

# Replace any nulls with 0
pivot_spend = pivot_spend.na.fill(0)

print("Pivoted spending data (pre-renaming):")
pivot_spend.show(10, truncate=False)

# Rename each subcategory column to a safe name (prefix with "Sp_")
for old_col in pivot_spend.columns:
    if old_col not in ["Municipality_Lowercase", "Year"]:
        new_col = clean_subcat_name(old_col)
        pivot_spend = pivot_spend.withColumnRenamed(old_col, new_col)

print("Pivoted spending data (with cleaned column names):")
pivot_spend.show(10, truncate=False)

################################################################################
# 5) MERGE ELECTION RESULTS WITH SPENDING DATA
################################################################################

merged_df = merged_df.join(pivot_spend, on=["Municipality_Lowercase", "Year"], how="inner")
# Drop rows with missing Winning_Party
merged_df = merged_df.na.drop(subset=["Winning_Party"])

# Fill missing numeric values with 0 (for all numeric columns)
numeric_cols = [f.name for f in merged_df.schema.fields if f.dataType in [IntegerType(), FloatType()]]
for ncol in numeric_cols:
    merged_df = merged_df.withColumn(ncol, when(col(ncol).isNull(), 0).otherwise(col(ncol)))

print("Merged election + spending data (sample):")
merged_df.cache()
merged_df.show(10, truncate=False)

################################################################################
# 6) PREPARE THE DATA FOR MODELING
################################################################################

# Clean the "Wahlbeteiligung" column: remove "%" signs, replace commas with dots, remove extraneous characters.
merged_df = merged_df.withColumn("Wahlbeteiligung_clean",
                                 regexp_replace(col("Wahlbeteiligung"), "%", ""))
merged_df = merged_df.withColumn("Wahlbeteiligung_clean",
                                 regexp_replace(col("Wahlbeteiligung_clean"), ",", "."))
merged_df = merged_df.withColumn("Wahlbeteiligung_clean",
                                 regexp_replace(col("Wahlbeteiligung_clean"), "[^\\d.]", ""))
merged_df = merged_df.withColumn("Wahlbeteiligung_clean", col("Wahlbeteiligung_clean").cast("float"))

# Replace the original column with the cleaned one
merged_df = merged_df.drop("Wahlbeteiligung") \
                     .withColumnRenamed("Wahlbeteiligung_clean", "Wahlbeteiligung")

# Identify subcategory columns (columns that start with "Sp_")—excluding any unwanted ones (e.g. "Sp_Summe")
all_cols = merged_df.columns
subcat_cols = [c for c in all_cols if c.startswith("Sp_") and c != "Sp_Summe"]
# We’ll use these plus "Wahlbeteiligung" as our feature set
feature_cols = subcat_cols + ["Wahlbeteiligung"]

print("Feature columns used for modeling:", feature_cols)

# Ensure that all feature columns are of type float
for f in feature_cols:
    merged_df = merged_df.withColumn(f, col(f).cast("float"))

merged_df.cache()
print("Merged data (post cleaning):")
merged_df.show(10, truncate=False)

################################################################################
# 7) BUILD THE SPARK ML PIPELINE AND TRAIN A RANDOM FOREST MODEL
################################################################################

# Step 1: Convert the Winning_Party string label into a numeric label.
label_indexer = StringIndexer(inputCol="Winning_Party", outputCol="label", handleInvalid="skip")

# Step 2: Assemble all feature columns into a feature vector.
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

# Step 3: Define a Random Forest classifier.
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=30, maxDepth=5, seed=42)

# Create the pipeline
pipeline = Pipeline(stages=[label_indexer, assembler, rf_classifier])

# Split the data into training and testing sets.
train_df, test_df = merged_df.randomSplit([0.8, 0.2], seed=42)

# Train the pipeline.
model = pipeline.fit(train_df)

# Make predictions on the test set.
predictions = model.transform(test_df)

print("Predictions (sample):")
predictions.select("Municipality_Name", "Year", "Winning_Party", "prediction").show(10, truncate=False)

# Evaluate the model's accuracy.
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"\nTest Accuracy = {accuracy * 100:.2f}%\n")

# Display feature importances.
rf_model = model.stages[-1]
importances = rf_model.featureImportances
print("Random Forest Feature Importances:\n", importances)

print("\nFeature importances by column:")
for i, feat_name in enumerate(assembler.getInputCols()):
    print(f"  {feat_name}: {importances[i]:.4f}")

print("\nStringIndexer label mapping (0.0-based):", model.stages[0].labels)

################################################################################
# 8) ONE-VS-REST BINARY CLASSIFICATION FOR SELECTED PARTIES
################################################################################

# Assemble a separate features vector for one-vs-rest classifiers.
assembler_ovr = VectorAssembler(inputCols=feature_cols, outputCol="features_ovr", handleInvalid="skip")

# Define the parties for which to build a binary classifier.
parties = ["OVP", "SPO", "FPO"]
results_ovr = []

for party in parties:
    print(f"\n=== One-vs-Rest for: {party} vs. ALL ===")
    # Create a binary label: 1 if Winning_Party equals the current party, else 0.
    df_party = merged_df.withColumn("target", when(col("Winning_Party") == party, 1).otherwise(0).cast(IntegerType()))
    df_party = assembler_ovr.transform(df_party)
    
    # Split into training and testing sets.
    train_party, test_party = df_party.randomSplit([0.8, 0.2], seed=42)
    
    # Train a Random Forest classifier for this binary task.
    rf_bin = RandomForestClassifier(labelCol="target", featuresCol="features_ovr", numTrees=30, maxDepth=5, seed=42)
    rf_bin_model = rf_bin.fit(train_party)
    
    # Predict on the test set.
    predictions_bin = rf_bin_model.transform(test_party)
    
    # Evaluate using areaUnderROC.
    evaluator_bin = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator_bin.evaluate(predictions_bin)
    
    print(f"Area Under ROC for {party} vs. All: {auc:.3f}")
    
    # Print feature importances for this classifier.
    importances_bin = rf_bin_model.featureImportances
    for i, feat_name in enumerate(feature_cols):
        print(f"  {feat_name}: {importances_bin[i]:.4f}")
    
    results_ovr.append({"party": party, "AUC": auc, "importances": importances_bin})

print("\nDone with one-vs-rest training for the 3 major parties.\n")

################################################################################
# 9) ANALYZE FPÖ WIN DISTRIBUTION BY TURNOUT LEVEL USING SPARK
################################################################################

# Load a merged CSV file into Spark (this file should contain columns including "Wahlbeteiligung" and "Winning_Party")
merged_data = spark.read.csv("data/merged_data.csv", header=True, inferSchema=True)
merged_data = merged_data.withColumn("Wahlbeteiligung", col("Wahlbeteiligung").cast("float"))

# Compute the median turnout using approxQuantile.
median_turnout = merged_data.approxQuantile("Wahlbeteiligung", [0.5], 0.01)[0]
print(f"Median Turnout: {median_turnout:.2f}%")

# Classify each municipality as "High Turnout" or "Low Turnout"
merged_data = merged_data.withColumn("Turnout_Category",
                                     when(col("Wahlbeteiligung") >= median_turnout, "High Turnout")
                                     .otherwise("Low Turnout"))

# Count FPÖ wins in each turnout category.
fpö_wins = merged_data.filter(col("Winning_Party") == "FPO") \
                      .groupBy("Turnout_Category") \
                      .count()
                      
total_count = fpö_wins.agg(F.sum("count").alias("total_count")).collect()[0]["total_count"]

# Calculate percentage
fpö_wins = fpö_wins.withColumn("Percentage", (col("count") / total_count) * 100)

print("FPÖ Win Distribution by Turnout Level:")
fpö_wins.show()

################################################################################
# FINISH UP
################################################################################

spark.stop()
print("\nDone. Script finished successfully.")