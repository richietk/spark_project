#!/usr/bin/env python3

import pandas as pd
import numpy as np

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

################################################################################
# 1. DEFINE A HELPER FUNCTION FOR “CLOSEST YEAR” LOOKUP
################################################################################

def find_closest_value(muni_slug, election_year, df, value_col):
    """
    Return the value of `value_col` in `df` for the row whose orig_year
    is closest to `election_year`, among rows with the same municipality slug.
    If no match is found, return np.nan.
    """
    sub = df[df["Municipality_Lowercase"] == muni_slug]
    if sub.empty:
        return np.nan
    
    # We'll compute absolute difference in years to find the best match
    sub = sub.assign(year_diff=(sub["orig_year"] - election_year).abs())
    best_idx = sub["year_diff"].idxmin()
    return sub.loc[best_idx, value_col]


################################################################################
# 2. LOAD EXISTING ELECTION+SPENDING DATA (merged_data.csv)
################################################################################

merged_data_path = "data/merged_data.csv"
merged_df = pd.read_csv(merged_data_path)
print(f"Loaded existing merged dataset from {merged_data_path}")
print("Sample of merged_df:\n", merged_df.head(), "\n")

################################################################################
# 3. IMPORT DEMOGRAPHIC DATA & BUILD RATIO COLUMNS
#
# We'll rename their 'YEAR' -> 'orig_year' to avoid clashing with the election
# data's 'Year'. Then we create ratio columns like 'Austria_Ratio',
# 'Uni_Grad_Ratio', 'Pop_65plus_Ratio'. 
#
# The sets we have are:
#   OOE_Bev_Staatsangehoerigkeit.csv   (has years 2023,...,2019,...,2013,...)
#   OOE_Bev_Hoechste_abgeschl_Ausbildung.csv (years 2022,2021,2011,...)
#   OOE_Bev_laut_Volkszaehlung_Geschl_Alt5J.csv (years 1971,...,2021)
################################################################################

# ------------------------------------------------------------------------------
# A) Staatsangehoerigkeit (Nationality)
# ------------------------------------------------------------------------------
df_nation = pd.read_csv("data/OOE_Bev_Staatsangehoerigkeit.csv", sep=";", encoding="latin1")

df_nation["Municipality_Lowercase"] = (
    df_nation["LAU2_NAME"]
    .str.lower()
    .str.replace(" ", "-", regex=True)
)
df_nation = df_nation.rename(columns={
    "YEAR": "orig_year",
    "NATION_AUSTRIA": "Nation_Austria",
    "NATION_TOTAL": "Nation_Total"
})

df_nation["Nation_Austria"] = pd.to_numeric(df_nation["Nation_Austria"], errors="coerce")
df_nation["Nation_Total"] = pd.to_numeric(df_nation["Nation_Total"], errors="coerce")

df_nation["Austria_Ratio"] = df_nation["Nation_Austria"] / df_nation["Nation_Total"]

# Keep only what's needed
df_nation = df_nation[["Municipality_Lowercase", "orig_year", "Austria_Ratio"]]


# ------------------------------------------------------------------------------
# B) Höchste_abgeschl_Ausbildung (Education level)
# ------------------------------------------------------------------------------
df_edu = pd.read_csv("data/OOE_Bev_Hoechste_abgeschl_Ausbildung.csv", sep=";", encoding="latin1")

df_edu["Municipality_Lowercase"] = (
    df_edu["COMMUNE_NAME"]
    .str.lower()
    .str.replace(" ", "-", regex=True)
)
df_edu = df_edu.rename(columns={
    "YEAR": "orig_year",
    "EDU_UNIVERSITY_FACHHOCHSCHULE": "Uni_Grads",
    "EDU_TOTAL": "Edu_Total"
})

df_edu["Uni_Grads"] = pd.to_numeric(df_edu["Uni_Grads"], errors="coerce")
df_edu["Edu_Total"] = pd.to_numeric(df_edu["Edu_Total"], errors="coerce")
df_edu["Uni_Grad_Ratio"] = df_edu["Uni_Grads"] / df_edu["Edu_Total"]

df_edu = df_edu[["Municipality_Lowercase", "orig_year", "Uni_Grad_Ratio"]]


# ------------------------------------------------------------------------------
# C) Bevölkerung laut Volkszählung (Age/Gender)
# ------------------------------------------------------------------------------
df_age = pd.read_csv("data/OOE_Bev_laut_Volkszaehlung_Geschl_Alt5J.csv", sep=";", encoding="latin1")

df_age["Municipality_Lowercase"] = (
    df_age["LAU2_NAME"]
    .str.lower()
    .str.replace(" ", "-", regex=True)
)

# We'll sum up the 65+ columns
age_cols_65plus = ["AGE_65_TO_69", "AGE_70_TO_74", "AGE_75_TO_79", 
                   "AGE_80_TO_84", "AGE_85_TO_89", "AGE_90_PLUS"]
for c in age_cols_65plus:
    df_age[c] = pd.to_numeric(df_age[c], errors="coerce").fillna(0)
df_age["AGE_TOTAL"] = pd.to_numeric(df_age["AGE_TOTAL"], errors="coerce").fillna(0)

# create a row-wise sum of 65+ population
df_age["POP_65plus"] = df_age[age_cols_65plus].sum(axis=1)

# rename YEAR -> orig_year
df_age = df_age.rename(columns={"YEAR": "orig_year"})

# group by municipality-year to combine possible sex=1/2 rows
grouped_age = df_age.groupby(["Municipality_Lowercase", "orig_year"], as_index=False).agg({
    "POP_65plus": "sum",
    "AGE_TOTAL": "sum"
})

grouped_age["Pop_65plus_Ratio"] = grouped_age["POP_65plus"] / grouped_age["AGE_TOTAL"]
df_age = grouped_age[["Municipality_Lowercase","orig_year","Pop_65plus_Ratio"]]

################################################################################
# 4. FOR EACH ROW IN merged_df, FIND THE CLOSEST YEAR DEMOGRAPHIC VALUES
################################################################################

def assign_closest_demographics(row):
    muni = row["Municipality_Lowercase"]
    yr = row["Year"]
    
    # Austria_Ratio
    a_ratio = find_closest_value(muni, yr, df_nation, "Austria_Ratio")
    
    # Uni_Grad_Ratio
    u_ratio = find_closest_value(muni, yr, df_edu, "Uni_Grad_Ratio")
    
    # Pop_65plus_Ratio
    p_ratio = find_closest_value(muni, yr, df_age, "Pop_65plus_Ratio")
    
    return pd.Series({
        "Austria_Ratio": a_ratio,
        "Uni_Grad_Ratio": u_ratio,
        "Pop_65plus_Ratio": p_ratio
    })

# Apply row-by-row
merged_df[["Austria_Ratio","Uni_Grad_Ratio","Pop_65plus_Ratio"]] = (
    merged_df.apply(assign_closest_demographics, axis=1)
)

# Now each row has approximate values for these three columns
print("\nAfter approximate merges, sample:\n", merged_df.head(), "\n")

################################################################################
# 5. CLEAN / FILTER / PROCEED WITH SPARK ML
################################################################################

# Drop any rows missing key columns
merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
merged_df = merged_df.dropna(subset=[
    "Spending_Summe",
    "Wahlbeteiligung",
    "Winning_Party",
    "Austria_Ratio",
    "Uni_Grad_Ratio",
    "Pop_65plus_Ratio"
])
print(f"Number of rows after final cleaning: {len(merged_df)}")

# Start Spark session
spark = SparkSession.builder.appName("PredictWinningParty_DemographicsClosestYear").getOrCreate()

# Convert Pandas -> Spark DataFrame
spark_df = spark.createDataFrame(merged_df)

# Ensure correct types
spark_df = spark_df.withColumn("Spending_Summe", col("Spending_Summe").cast("float"))
spark_df = spark_df.withColumn("Wahlbeteiligung", col("Wahlbeteiligung").cast("float"))
spark_df = spark_df.withColumn("Austria_Ratio", col("Austria_Ratio").cast("float"))
spark_df = spark_df.withColumn("Uni_Grad_Ratio", col("Uni_Grad_Ratio").cast("float"))
spark_df = spark_df.withColumn("Pop_65plus_Ratio", col("Pop_65plus_Ratio").cast("float"))

# Label indexer
label_indexer = StringIndexer(
    inputCol="Winning_Party",
    outputCol="label",
    handleInvalid="skip"
)

# Assemble feature vector
assembler = VectorAssembler(
    inputCols=[
        "Spending_Summe",
        "Wahlbeteiligung",
        "Austria_Ratio",
        "Uni_Grad_Ratio",
        "Pop_65plus_Ratio",
    ],
    outputCol="features",
    handleInvalid="skip"
)

# RandomForest
rf_classifier = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=5,
    seed=42
)

# Pipeline
pipeline = Pipeline(stages=[label_indexer, assembler, rf_classifier])

# Train/Test Split
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)

# Predictions
predictions = model.transform(test_df)
predictions.select("Municipality_Name","Year","Winning_Party","prediction").show(10, truncate=False)

# Evaluate Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"\nTest Accuracy = {accuracy * 100:.2f}%\n")

# Feature Importances
rf_model = model.stages[-1]
importances = rf_model.featureImportances
feature_cols = assembler.getInputCols()

print("Random Forest Feature Importances (Spark Vector):", importances)
print("Feature columns:", feature_cols)
for idx, fname in enumerate(feature_cols):
    print(f"  {fname}: {importances[idx]:.4f}")

# Label mapping
label_stage = model.stages[0]  # The StringIndexer
print("\nStringIndexer label mapping (0.0-based):", label_stage.labels)

spark.stop()
print("\nDone. Script finished successfully.")
