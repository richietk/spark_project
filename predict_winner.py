#!/usr/bin/env python3
"""
Scalable version of the election+demographics processing and ML pipeline.
Replaces all Pandas operations with Spark DataFrame operations.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ------------------------------------------------------------------------------
# 1. START SPARK SESSION
# ------------------------------------------------------------------------------
spark = SparkSession.builder.appName("PredictWinningParty_Scalable").getOrCreate()

# ------------------------------------------------------------------------------
# 2. LOAD EXISTING ELECTION+SPENDING DATA (merged_data.csv)
# ------------------------------------------------------------------------------
merged_data_path = "data/merged_data.csv"
# Adjust options if needed (for example, if there is a nonstandard delimiter)
merged_df = (
    spark.read
         .option("header", True)
         .option("inferSchema", True)
         .option("sep", ",")  # Ensure the separator is correct (replace with `;` if needed)
         .option("enforceSchema", True)
         .csv(merged_data_path)
         .drop("_c0")  # Remove unnecessary column if `_c0` appears
)

print(f"Loaded merged dataset from {merged_data_path}")
merged_df.show(5, truncate=False)

# To join later on demographic data we add a unique ID.
merged_df = merged_df.withColumn("id", F.monotonically_increasing_id())

# Ensure the election year is numeric (cast to integer if needed)
merged_df = merged_df.withColumn("Year", F.col("Year").cast("integer"))

# ------------------------------------------------------------------------------
# 3. IMPORT DEMOGRAPHIC DATA & BUILD RATIO COLUMNS
#     (Nationality, Education, Age)
# ------------------------------------------------------------------------------
# Helper: a function to create a "Municipality_Lowercase" column from a source column.
def add_muni_col(df, source_col):
    return df.withColumn(
        "Municipality_Lowercase",
        F.regexp_replace(F.lower(F.col(source_col)), " ", "-")
    )

# --- A) NATIONALITY (Staatsangehoerigkeit) ---
nation_path = "data/OOE_Bev_Staatsangehoerigkeit.csv"
df_nation = (
    spark.read
         .option("header", True)
         .option("sep", ";")
         .option("encoding", "latin1")
         .option("inferSchema", True)
         .csv(nation_path)
)
df_nation = add_muni_col(df_nation, "LAU2_NAME")
df_nation = df_nation.withColumnRenamed("YEAR", "orig_year") \
                     .withColumnRenamed("NATION_AUSTRIA", "Nation_Austria") \
                     .withColumnRenamed("NATION_TOTAL", "Nation_Total")
# Cast to numeric (float) as needed
df_nation = df_nation.withColumn("Nation_Austria", F.col("Nation_Austria").cast("float")) \
                     .withColumn("Nation_Total", F.col("Nation_Total").cast("float")) \
                     .withColumn("orig_year", F.col("orig_year").cast("integer"))
df_nation = df_nation.withColumn("Austria_Ratio", F.col("Nation_Austria") / F.col("Nation_Total"))
df_nation = df_nation.select("Municipality_Lowercase", "orig_year", "Austria_Ratio")

# --- B) EDUCATION LEVEL (Höchste_abgeschl_Ausbildung) ---
edu_path = "data/OOE_Bev_Hoechste_abgeschl_Ausbildung.csv"
df_edu = (
    spark.read
         .option("header", True)
         .option("sep", ";")
         .option("encoding", "latin1")
         .option("inferSchema", True)
         .csv(edu_path)
)
df_edu = add_muni_col(df_edu, "COMMUNE_NAME")
df_edu = df_edu.withColumnRenamed("YEAR", "orig_year") \
               .withColumnRenamed("EDU_UNIVERSITY_FACHHOCHSCHULE", "Uni_Grads") \
               .withColumnRenamed("EDU_TOTAL", "Edu_Total")
df_edu = df_edu.withColumn("Uni_Grads", F.col("Uni_Grads").cast("float")) \
               .withColumn("Edu_Total", F.col("Edu_Total").cast("float")) \
               .withColumn("orig_year", F.col("orig_year").cast("integer"))
df_edu = df_edu.withColumn("Uni_Grad_Ratio", F.col("Uni_Grads") / F.col("Edu_Total"))
df_edu = df_edu.select("Municipality_Lowercase", "orig_year", "Uni_Grad_Ratio")

# --- C) AGE / POPULATION (Bevölkerung laut Volkszählung) ---
age_path = "data/OOE_Bev_laut_Volkszaehlung_Geschl_Alt5J.csv"
df_age = (
    spark.read
         .option("header", True)
         .option("sep", ";")
         .option("encoding", "latin1")
         .option("inferSchema", True)
         .csv(age_path)
)
df_age = add_muni_col(df_age, "LAU2_NAME")
# Define the age columns for 65+
age_cols_65plus = [
    "AGE_65_TO_69", "AGE_70_TO_74", "AGE_75_TO_79",
    "AGE_80_TO_84", "AGE_85_TO_89", "AGE_90_PLUS"
]
# Ensure the age columns and AGE_TOTAL are numeric (fill missing with 0)
for c in age_cols_65plus:
    df_age = df_age.withColumn(c, F.when(F.col(c).isNull(), 0)
                                       .otherwise(F.col(c).cast("float")))
df_age = df_age.withColumn("AGE_TOTAL", F.when(F.col("AGE_TOTAL").isNull(), 0)
                                         .otherwise(F.col("AGE_TOTAL").cast("float")))
# Create row-wise sum of 65+ population
df_age = df_age.withColumn("POP_65plus", sum([F.col(c) for c in age_cols_65plus]))
df_age = df_age.withColumnRenamed("YEAR", "orig_year") \
               .withColumn("orig_year", F.col("orig_year").cast("integer"))
# Group by municipality and orig_year to combine potential duplicate rows
df_age = df_age.groupBy("Municipality_Lowercase", "orig_year") \
               .agg(
                   F.sum("POP_65plus").alias("POP_65plus"),
                   F.sum("AGE_TOTAL").alias("AGE_TOTAL")
               )
df_age = df_age.withColumn("Pop_65plus_Ratio", F.col("POP_65plus") / F.col("AGE_TOTAL"))
df_age = df_age.select("Municipality_Lowercase", "orig_year", "Pop_65plus_Ratio")

# ------------------------------------------------------------------------------
# 4. FOR EACH ROW IN merged_df, FIND THE CLOSEST YEAR DEMOGRAPHIC VALUES
#     (Using window functions on the join between merged_df and each demographic DF)
# ------------------------------------------------------------------------------
# We join on Municipality_Lowercase and compute the absolute difference in years.
# Then, by partitioning on the unique ID (added above) we select the row with minimum difference.


print((merged_df.count(), len(merged_df.columns)))
print((df_nation.count(), len(df_nation.columns)))
print((df_edu.count(), len(df_edu.columns)))
print((df_age.count(), len(df_age.columns)))

# --- Helper function for "closest" join ---
# Adjust the closest_demographic_join function to handle ties by orig_year descending
def closest_demographic_join(merged, demo_df, ratio_col):
    # Join without filtering to calculate the absolute year difference
    joined = merged.join(demo_df, on="Municipality_Lowercase", how="left") \
                   .withColumn("year_diff", F.abs(F.col("Year") - F.col("orig_year")))

    # Find the row with the smallest difference using groupBy
    closest = (
        joined.groupBy("id")
              .agg(F.first(F.struct("year_diff", ratio_col)).alias("best_match"))
              .select("id", F.col("best_match." + ratio_col).alias(ratio_col))
    )

    return closest


nation_closest = closest_demographic_join(merged_df, df_nation, "Austria_Ratio")
edu_closest = closest_demographic_join(merged_df, df_edu, "Uni_Grad_Ratio")
age_closest = closest_demographic_join(merged_df, df_age, "Pop_65plus_Ratio")

merged_df = merged_df.join(nation_closest, on="id", how="left") \
                     .join(edu_closest, on="id", how="left") \
                     .join(age_closest, on="id", how="left")
print(f"Spark: Row count after joining demographics: {merged_df.count()}")
print("\nAfter adding demographic ratios (closest year):")
merged_df.select("Municipality_Lowercase", "Year", "Austria_Ratio", "Uni_Grad_Ratio", "Pop_65plus_Ratio") \
         .show(5, truncate=False)

# ------------------------------------------------------------------------------
# 5. CLEAN / FILTER / PREPARE DATA FOR SPARK ML
# ------------------------------------------------------------------------------
# Drop rows with missing key columns.
# Handle inf/-inf before filtering
merged_df = merged_df.replace(float("inf"), None).replace(float("-inf"), None)

# Drop rows with missing key columns (including those with inf converted to null)
cols_required = ["Spending_Summe", "Wahlbeteiligung", "Winning_Party",
                 "Austria_Ratio", "Uni_Grad_Ratio", "Pop_65plus_Ratio"]
                 
merged_df = merged_df.dropna(subset=cols_required)
    
print(f"Spark: Row count after filtering nulls: {merged_df.count()}")
print(f"Number of rows after cleaning: {merged_df.count()}")

# Ensure correct types for ML features (casting to float)
merged_df = merged_df.withColumn("Spending_Summe", F.col("Spending_Summe").cast("float")) \
                     .withColumn("Wahlbeteiligung", F.col("Wahlbeteiligung").cast("float")) \
                     .withColumn("Austria_Ratio", F.col("Austria_Ratio").cast("float")) \
                     .withColumn("Uni_Grad_Ratio", F.col("Uni_Grad_Ratio").cast("float")) \
                     .withColumn("Pop_65plus_Ratio", F.col("Pop_65plus_Ratio").cast("float"))

print("Spark: DataFrame Schema:")
merged_df.printSchema()

# ------------------------------------------------------------------------------
# 6. BUILD SPARK ML PIPELINE
# ------------------------------------------------------------------------------
# Index the label column ("Winning_Party")
# Extract unique labels from Pandas dataset (assuming pandas_df is available)
party_labels = merged_df.select("Winning_Party").distinct().rdd.flatMap(lambda x: x).collect()

# Ensure StringIndexer follows this order instead of lexicographic order
label_indexer = StringIndexer(
    inputCol="Winning_Party",
    outputCol="label",
    handleInvalid="skip"
).setStringOrderType("alphabetDesc")  # Instead of "alphabetAsc"



# Assemble feature vector from selected columns
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

rf_classifier = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,  # Number of trees
    maxDepth=5,  # Maximum depth of trees
    seed=42
)

# Create the pipeline
pipeline = Pipeline(stages=[label_indexer, assembler, rf_classifier])

# ------------------------------------------------------------------------------
# 7. TRAIN/TEST SPLIT, TRAIN MODEL, AND EVALUATE
# ------------------------------------------------------------------------------
merged_df.select("Winning_Party").distinct().show()

train_df, test_df = merged_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)

# Get predictions on the test set
predictions = model.transform(test_df)
# Show a sample of predictions (assumes there is a "Municipality_Name" column)
predictions.select("Municipality_Name", "Year", "Winning_Party", "prediction") \
           .show(10, truncate=False)

# Evaluate classification accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"\nTest Accuracy = {accuracy * 100:.2f}%\n")

# Feature importances (from the RandomForest model in the pipeline)
rf_model = model.stages[-1]
importances = rf_model.featureImportances
feature_cols = assembler.getInputCols()

print("Random Forest Feature Importances (Spark Vector):", importances)
print("Feature columns:", feature_cols)
for idx, fname in enumerate(feature_cols):
    # Feature importances is a vector so we extract using indexing
    print(f"  {fname}: {importances[idx]:.4f}")

# Print label mapping (from the StringIndexer stage)
label_stage = model.stages[0]  # The StringIndexer
print("\nStringIndexer label mapping (0.0-based):", label_stage.labels)

# ------------------------------------------------------------------------------
# 8. FINISH
# ------------------------------------------------------------------------------
spark.stop()
print("\nDone. Script finished successfully.")
