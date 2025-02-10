#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import re

# PySpark imports
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when
from pyspark.sql.types import IntegerType
from pyspark.ml.evaluation import BinaryClassificationEvaluator

################################################################################
# 1) HELPER FUNCTIONS
################################################################################

def slugify_municipality(name: str) -> str:
    """
    Convert a municipality string to lowercase, strip whitespace,
    and replace spaces with dashes (and remove other weird chars if needed).
    """
    if not isinstance(name, str):
        return None
    name = name.lower().strip()
    # Replace spaces with dashes
    name = re.sub(r"\\s+", "-", name)
    return name

################################################################################
# 2) LOAD & FLATTEN ELECTION RESULTS (similar to spending_turnout.py)
################################################################################

with open("data/election_results.json", "r", encoding="utf-8") as f:
    election_data = json.load(f)

records = []
for year_str, data in election_data.items():
    year = int(year_str)
    for wahlkreis in data.get("wahlkreise", {}).values():
        for bezirk in wahlkreis.get("bezirke", {}).values():
            for municipality_id, details in bezirk.get("municipalities", {}).items():
                votes = details.get("votes", {})
                # Extract party votes (excluding 'Wahlberechtigte', 'abgegebene Stimmen', etc.)
                party_votes = {
                    k: v for k, v in votes.items()
                    if k not in ["Wahlberechtigte", "abgegebene Stimmen", "gültige Stimmen", "Wahlbeteiligung"]
                }
                winning_party = max(party_votes, key=party_votes.get) if party_votes else None
                records.append({
                    "Year": year,
                    "Municipality_ID": municipality_id,
                    "Municipality_Name": details.get("name", None),
                    "Wahlberechtigte": votes.get("Wahlberechtigte"),
                    "abgegebene_Stimmen": votes.get("abgegebene Stimmen"),
                    "gueltige_Stimmen": votes.get("gültige Stimmen"),
                    "Wahlbeteiligung": votes.get("Wahlbeteiligung"),
                    "Winning_Party": winning_party
                })

election_df = pd.DataFrame(records)

# Create municipality slug
election_df["Municipality_Lowercase"] = election_df["Municipality_Name"].apply(slugify_municipality)

print("Election data (head):\n", election_df.head(), "\n")

################################################################################
# 3) LOAD SPENDING CSV WITH SUBCATEGORIES
#
# This CSV has columns like:
#   Gemeinde | Year | Abschnitt | Betrag in Euro
# We'll pivot the data so each subcategory (Abschnitt) becomes its own column.
################################################################################

csv_path = "data/Bildungsausgaben_Gemeinden_Oberösterreich_data_2007_bis_2019.csv"

# NOTE: If this file is truly tab-delimited, use sep='\\t'.
# If it's something else (e.g. semicolon), adjust accordingly.
spend_raw = pd.read_csv(csv_path, sep=",", engine="python")

# Example columns: "Gemeinde", "Year", "Abschnitt", "Betrag in Euro"
spend_raw.rename(
    columns={
        "Gemeinde": "Municipality",
        "Year": "Year",
        "Abschnitt": "Subcategory",
        "Betrag in Euro": "Spending"
    },
    inplace=True
)
spend_raw["Municipality_Lowercase"] = spend_raw["Municipality"].apply(slugify_municipality) # might be redundant
spend_raw["Year"] = spend_raw["Year"].astype(int)
spend_raw["Spending"] = pd.to_numeric(spend_raw["Spending"], errors="coerce")

print("Raw subcategory spending sample:\n", spend_raw.head(), "\n")

# Pivot to wide format: each row is (Municipality_Lowercase, Year) with
# separate columns for each subcategory (e.g. 'Allgemeinbildender Unterricht', etc.)
pivot_spend = spend_raw.pivot_table(
    index=["Municipality_Lowercase","Year"],
    columns="Subcategory",
    values="Spending",
    aggfunc="sum"  # In case there's more than one row per subcategory
).reset_index()

# Some subcategory names can have strange chars, so let's do a safe rename
# for each subcategory col => "Sp_{clean_subcat_name}"
def clean_subcat_name(name):
    # remove special chars, spaces, unify
    # e.g. 'Allgemeinbildender Unterricht' -> 'Sp_Allgemeinbildender_Unterricht'
    c = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    return f"Sp_{c}"

# We skip the first two columns in pivot_spend (Municipality_Lowercase, Year).
cat_cols = pivot_spend.columns[2:]
renamer = {}
for cat in cat_cols:
    new_col = clean_subcat_name(str(cat))
    renamer[cat] = new_col

pivot_spend.rename(columns=renamer, inplace=True)
print("Pivoted subcategories (head):\n", pivot_spend.head(6), "\n")

################################################################################
# 4) MERGE ELECTION_DF WITH SUBCATEGORY SPEND DF
################################################################################

merged_df = pd.merge(
    election_df,
    pivot_spend,
    on=["Municipality_Lowercase","Year"],
    how="inner"
)

print("Merged election + subcat spend (head):\n", merged_df.head(), "\n")

################################################################################
# 5) BUILD A SPARK MODEL
#    We'll treat each subcategory column as a separate feature. We can also
#    include 'Wahlbeteiligung' if we want it as a feature. 
#
#    For brevity, we skip demographics. You can do the same
#    'closest year' approach if desired. 
################################################################################

# Convert numeric columns
#  - We might want to treat 'Wahlbeteiligung' as numeric if we want it as a feature
#  - We'll parse it removing % if needed.
def clean_numeric(series):
    return (series.astype(str)
            .str.replace("%","", regex=False)
            .str.replace(",",".", regex=False)
            .str.replace("[^\\d.]", "", regex=True)
            .astype(float))

merged_df["Wahlbeteiligung"] = clean_numeric(merged_df["Wahlbeteiligung"])

# Drop rows with missing key info
merged_df = merged_df.dropna(subset=["Winning_Party"])
# Also drop if subcategories are all NaN
# Alternatively, we can fillna(0) if that makes sense
merged_df.fillna(0, inplace=True)

# Start Spark
spark = SparkSession.builder.appName("PredictWinningParty_SubcategorySpending").getOrCreate()

# Convert to Spark DF
spark_df = spark.createDataFrame(merged_df)

# Identify subcategory columns
all_cols = list(merged_df.columns)
# We'll gather columns that start with "Sp_" as features
subcat_cols = [c for c in pivot_spend.columns if c.startswith("Sp_") and c != "Sp_Summe"]
# Then build feature_cols from subcat_cols + maybe "Wahlbeteiligung"
feature_cols = subcat_cols + ["Wahlbeteiligung"]
print("Using feature columns:\n", feature_cols)

# Cast numeric features
for fcol in feature_cols:
    spark_df = spark_df.withColumn(fcol, col(fcol).cast("float"))

# Index label (Winning_Party)
label_indexer = StringIndexer(
    inputCol="Winning_Party",
    outputCol="label",
    handleInvalid="skip"
)

# Assemble features
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

# Random Forest
rf_classifier = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=5,
    seed=42
)

pipeline = Pipeline(stages=[label_indexer, assembler, rf_classifier])

# Train/Test split
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_df)
predictions = model.transform(test_df)

predictions.select(
    "Municipality_Name","Year","Winning_Party","prediction"
).show(10, truncate=False)

# Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"\nTest Accuracy = {accuracy * 100:.2f}%\n")

# Feature Importances
rf_model = model.stages[-1]
importances = rf_model.featureImportances
print("Random Forest Feature Importances:\n", importances)

# Map indices to subcategory col names
feat_cols = assembler.getInputCols()  # same as feature_cols
print("\nFeature importances by column:")
for i, col_name in enumerate(feat_cols):
    print(f"  {col_name}: {importances[i]:.4f}")

# Label mapping
label_stage = model.stages[0]
print("\nStringIndexer label mapping (0.0-based):", label_stage.labels)

# We'll build a new VectorAssembler for our binary tasks
assembler_ovr = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_ovr",
    handleInvalid="skip"
)

# We define the parties of interest
parties = ["ÖVP", "SPÖ", "FPÖ"]

# We'll store results for each party's classifier
results_ovr = []

for party in parties:
    print(f"\n=== One-vs-Rest for: {party} vs. ALL ===")
    
    # 1) Create a new binary label: 1 if Winning_Party == party, else 0
    df_party = spark_df.withColumn(
        "target",
        when(col("Winning_Party") == party, 1).otherwise(0).cast(IntegerType())
    )
    
    # 2) Assemble features
    df_party_assembled = assembler_ovr.transform(df_party)
    
    # 3) Split train/test
    train_df, test_df = df_party_assembled.randomSplit([0.8, 0.2], seed=42)
    
    # 4) Train a RandomForest on labelCol="target", featuresCol="features_ovr"
    rf_bin = RandomForestClassifier(
        labelCol="target",
        featuresCol="features_ovr",
        numTrees=50,
        maxDepth=5,
        seed=42
    )
    
    rf_bin_model = rf_bin.fit(train_df)
    
    # 5) Predict on test set
    predictions_bin = rf_bin_model.transform(test_df)
    
    # 6) Evaluate with a binary metric (e.g. areaUnderROC or accuracy)
    evaluator_bin = BinaryClassificationEvaluator(
        labelCol="target",
        rawPredictionCol="rawPrediction",  
        metricName="areaUnderROC"  # or 'areaUnderPR'
    )
    auc = evaluator_bin.evaluate(predictions_bin)
    
    # Alternatively, we can measure accuracy by converting 
    # the predicted probability to 0/1 threshold:
    # We'll do a quick approach using a small helper:
    bin_evaluator_accuracy = BinaryClassificationEvaluator(
        labelCol="target",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"  # We'll do areaUnderROC as a proxy
    )
    # If you want an actual accuracy measure, you'd do:
    # from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    # ... but that requires a label of 0/1 which we do have, 
    # just note it's a 2-class confusion matrix.

    print(f"Area Under ROC for {party} vs. All: {auc:.3f}")
    
    # 7) Print feature importances for this binary classifier
    importances_bin = rf_bin_model.featureImportances
    # Map indices -> feature_cols
    for i, feat_name in enumerate(feature_cols):
        print(f"  {feat_name}: {importances_bin[i]:.4f}")
    
    # Store results for later analysis
    results_ovr.append({
        "party": party,
        "AUC": auc,
        "importances": importances_bin
    })

# After this loop, you'll have separate random forests 
# that treat each party as "1" and everything else as "0".
# That yields per-party feature importance and performance metrics.

print("\nDone with one-vs-rest training for the 3 major parties.\n")

spark.stop()
print("\nDone. Script finished successfully.")


# misc: does FPÖ win in places where turnout is high, or where it's low?
df = pd.read_csv("data/merged_data.csv")

# Ensure numeric format for turnout
df["Wahlbeteiligung"] = pd.to_numeric(df["Wahlbeteiligung"], errors="coerce")

# Compute median turnout
median_turnout = df["Wahlbeteiligung"].median()

# Classify municipalities as "High Turnout" or "Low Turnout"
df["Turnout_Category"] = df["Wahlbeteiligung"].apply(lambda x: "High Turnout" if x >= median_turnout else "Low Turnout")

# Count FPÖ wins in both categories
fpö_wins = df[df["Winning_Party"] == "FPÖ"].groupby("Turnout_Category").size()

# Compute percentage distribution
fpö_win_percentages = (fpö_wins / fpö_wins.sum()) * 100

# Print results
print(f"Median Turnout: {median_turnout:.2f}%\n")
print("FPÖ Win Distribution by Turnout Level:")
print(fpö_win_percentages)
