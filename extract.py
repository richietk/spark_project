#!/usr/bin/env python3
# Filename: spark_elections_spending.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
import json
import os

################################################################################
# 1. Initialize Spark
################################################################################
spark = (
    SparkSession.builder
    .appName("MunicipalSpendingAndElectionAnalysis")
    .getOrCreate()
)

################################################################################
# 2. Read & Transform Municipal Spending Data
################################################################################
spending_csv = "data/Bildungsausgaben_Gemeinden_Oberösterreich_data_2007_bis_2019.csv"
df_spending = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")    # So numeric columns become floats/ints
    .option("encoding", "UTF-8")      # Change if your CSV uses different encoding
    .csv(spending_csv)
)

# Rename columns to something consistent
df_spending = (df_spending
    .withColumnRenamed("Gemeinde", "Municipality")
    .withColumnRenamed("Year", "Year")
    .withColumnRenamed("Abschnitt", "Category")
    .withColumnRenamed("Betrag in Euro", "Spending")
)

# Fix odd encodings in the Category column
df_spending = df_spending.withColumn(
    "Category",
    regexp_replace(col("Category"), "FĂ¶rderung", "Förderung")
)
df_spending = df_spending.withColumn(
    "Category",
    regexp_replace(col("Category"), "Sport und auĂźerschulische Leibeserziehung", "Sport und außerschulische Leibeserziehung")
)

# Cast columns to proper types
df_spending = df_spending.withColumn("Year", col("Year").cast("int"))
df_spending = df_spending.withColumn("Spending", col("Spending").cast("float"))

# Convert rows into the nested JSON structure
municipal_data = {}  # { municipality -> { year -> { "Summe": float, "Details": {...} } } }

# Using collect() because the dataset is presumably not huge. Otherwise, consider rdd aggregation.
for row in df_spending.collect():
    municipality = row["Municipality"]
    year = row["Year"]
    category = row["Category"]
    spending_val = row["Spending"]

    if municipality not in municipal_data:
        municipal_data[municipality] = {}
    if year not in municipal_data[municipality]:
        municipal_data[municipality][year] = {
            "Summe": None,
            "Details": {}
        }
    if category == "Summe":
        municipal_data[municipality][year]["Summe"] = spending_val
    else:
        municipal_data[municipality][year]["Details"][category] = spending_val

# Write out JSON for municipal spending
spending_json_path = "data/municipal_spending_new.json"
with open(spending_json_path, "w", encoding="utf-8") as f:
    json.dump(municipal_data, f, indent=4, ensure_ascii=False)

print(f"Municipal spending JSON saved to: {spending_json_path}")

################################################################################
# 3. Read & Transform Election Data (Votes + Mandates)
################################################################################

# The years we want
election_yrs = [2008, 2013, 2017, 2019, 2024]

# Master dictionaries to hold results
election_data = {}  # For votes
mandates_data = {}  # For mandates

# Function to identify "level" of location ID
def get_id_level(loc_id_str):
    """
    Example logic matching your original scripts:
      - ends with '000' -> wahlkreis
      - ends with '099' -> wahlkarten
      - ends with '00'  -> bezirk
      - else           -> municipality
    """
    if loc_id_str.endswith("000"):
        return "wahlkreis"
    elif loc_id_str.endswith("099"):
        return "wahlkarten"
    elif loc_id_str.endswith("00"):
        return "bezirk"
    else:
        return "municipality"

# Loop over each election year and read the two CSVs
for year in election_yrs:
    votes_file = f"data/OÖ_{year}_Stimmen.csv"
    mandates_file = f"data/OÖ_{year}_Mandate.csv"

    if not os.path.exists(votes_file) or not os.path.exists(mandates_file):
        print(f"[WARNING] Missing CSV(s) for year {year} -> Skipping.")
        continue

    # Read votes CSV with Spark
    votes_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("encoding", "latin1")
        .csv(votes_file)
    )

    # Read mandates CSV with Spark
    mandate_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("encoding", "latin1")
        .csv(mandates_file)
    )

    ############################################################################
    # 3A. Build election_data[year] (Votes)
    ############################################################################
    election_data[str(year)] = {
        "name": "Land Oberösterreich",
        "wahlkreise": {}
    }

    # Collect all rows from votes_df
    all_votes = votes_df.collect()

    # Common party columns that appear in your data
    # Adjust as needed if your CSV has slightly different or more parties

    for row in all_votes:
        loc_id = str(row["Nr."])  # e.g. 40000
        name = row["Name"]

        # Fix weird "Ober�sterreich" if needed:
        if isinstance(name, str) and "Ober�sterreich" in name:
            name = name.replace("Ober�sterreich", "Oberösterreich")

        # Construct the "votes" dict matching your original script’s keys
        votes_entry = {
            "name": name,
            "votes": {}
        }

        # Check for columns: "Wahlberechtigte", "abgegeb. Stimmen", "gültige", "Wahlbet."
        if "Wahlberechtigte" in votes_df.columns:
            votes_entry["votes"]["Wahlberechtigte"] = row["Wahlberechtigte"]
        if "abgegeb. Stimmen" in votes_df.columns:
            votes_entry["votes"]["abgegebene Stimmen"] = row["abgegeb. Stimmen"]
        if "gültige" in votes_df.columns:
            votes_entry["votes"]["gültige Stimmen"] = row["gültige"]
        if "Wahlbet." in votes_df.columns:
            votes_entry["votes"]["Wahlbeteiligung"] = row["Wahlbet."]

        party_columns = [col for col in votes_df.columns if col not in {"Nr.", "Name", "Wahlberechtigte", "abgegeb. Stimmen", "gültige", "Wahlbet.", "ungültige"}]

        for party_col in party_columns:
            votes_entry["votes"][party_col] = row[party_col]

        # Insert into the nested structure
        level = get_id_level(loc_id)
        if level == "wahlkreis":
            # This is a top-level wahlkreis
            election_data[str(year)]["wahlkreise"][loc_id] = {
                "name": name,
                "bezirke": {}
            }
        elif level == "bezirk":
            # Insert under parent wahlkreis
            parent_wahlkreis = loc_id[:2] + "000"  # e.g. "40100" -> "40000"
            if parent_wahlkreis in election_data[str(year)]["wahlkreise"]:
                election_data[str(year)]["wahlkreise"][parent_wahlkreis]["bezirke"][loc_id] = {
                    "name": name,
                    "municipalities": {}
                }
        elif level == "municipality":
            parent_bezirk = loc_id[:3] + "00"    # e.g. "40101" -> "40100"
            parent_wahlkreis = loc_id[:2] + "000"
            # Make sure parent objects exist
            if (
                parent_wahlkreis in election_data[str(year)]["wahlkreise"] and
                parent_bezirk in election_data[str(year)]["wahlkreise"][parent_wahlkreis]["bezirke"]
            ):
                election_data[str(year)]["wahlkreise"][parent_wahlkreis]["bezirke"][parent_bezirk]["municipalities"][loc_id] = votes_entry

    ############################################################################
    # 3B. Build mandates_data[year]
    ############################################################################
    mandates_data[str(year)] = {}

    all_mandates = mandate_df.collect()

    # Potential party columns in mandates


    for row in all_mandates:
        loc_id = str(row["Nr."])
        name = row["Name"]
        if isinstance(name, str) and "Ober�sterreich" in name:
            name = name.replace("Ober�sterreich", "Oberösterreich")
            
        # Match the keys from your old script exactly:
        valid_votes = row["gültige Stimmen"] if "gültige Stimmen" in mandate_df.columns else None
        wahlzahl = row["Wahlzahl"] if "Wahlzahl" in mandate_df.columns else None
        total_mandates = row["zu vergebende Mandate"] if "zu vergebende Mandate" in mandate_df.columns else None

        # Build dictionary of party mandates
        pm_dict = {}
        party_cols_for_mandates = [col for col in mandate_df.columns if col not in {"Nr.", "Name", "gültige Stimmen", "ungültige", "Wahlzahl", "zu vergebende Mandate"}]
        
        for pc in party_cols_for_mandates:
            if pc in mandate_df.columns:
                pm_dict[pc] = row[pc]

        # Final entry
        mandate_entry = {
            "name": name,
            "valid_votes": valid_votes,
            "wahlzahl": wahlzahl,
            "total_mandates": total_mandates,
            "party_mandates": pm_dict
        }

        mandates_data[str(year)][loc_id] = mandate_entry

################################################################################
# 4. Write the Final JSON Files (Elections)
################################################################################
votes_json_path = "data/election_results_new.json"
mandates_json_path = "data/mandates_new.json"

with open(votes_json_path, "w", encoding="utf-8") as f:
    json.dump(election_data, f, indent=4, ensure_ascii=False)
print(f"Votes JSON saved to: {votes_json_path}")

with open(mandates_json_path, "w", encoding="utf-8") as f:
    json.dump(mandates_data, f, indent=4, ensure_ascii=False)
print(f"Mandates JSON saved to: {mandates_json_path}")

################################################################################
# 5. Done
################################################################################
print("All done!")
