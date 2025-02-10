from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
import pandas as pd
import json
import os

# spark init
spark = SparkSession.builder.appName("MunicipalSpendingAnalysis").getOrCreate()

# load csv
file_path = "data/Bildungsausgaben_Gemeinden_Oberösterreich_data_2007_bis_2019.csv"
df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# rename columns
df = df.withColumnRenamed("Gemeinde", "Municipality") \
       .withColumnRenamed("Year", "Year") \
       .withColumnRenamed("Abschnitt", "Category") \
       .withColumnRenamed("Betrag in Euro", "Spending")

# fix encoding
df = df.withColumn("Category", regexp_replace(col("Category"), "FĂ¶rderung", "Förderung"))
df = df.withColumn("Category", regexp_replace(col("Category"), "Sport und auĂźerschulische Leibeserziehung", "Sport und außerschulische Leibeserziehung"))

# convert year and spending to int/float
df = df.withColumn("Year", col("Year").cast("int"))
df = df.withColumn("Spending", col("Spending").cast("float"))

# restructure json format
nested_data = {}
for _, row in df.toPandas().iterrows():
    municipality = row["Municipality"]
    year = row["Year"]
    
    if municipality not in nested_data:
        nested_data[municipality] = {}
    
    if year not in nested_data[municipality]:
        nested_data[municipality][year] = {
            "Summe": None,  # Placeholder for sum
            "Details": {}
        }
    
    if row["Category"] == "Summe":
        nested_data[municipality][year]["Summe"] = row["Spending"]
    else:
        nested_data[municipality][year]["Details"][row["Category"]] = row["Spending"]

# convert structured data to json
json_output = json.dumps(nested_data, indent=4, ensure_ascii=False)

# save json
output_path = "data/municipal_spending.json"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(json_output)


# election years
election_yrs = [2008, 2013, 2017, 2019, 2024]
base_path = "data/OÖ "

election_data = {}
mandates_data = {}

def get_id_level(loc_id):
    """
    categorize the "level"
    """
    if loc_id.endswith("000"):  # Wahlkreis
        return "wahlkreis"
    elif loc_id.endswith("099"):  # Wahlkarten
        return "wahlkarten"
    elif loc_id.endswith("00"):  # Bezirk
        return "bezirk"
    else:  # Municipality level
        return "municipality"

# process data for each year
for year in election_yrs:
    xlsx_file = f"data/OÖ {year}.xlsx"

    if not os.path.exists(xlsx_file):
        print(f"File not found: {xlsx_file}, skipping year {year}")
        continue

    print(f"Processing election data for {year}...")
    votes_df = pd.read_excel(xlsx_file, sheet_name="Stimmen")
    mandate_df = pd.read_excel(xlsx_file, sheet_name="Mandate")

    votes_df.columns = votes_df.columns.astype(str)
    mandate_df.columns = mandate_df.columns.astype(str)

    party_columns_votes = votes_df.columns[7:]
    party_columns_mandate = mandate_df.columns[5:]

    # initialize the year's data
    election_data[str(year)] = {"name": "Land Oberösterreich", "wahlkreise": {}}
    mandates_data[str(year)] = {}

    # process votes
    for _, row in votes_df.iterrows():
        loc_id = str(row["Nr."])
        name = row["Name"]
        level = get_id_level(loc_id)

        # Create structured vote entry
        vote_entry = {
            "name": name,
            "votes": {
                "Wahlberechtigte": row["Wahlberechtigte"],
                "abgegebene Stimmen": row["abgegeb. Stimmen"],
                "gültige Stimmen": row["gültige"],
                "Wahlbeteiligung": row["Wahlbet."]  # Include voter turnout data
            }
        }

        for party in party_columns_votes:
            vote_entry["votes"][party] = row[party]

        if level == "wahlkreis":
            election_data[str(year)]["wahlkreise"][loc_id] = {
                "name": name,
                "bezirke": {}
            }
        elif level == "bezirk":
            parent_wahlkreis = loc_id[:2] + "000" # e.g. 41600 -> 41000
            if parent_wahlkreis in election_data[str(year)]["wahlkreise"]:
                election_data[str(year)]["wahlkreise"][parent_wahlkreis]["bezirke"][loc_id] = {
                    "name": name,
                    "municipalities": {}
                }
        elif level == "municipality":
            parent_bezirk = loc_id[:3] + "00" # e.g. 41650 -> 41600
            parent_wahlkreis = loc_id[:2] + "000" # e.g. 41650 -> 41000
            if (
                parent_wahlkreis in election_data[str(year)]["wahlkreise"]
                and parent_bezirk in election_data[str(year)]["wahlkreise"][parent_wahlkreis]["bezirke"]
            ):
                election_data[str(year)]["wahlkreise"][parent_wahlkreis]["bezirke"][parent_bezirk]["municipalities"][loc_id] = vote_entry

    # process mandates
    for _, row in mandate_df.iterrows():
        loc_id = str(row["Nr."])
        name = row["Name"]

        mandate_entry = {
            "name": name,
            "valid_votes": row["gültige Stimmen"],
            "wahlzahl": row["Wahlzahl"],
            "total_mandates": row["zu vergebende Mandate"],  # Assuming "Mandate" is the correct column name
            "party_mandates": {party: row[party] for party in party_columns_mandate}
        }

        mandates_data[str(year)][loc_id] = mandate_entry
        
# converting to json
votes_json_path = "data/election_results.json"
mandates_json_path = "data/mandates.json"

with open(votes_json_path, "w", encoding="utf-8") as f:
    json.dump(election_data, f, indent=4, ensure_ascii=False)
with open(mandates_json_path, "w", encoding="utf-8") as f:
    json.dump(mandates_data, f, indent=4, ensure_ascii=False)

print(f"Votes JSON saved to {votes_json_path}")
print(f"Mandates JSON saved to {mandates_json_path}")