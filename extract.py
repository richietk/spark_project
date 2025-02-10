from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when, lit, expr
import pyspark.sql.functions as F
import os
from unidecode import unidecode  # Install with: pip install unidecode

# -------------------------
# Helper Functions
# -------------------------

def standardize_municipality(col_name):
    """Convert a municipality name to lowercase, trim whitespace, and replace spaces with dashes."""
    return F.regexp_replace(F.lower(F.trim(F.col(col_name))), " ", "-")

# -------------------------
# Main Script
# -------------------------

# Initialize Spark
spark = (
    SparkSession.builder
    .appName("MunicipalSpendingAndElectionAnalysis")
    .getOrCreate()
)

# --- Read & Transform Municipal Spending Data ---
spending_csv = "data/Bildungsausgaben_Gemeinden_Oberösterreich_data_2007_bis_2019.csv"
df_spending = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("encoding", "UTF-8")
    .csv(spending_csv)
)

df_spending = (df_spending
    .withColumnRenamed("Gemeinde", "Municipality")
    .withColumnRenamed("Year", "Year")
    .withColumnRenamed("Abschnitt", "Category")
    .withColumnRenamed("Betrag in Euro", "Spending")
)

df_spending = df_spending.withColumn(
    "Category",
    regexp_replace(col("Category"), "FĂ¶rderung", "Förderung")
)
df_spending = df_spending.withColumn(
    "Category",
    regexp_replace(col("Category"), "Sport und auĂźerschulische Leibeserziehung",
                   "Sport und außerschulische Leibeserziehung")
)

df_spending = df_spending.withColumn("Year", col("Year").cast("int"))
df_spending = df_spending.withColumn("Spending", col("Spending").cast("float"))

df_spending_agg = (
    df_spending
    .groupBy("Municipality", "Year")
    .agg(F.sum("Spending").alias("Spending_Summe"))
)

df_spending_agg = df_spending_agg.withColumn("Municipality_Lowercase", standardize_municipality("Municipality"))

# --- Read & Transform Election Data ---
election_yrs = [2008, 2013, 2017, 2019, 2024]
all_election_df = spark.createDataFrame([], schema="Year INT, Municipality_ID STRING, Municipality_Name STRING, Wahlberechtigte INT, abgegebene_Stimmen INT, gueltige_Stimmen INT, Wahlbeteiligung STRING, Winning_Party STRING, Municipality_Lowercase STRING")

for year in election_yrs:
    votes_file = f"data/OÖ_{year}_Stimmen.csv"

    if not os.path.exists(votes_file):
        print(f"[WARNING] Missing CSV(s) for year {year} -> Skipping.")
        continue

    votes_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("encoding", "latin1")
        .csv(votes_file)
    )

    # Normalize column names to remove special characters and standardize them
    renamed_columns = {c: unidecode(c.strip()).replace(" ", "_") for c in votes_df.columns}
    votes_df = votes_df.selectExpr(*[f"`{old}` as `{new}`" for old, new in renamed_columns.items()])

    print("Columns found in votes_df:", votes_df.columns)  # Debugging

    # Identify party columns dynamically
    party_columns = [c for c in votes_df.columns if c not in {
        "Nr.", "Name", "Wahlberechtigte", "abgegeb._Stimmen", "gultige", "Wahlbet.", "ungultige"
    }]

    print("Party columns:", party_columns)  # Debugging

    # Ensure necessary columns exist
    votes_df = votes_df.withColumnRenamed("Nr.", "Municipality_ID")
    votes_df = votes_df.withColumnRenamed("Name", "Municipality_Name")
    votes_df = votes_df.withColumnRenamed("Wahlberechtigte", "Wahlberechtigte")
    votes_df = votes_df.withColumnRenamed("abgegeb._Stimmen", "abgegebene_Stimmen")
    votes_df = votes_df.withColumnRenamed("gultige", "gueltige_Stimmen")
    votes_df = votes_df.withColumnRenamed("Wahlbet.", "Wahlbeteiligung")

    # Cast party columns to int
    for party_col in party_columns:
        print(f"Processing column: {party_col}")  # Debugging
        votes_df = votes_df.withColumn(f"votes_{party_col}", col(party_col).cast("int"))

    # Determine winning party
    vote_cols = [F.coalesce(col(f"votes_{party_col}"), lit(0)) for party_col in party_columns]
    votes_df = votes_df.withColumn("max_votes", F.greatest(*vote_cols))

    winning_party_cases = [when(col("max_votes") == col(f"votes_{party_col}"), lit(party_col)) for party_col in party_columns]
    votes_df = votes_df.withColumn("Winning_Party", F.coalesce(*winning_party_cases))

    # Drop temporary columns
    for party_col in party_columns:
        votes_df = votes_df.drop(f"votes_{party_col}")

    votes_df = votes_df.drop("max_votes")

    # Add Year & Municipality Lowercase for merging
    votes_df = votes_df.withColumn("Year", lit(year))
    votes_df = votes_df.withColumn("Municipality_Lowercase", standardize_municipality("Municipality_Name"))
    votes_df = votes_df.filter(col("Municipality_ID").rlike("^[0-9]+$"))

    # Add to main dataset
    all_election_df = all_election_df.unionByName(votes_df.select(
        "Year", "Municipality_ID", "Municipality_Name", "Wahlberechtigte",
        "abgegebene_Stimmen", "gueltige_Stimmen", "Wahlbeteiligung", "Winning_Party", "Municipality_Lowercase"
    ))

# --- Merging and Output ---
merged_df = all_election_df.join(df_spending_agg, on=["Municipality_Lowercase", "Year"], how="inner").drop("Municipality")
merged_df.coalesce(1).write.mode("overwrite").option("header", "true").option("encoding", "latin1").csv("data/merged_data.csv")
print("Merged data sample:")
merged_df.show(5, truncate=False)

print("All done!")
