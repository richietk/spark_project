import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# --- Helper Functions ---
def standardize_municipality_name(series):
    return series.str.lower().str.strip().str.replace(" ", "-", regex=True)

def clean_numeric_column(series):
    return (
        series.astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.replace('[^\d.]', '', regex=True)
        .astype(float)
    )

# --- Load and Process Election Data ---
with open("data/election_results.json", "r", encoding="utf-8") as f:
    election_data = json.load(f)

records = []
for year, data in election_data.items():
    for wahlkreis in data.get("wahlkreise", {}).values():
        for bezirk in wahlkreis.get("bezirke", {}).values():
            for municipality_id, details in bezirk.get("municipalities", {}).items():
                votes = details.get("votes", {})
                party_votes = {k: v for k, v in votes.items() if k not in ["Wahlberechtigte", "abgegebene Stimmen", "gültige Stimmen", "Wahlbeteiligung"]}
                winning_party = max(party_votes, key=party_votes.get) if party_votes else None
                records.append({
                    "Year": int(year),
                    "Municipality_ID": municipality_id,
                    "Municipality_Name": details.get("name", None),
                    "Wahlberechtigte": votes.get("Wahlberechtigte"),
                    "abgegebene_Stimmen": votes.get("abgegebene Stimmen"),
                    "gueltige_Stimmen": votes.get("gültige Stimmen"),
                    "Wahlbeteiligung": votes.get("Wahlbeteiligung"),
                    "Winning_Party": winning_party
                })

election_df = pd.DataFrame(records)
election_df["Municipality_Lowercase"] = standardize_municipality_name(election_df["Municipality_Name"])
print("Election data flattened:\n", election_df.head())

# --- Load and Process Spending Data ---
with open("data/municipal_spending.json", "r", encoding="utf-8") as f:
    spending_data = json.load(f)

spending_records = []
for municipality, years in spending_data.items():
    for year, data in years.items():
        spending_records.append({
            "Municipality": municipality,
            "Year": int(year),
            "Spending_Summe": data.get("Summe", None)
        })

spending_df = pd.DataFrame(spending_records)
spending_df["Municipality_Lowercase"] = standardize_municipality_name(spending_df["Municipality"])
print("Spending data sample:\n", spending_df.head())

# --- Merge Election and Spending Data (Yearly Spending) ---
merged_df = pd.merge(
    election_df,
    spending_df,
    left_on=["Municipality_Lowercase", "Year"],
    right_on=["Municipality_Lowercase", "Year"],
    how="inner"
).drop(columns=["Municipality"])
merged_df.to_csv("data/merged_data.csv")
print("Merged data sample (for yearly spending correlation):\n", merged_df.head()) # ADDED PRINT STATEMENT TO CHECK merged_df

# --- Define Election Periods and Process Data ---
election_years = sorted([2008, 2013, 2017, 2019, 2024])
election_periods = [(election_years[i], election_years[i+1]) for i in range(len(election_years)-1)]
period_dataframes = []

for start_year, end_year in election_periods:
    print(f"Processing period {start_year}-{end_year}")
    spending_period = spending_df[(spending_df["Year"] > start_year) & (spending_df["Year"] <= end_year)]
    cumulative_spending = spending_period.groupby("Municipality_Lowercase")["Spending_Summe"].sum().reset_index(name="Cumulative_Spending")
    election_period_data = election_df[election_df["Year"] == end_year].copy()
    period_merged_df = pd.merge(election_period_data, cumulative_spending, on="Municipality_Lowercase", how="inner")
    period_merged_df['Election_Year'] = end_year
    period_dataframes.append(period_merged_df)

filtered_df = pd.concat(period_dataframes)

# --- Clean Numeric Columns ---
filtered_df["Wahlbeteiligung"] = clean_numeric_column(filtered_df["Wahlbeteiligung"])
filtered_df["Cumulative_Spending"] = clean_numeric_column(filtered_df["Cumulative_Spending"])
filtered_df = filtered_df.dropna(subset=["Cumulative_Spending", "Wahlbeteiligung"])

print(f"Filtered dataset contains {len(filtered_df)} records")
print("Filtered data with cumulative spending sample:\n", filtered_df.head())

# --- Visualizations ---
if not filtered_df.empty:
    # 1. Cumulative Spending vs. Voter Turnout (Log Scale, Color-Coded by Year)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x="Cumulative_Spending", y="Wahlbeteiligung", hue="Election_Year", palette="viridis", alpha=0.7, edgecolor="k")
    plt.xscale("log"); plt.xlabel("Cumulative Education Spending (log scale)"); plt.ylabel("Voter Turnout (%)")
    plt.title("Cumulative Education Spending vs. Voter Turnout"); plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Election Year", bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(); plt.show()

    # 2. Trend in Cumulative Spending & Voter Turnout Over Years
    fig, ax1 = plt.subplots(figsize=(10, 6)); ax2 = ax1.twinx()
    trend_df = filtered_df.groupby("Election_Year").agg({"Cumulative_Spending": "mean", "Wahlbeteiligung": "mean"}).reset_index()
    ax1.plot(trend_df["Election_Year"], trend_df["Cumulative_Spending"], label="Cumulative Spending", marker="o", color='b')
    ax2.plot(trend_df["Election_Year"], trend_df["Wahlbeteiligung"], label="Voter Turnout (%)", marker="o", color='r')
    ax1.set_xlabel("Election Year"); ax1.set_ylabel("Cumulative Spending", color='b'); ax2.set_ylabel("Voter Turnout (%)", color='r')
    plt.title("Trend in Cumulative Spending & Voter Turnout"); ax1.grid(True); plt.show()

    # 3. Correlation: Cumulative Spending vs. Voter Turnout (Color-Coded)
    plt.figure(figsize=(10, 6))
    sns.regplot(data=filtered_df, x="Cumulative_Spending", y="Wahlbeteiligung", scatter_kws={"alpha": 0.6, "edgecolor": "k"}, line_kws={"color": "red"})
    sns.scatterplot(data=filtered_df, x="Cumulative_Spending", y="Wahlbeteiligung", hue="Election_Year", palette="viridis", alpha=0.7, edgecolor="k", legend=False)
    plt.xscale("log"); plt.xlabel("Cumulative Education Spending"); plt.ylabel("Voter Turnout (%)")
    plt.title("Correlation: Cumulative Spending vs. Voter Turnout"); plt.legend(title="Election Year", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.show()
else:
    print("No data available for analysis after filtering")

# --- Correlation Calculation ---
correlation_cumulative = filtered_df[["Cumulative_Spending", "Wahlbeteiligung"]].corr().iloc[0, 1]
print(f"Cumulative Spending Correlation: {correlation_cumulative:.4f}")