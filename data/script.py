import os
import pandas as pd

# Get all .xlsx files in the current directory
xlsx_files = [f for f in os.listdir() if f.endswith(".xlsx")]

for file in xlsx_files:
    try:
        # Load the Excel file
        df = pd.read_excel(file, sheet_name=None)
        
        # Iterate through all sheets and save them as CSV
        for sheet_name, sheet_df in df.items():
            csv_filename = f"{os.path.splitext(file)[0]}_{sheet_name}.csv"
            sheet_df.to_csv(csv_filename, index=False, sep=",", encoding="latin1")
            print(f"Converted: {file} (Sheet: {sheet_name}) -> {csv_filename}")
    except Exception as e:
        print(f"Error converting {file}: {e}")

print("All conversions completed.")
