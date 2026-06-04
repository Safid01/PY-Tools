import pandas as pd

# Ask for file path
path = input("Enter the path to your Excel file: ").strip().strip('"')

# Load the file
df = pd.read_excel(path)

input(f"Loaded {len(df)} rows. Press Enter to shuffle...")

# Extract year from created_date column
df["_year"] = pd.to_datetime(df["created_date"]).dt.year

# Shuffle within each year, then sort by year
df_shuffled = (
    df.groupby("_year", group_keys=False)
    .apply(lambda x: x.sample(frac=1))
    .sort_values("_year")
    .drop(columns="_year")
    .reset_index(drop=True)
)

# Save as a new file
output_path = path.replace(".xlsx", "_shuffled.xlsx")
df_shuffled.to_excel(output_path, index=False)

print(f"Done! Saved to: {output_path}")