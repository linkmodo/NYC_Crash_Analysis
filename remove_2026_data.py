import pandas as pd
import numpy as np

print("="*80)
print("REMOVING 2026 DATA (INCOMPLETE YEAR)")
print("="*80)

# Load the STANDARDIZED dataset
print("\nLoading standardized dataset...")
df = pd.read_csv(r"E:\PycharmProjects\data visualizer\NYC_crashes_dataset_STANDARDIZED.csv", 
                 parse_dates=['CRASH DATE'], low_memory=False)

print(f"Original dataset: {len(df):,} rows")

# Check for 2026 data
df['YEAR'] = df['CRASH DATE'].dt.year
year_counts = df['YEAR'].value_counts().sort_index()

print("\n" + "="*80)
print("YEAR DISTRIBUTION")
print("="*80)

print("\nRecords by year:")
for year, count in year_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {year}: {count:>10,} ({pct:>5.2f}%)")

# Count 2026 records
records_2026 = (df['YEAR'] == 2026).sum()
print(f"\n2026 records to remove: {records_2026:,} ({records_2026/len(df)*100:.2f}%)")

# Remove 2026 data
print("\n" + "="*80)
print("REMOVING 2026 DATA")
print("="*80)

df_filtered = df[df['YEAR'] < 2026].copy()
removed = len(df) - len(df_filtered)

print(f"\nRecords removed: {removed:,}")
print(f"Records remaining: {len(df_filtered):,}")

# Drop the temporary YEAR column
df_filtered = df_filtered.drop('YEAR', axis=1)

# Verify year distribution after removal
df_filtered['YEAR'] = df_filtered['CRASH DATE'].dt.year
year_counts_after = df_filtered['YEAR'].value_counts().sort_index()

print("\n" + "="*80)
print("YEAR DISTRIBUTION AFTER REMOVAL")
print("="*80)

print("\nRecords by year (after removal):")
for year, count in year_counts_after.items():
    pct = (count / len(df_filtered)) * 100
    print(f"  {year}: {count:>10,} ({pct:>5.2f}%)")

# Drop the temporary YEAR column again
df_filtered = df_filtered.drop('YEAR', axis=1)

print("\n" + "="*80)
print("SAVING UPDATED DATASET")
print("="*80)

output_file = r"E:\PycharmProjects\data visualizer\NYC_crashes_dataset_STANDARDIZED.csv"
df_filtered.to_csv(output_file, index=False)
print(f"\nUpdated dataset saved to: {output_file}")
print(f"Final dataset: {len(df_filtered):,} rows, {len(df_filtered.columns)} columns")

print("\n" + "="*80)
print("2026 DATA REMOVAL COMPLETE!")
print("="*80)

print("\nSummary:")
print(f"  ✓ Removed {removed:,} records from 2026 (incomplete year)")
print(f"  ✓ Dataset now contains complete years only")
print(f"  ✓ Final dataset: {len(df_filtered):,} rows")
print(f"\n  Output file: NYC_crashes_dataset_STANDARDIZED.csv")
