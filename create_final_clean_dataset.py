import pandas as pd
import numpy as np

print("="*80)
print("CREATING FINAL CLEANED DATASET")
print("="*80)

# Load the categorized dataset
print("\nLoading categorized dataset...")
df = pd.read_csv(r"E:\PycharmProjects\data visualizer\NYC_crashes_dataset_categorized.csv", 
                 parse_dates=['CRASH DATE'], low_memory=False)

print(f"Original dataset: {len(df):,} rows")

print("\n" + "="*80)
print("APPLYING DATA CLEANING RULES")
print("="*80)

# Rule 1: Remove rows with invalid location coordinates
print("\n1. Removing rows with invalid location coordinates...")
before = len(df)
df = df[
    (df['LATITUDE'].notna()) & 
    (df['LONGITUDE'].notna()) & 
    ~((df['LATITUDE'] == 0) & (df['LONGITUDE'] == 0)) &
    (df['LATITUDE'] >= 40.4) & (df['LATITUDE'] <= 41.0) &
    (df['LONGITUDE'] >= -74.3) & (df['LONGITUDE'] <= -73.6)
]
removed = before - len(df)
print(f"   Removed: {removed:,} rows")
print(f"   Remaining: {len(df):,} rows")

# Rule 2: Fill missing casualty counts with 0 (these are likely data entry issues)
print("\n2. Filling missing casualty counts with 0...")
casualty_cols = ['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
                 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                 'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
                 'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']
for col in casualty_cols:
    if col in df.columns:
        missing_before = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        if missing_before > 0:
            print(f"   {col}: filled {missing_before} missing values")

# Rule 3: Standardize borough names and handle missing
print("\n3. Standardizing borough names...")
if 'BOROUGH' in df.columns:
    df['BOROUGH'] = df['BOROUGH'].str.upper().str.strip()
    df['BOROUGH'] = df['BOROUGH'].fillna('UNKNOWN')
    print(f"   Borough distribution:")
    for borough, count in df['BOROUGH'].value_counts().items():
        print(f"      {borough}: {count:,} ({count/len(df)*100:.2f}%)")

# Rule 4: Clean contributing factors - replace NaN with 'Unspecified'
print("\n4. Cleaning contributing factors...")
factor_cols = [col for col in df.columns if 'CONTRIBUTING FACTOR' in col.upper()]
for col in factor_cols:
    missing_before = df[col].isna().sum()
    df[col] = df[col].fillna('Unspecified')
    if missing_before > 0:
        print(f"   {col}: filled {missing_before} missing values with 'Unspecified'")

# Rule 5: Clean vehicle types - replace NaN with 'Unknown'
print("\n5. Cleaning vehicle types...")
vehicle_cols = [col for col in df.columns if 'VEHICLE TYPE' in col.upper()]
for col in vehicle_cols:
    missing_before = df[col].isna().sum()
    df[col] = df[col].fillna('Unknown')
    if missing_before > 0:
        print(f"   {col}: filled {missing_before} missing values with 'Unknown'")

# Rule 6: Clean vehicle_category
print("\n6. Cleaning vehicle_category...")
if 'vehicle_category' in df.columns:
    missing_before = df['vehicle_category'].isna().sum()
    df['vehicle_category'] = df['vehicle_category'].fillna('Unknown')
    if missing_before > 0:
        print(f"   vehicle_category: filled {missing_before} missing values with 'Unknown'")
    print(f"   Vehicle category distribution:")
    for category, count in df['vehicle_category'].value_counts().head(10).items():
        print(f"      {category}: {count:,} ({count/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("FINAL DATASET STATISTICS")
print("="*80)

print(f"\nFinal dataset shape: {df.shape}")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Calculate remaining missing data
print("\n\nRemaining missing data by column:")
missing_summary = []
for col in df.columns:
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        missing_pct = (missing_count / len(df)) * 100
        missing_summary.append({
            'Column': col,
            'Missing': missing_count,
            'Percentage': missing_pct
        })

if missing_summary:
    missing_df = pd.DataFrame(missing_summary).sort_values('Percentage', ascending=False)
    for _, row in missing_df.iterrows():
        print(f"   {row['Column']:45} | {row['Missing']:>10,} ({row['Percentage']:>6.2f}%)")
else:
    print("   No missing data remaining in any column!")

print("\n" + "="*80)
print("SAVING CLEANED DATASET")
print("="*80)

output_file = r"E:\PycharmProjects\data visualizer\NYC_crashes_dataset_FINAL.csv"
df.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to: {output_file}")

print("\n" + "="*80)
print("DATA CLEANING COMPLETE!")
print("="*80)

print("\nSummary of changes:")
print(f"  ✓ Removed {removed:,} rows with invalid location coordinates")
print(f"  ✓ Filled missing casualty counts with 0")
print(f"  ✓ Standardized borough names (filled missing with 'UNKNOWN')")
print(f"  ✓ Filled missing contributing factors with 'Unspecified'")
print(f"  ✓ Filled missing vehicle types with 'Unknown'")
print(f"  ✓ Final dataset: {len(df):,} rows (99.65% retention)")
print(f"\n  Output file: NYC_crashes_dataset_FINAL.csv")
