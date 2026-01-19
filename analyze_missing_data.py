import pandas as pd
import numpy as np

print("="*80)
print("ANALYZING NYC CRASHES DATASET - MISSING DATA ANALYSIS")
print("="*80)

# Load the categorized dataset
print("\nLoading categorized dataset...")
df = pd.read_csv(r"E:\PycharmProjects\data visualizer\NYC_crashes_dataset_categorized.csv", 
                 parse_dates=['CRASH DATE'], low_memory=False)

print(f"\nDataset shape: {df.shape}")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "="*80)
print("COLUMN OVERVIEW")
print("="*80)
print("\nColumns in dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "="*80)
print("MISSING DATA ANALYSIS BY COLUMN")
print("="*80)

# Calculate missing data statistics for each column
missing_stats = []
for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    non_missing = len(df) - missing_count
    missing_stats.append({
        'Column': col,
        'Missing Count': missing_count,
        'Missing %': missing_pct,
        'Non-Missing': non_missing,
        'Data Type': str(df[col].dtype)
    })

missing_df = pd.DataFrame(missing_stats).sort_values('Missing %', ascending=False)

print("\nColumns sorted by missing data percentage:")
print("-" * 80)
for _, row in missing_df.iterrows():
    print(f"{row['Column']:45} | Missing: {row['Missing Count']:>10,} ({row['Missing %']:>6.2f}%) | Type: {row['Data Type']}")

print("\n" + "="*80)
print("CRITICAL COLUMNS ANALYSIS")
print("="*80)

critical_cols = ['CRASH DATE', 'LATITUDE', 'LONGITUDE', 'BOROUGH', 
                 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED']

print("\nMissing data in critical columns:")
for col in critical_cols:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"  {col:35} | {missing_count:>10,} ({missing_pct:>6.2f}%)")

print("\n" + "="*80)
print("ROW-LEVEL MISSING DATA ANALYSIS")
print("="*80)

# Calculate missing data per row
df['missing_count'] = df.isna().sum(axis=1)
df['missing_pct'] = (df['missing_count'] / len(df.columns)) * 100

print("\nDistribution of missing values per row:")
print(df['missing_count'].describe())

print("\n\nRows by missing data percentage:")
missing_ranges = [
    (0, 10, "0-10% missing"),
    (10, 25, "10-25% missing"),
    (25, 50, "25-50% missing"),
    (50, 75, "50-75% missing"),
    (75, 100, "75-100% missing")
]

for min_pct, max_pct, label in missing_ranges:
    count = ((df['missing_pct'] >= min_pct) & (df['missing_pct'] < max_pct)).sum()
    pct = (count / len(df)) * 100
    print(f"  {label:20} | {count:>10,} rows ({pct:>6.2f}%)")

print("\n" + "="*80)
print("ROWS WITH HIGH MISSING DATA (>50%)")
print("="*80)

high_missing = df[df['missing_pct'] > 50]
print(f"\nTotal rows with >50% missing data: {len(high_missing):,} ({len(high_missing)/len(df)*100:.2f}%)")

if len(high_missing) > 0:
    print("\nSample of rows with high missing data:")
    print(high_missing[['CRASH DATE', 'BOROUGH', 'LATITUDE', 'LONGITUDE', 
                        'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 
                        'missing_count', 'missing_pct']].head(10))

print("\n" + "="*80)
print("ROWS WITH MISSING CRITICAL DATA")
print("="*80)

# Check rows missing critical location data
missing_location = df[(df['LATITUDE'].isna()) | (df['LONGITUDE'].isna())]
print(f"\nRows missing LATITUDE or LONGITUDE: {len(missing_location):,} ({len(missing_location)/len(df)*100:.2f}%)")

# Check rows missing date
missing_date = df[df['CRASH DATE'].isna()]
print(f"Rows missing CRASH DATE: {len(missing_date):,} ({len(missing_date)/len(df)*100:.2f}%)")

# Check rows missing casualty data
missing_casualties = df[(df['NUMBER OF PERSONS INJURED'].isna()) & (df['NUMBER OF PERSONS KILLED'].isna())]
print(f"Rows missing both INJURED and KILLED counts: {len(missing_casualties):,} ({len(missing_casualties)/len(df)*100:.2f}%)")

# Check rows with invalid location (0,0)
invalid_location = df[((df['LATITUDE'] == 0) & (df['LONGITUDE'] == 0)) | 
                      ((df['LATITUDE'].notna()) & (df['LONGITUDE'].notna()) & 
                       ((df['LATITUDE'] < 40.4) | (df['LATITUDE'] > 41.0) | 
                        (df['LONGITUDE'] < -74.3) | (df['LONGITUDE'] > -73.6)))]
print(f"Rows with invalid location coordinates: {len(invalid_location):,} ({len(invalid_location)/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("VEHICLE TYPE ANALYSIS")
print("="*80)

vehicle_cols = [col for col in df.columns if 'VEHICLE TYPE' in col.upper()]
print(f"\nVehicle type columns found: {len(vehicle_cols)}")
for col in vehicle_cols:
    missing = df[col].isna().sum()
    missing_pct = (missing / len(df)) * 100
    unique_values = df[col].nunique()
    print(f"  {col:45} | Missing: {missing:>10,} ({missing_pct:>6.2f}%) | Unique: {unique_values:>5}")

# Check rows with no vehicle type at all
if vehicle_cols:
    no_vehicle = df[df[vehicle_cols].isna().all(axis=1)]
    print(f"\nRows with NO vehicle type data: {len(no_vehicle):,} ({len(no_vehicle)/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("CONTRIBUTING FACTOR ANALYSIS")
print("="*80)

factor_cols = [col for col in df.columns if 'CONTRIBUTING FACTOR' in col.upper()]
print(f"\nContributing factor columns found: {len(factor_cols)}")
for col in factor_cols:
    missing = df[col].isna().sum()
    missing_pct = (missing / len(df)) * 100
    unique_values = df[col].nunique()
    print(f"  {col:45} | Missing: {missing:>10,} ({missing_pct:>6.2f}%) | Unique: {unique_values:>5}")

# Check rows with no contributing factor at all
if factor_cols:
    no_factor = df[df[factor_cols].isna().all(axis=1)]
    print(f"\nRows with NO contributing factor data: {len(no_factor):,} ({len(no_factor)/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR DATA REMOVAL")
print("="*80)

recommendations = []

# Recommendation 1: Remove rows with missing critical location data
if len(missing_location) > 0:
    recommendations.append({
        'Category': 'Missing Location',
        'Rows': len(missing_location),
        'Percentage': f"{len(missing_location)/len(df)*100:.2f}%",
        'Reason': 'Cannot perform geographic analysis without coordinates',
        'Priority': 'HIGH'
    })

# Recommendation 2: Remove rows with invalid location
if len(invalid_location) > 0:
    recommendations.append({
        'Category': 'Invalid Location',
        'Rows': len(invalid_location),
        'Percentage': f"{len(invalid_location)/len(df)*100:.2f}%",
        'Reason': 'Coordinates outside NYC boundaries or (0,0)',
        'Priority': 'HIGH'
    })

# Recommendation 3: Remove rows with missing date
if len(missing_date) > 0:
    recommendations.append({
        'Category': 'Missing Date',
        'Rows': len(missing_date),
        'Percentage': f"{len(missing_date)/len(df)*100:.2f}%",
        'Reason': 'Cannot perform temporal analysis without crash date',
        'Priority': 'HIGH'
    })

# Recommendation 4: Rows with >50% missing data
if len(high_missing) > 0:
    recommendations.append({
        'Category': 'High Missing Data (>50%)',
        'Rows': len(high_missing),
        'Percentage': f"{len(high_missing)/len(df)*100:.2f}%",
        'Reason': 'Insufficient data for meaningful analysis',
        'Priority': 'MEDIUM'
    })

# Recommendation 5: Rows with no vehicle or factor data
if vehicle_cols and factor_cols:
    no_vehicle_or_factor = df[df[vehicle_cols + factor_cols].isna().all(axis=1)]
    if len(no_vehicle_or_factor) > 0:
        recommendations.append({
            'Category': 'No Vehicle/Factor Data',
            'Rows': len(no_vehicle_or_factor),
            'Percentage': f"{len(no_vehicle_or_factor)/len(df)*100:.2f}%",
            'Reason': 'Missing all vehicle types and contributing factors',
            'Priority': 'LOW'
        })

print("\nRecommended rows to remove:")
print("-" * 80)
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['Category']} [{rec['Priority']} PRIORITY]")
    print(f"   Rows to remove: {rec['Rows']:,} ({rec['Percentage']})")
    print(f"   Reason: {rec['Reason']}")

# Calculate combined removal
print("\n" + "="*80)
print("COMBINED REMOVAL IMPACT")
print("="*80)

# Create mask for all recommended removals
removal_mask = (df['LATITUDE'].isna()) | (df['LONGITUDE'].isna()) | \
               (df['CRASH DATE'].isna()) | \
               ((df['LATITUDE'] == 0) & (df['LONGITUDE'] == 0)) | \
               ((df['LATITUDE'].notna()) & (df['LONGITUDE'].notna()) & 
                ((df['LATITUDE'] < 40.4) | (df['LATITUDE'] > 41.0) | 
                 (df['LONGITUDE'] < -74.3) | (df['LONGITUDE'] > -73.6)))

rows_to_remove = removal_mask.sum()
rows_to_keep = len(df) - rows_to_remove

print(f"\nOriginal dataset: {len(df):,} rows")
print(f"Rows to remove: {rows_to_remove:,} ({rows_to_remove/len(df)*100:.2f}%)")
print(f"Rows to keep: {rows_to_keep:,} ({rows_to_keep/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Drop temporary columns
df = df.drop(['missing_count', 'missing_pct'], axis=1)
