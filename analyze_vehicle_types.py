import pandas as pd
import numpy as np
from collections import Counter

print("="*80)
print("ANALYZING VEHICLE TYPES IN FINAL DATASET")
print("="*80)

# Load the FINAL dataset
print("\nLoading dataset...")
df = pd.read_csv(r"E:\PycharmProjects\data visualizer\NYC_crashes_dataset_FINAL.csv", 
                 parse_dates=['CRASH DATE'], low_memory=False)

print(f"Dataset shape: {df.shape}")
print(f"Total records: {len(df):,}")

print("\n" + "="*80)
print("VEHICLE TYPE COLUMNS ANALYSIS")
print("="*80)

vehicle_cols = ['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 
                'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']

# Get all unique vehicle types across all columns
all_vehicles = []
for col in vehicle_cols:
    if col in df.columns:
        all_vehicles.extend(df[col].dropna().unique().tolist())

vehicle_counts = Counter(all_vehicles)
print(f"\nTotal unique vehicle type values: {len(vehicle_counts)}")

print("\n" + "="*80)
print("ALL VEHICLE TYPES (sorted alphabetically)")
print("="*80)

sorted_vehicles = sorted(vehicle_counts.items(), key=lambda x: x[0].lower())
print(f"\nShowing all {len(sorted_vehicles)} unique vehicle types:\n")

for i, (vehicle, count) in enumerate(sorted_vehicles, 1):
    print(f"{i:4}. {vehicle:50} | Count: {count:>10,}")

print("\n" + "="*80)
print("TOP 100 VEHICLE TYPES BY FREQUENCY")
print("="*80)

print("\nMost common vehicle types:\n")
for i, (vehicle, count) in enumerate(vehicle_counts.most_common(100), 1):
    pct = (count / sum(vehicle_counts.values())) * 100
    print(f"{i:3}. {vehicle:50} | {count:>10,} ({pct:>5.2f}%)")

print("\n" + "="*80)
print("IDENTIFYING DUPLICATES AND INCONSISTENCIES")
print("="*80)

# Group similar vehicle types (case-insensitive)
vehicle_groups = {}
for vehicle in vehicle_counts.keys():
    key = vehicle.lower().strip()
    if key not in vehicle_groups:
        vehicle_groups[key] = []
    vehicle_groups[key].append(vehicle)

# Find groups with multiple variations
print("\nVehicle types with case/spacing variations:")
duplicates_found = False
for key, variations in sorted(vehicle_groups.items()):
    if len(variations) > 1:
        duplicates_found = True
        total_count = sum(vehicle_counts[v] for v in variations)
        print(f"\n  '{key}' has {len(variations)} variations (Total: {total_count:,}):")
        for v in sorted(variations):
            print(f"    - '{v}': {vehicle_counts[v]:,}")

if not duplicates_found:
    print("  No case/spacing variations found!")

# Check for common patterns
print("\n" + "="*80)
print("VEHICLE TYPE PATTERNS")
print("="*80)

patterns = {
    'Sedan': [],
    'SUV/Station Wagon': [],
    'Taxi': [],
    'Truck': [],
    'Van': [],
    'Bus': [],
    'Motorcycle/Bike': [],
    'Unknown/Unspecified': [],
    'Other': []
}

for vehicle in vehicle_counts.keys():
    v_lower = vehicle.lower()
    
    if 'sedan' in v_lower:
        patterns['Sedan'].append(vehicle)
    elif 'suv' in v_lower or 'station wagon' in v_lower or 'sport utility' in v_lower:
        patterns['SUV/Station Wagon'].append(vehicle)
    elif 'taxi' in v_lower or 'cab' in v_lower:
        patterns['Taxi'].append(vehicle)
    elif 'truck' in v_lower or 'pickup' in v_lower or 'pick-up' in v_lower:
        patterns['Truck'].append(vehicle)
    elif 'van' in v_lower:
        patterns['Van'].append(vehicle)
    elif 'bus' in v_lower or 'omnibus' in v_lower:
        patterns['Bus'].append(vehicle)
    elif 'motorcycle' in v_lower or 'bike' in v_lower or 'bicycle' in v_lower or 'scooter' in v_lower or 'moped' in v_lower:
        patterns['Motorcycle/Bike'].append(vehicle)
    elif 'unknown' in v_lower or 'unspecified' in v_lower or 'unk' in v_lower:
        patterns['Unknown/Unspecified'].append(vehicle)
    else:
        patterns['Other'].append(vehicle)

print("\nVehicle types grouped by category:\n")
for category, vehicles in patterns.items():
    if vehicles:
        total = sum(vehicle_counts[v] for v in vehicles)
        print(f"\n{category} ({len(vehicles)} types, {total:,} total):")
        for v in sorted(vehicles, key=lambda x: vehicle_counts[x], reverse=True)[:20]:
            print(f"  - {v:50} | {vehicle_counts[v]:>10,}")
        if len(vehicles) > 20:
            print(f"  ... and {len(vehicles) - 20} more")

print("\n" + "="*80)
print("VEHICLE_CATEGORY COLUMN ANALYSIS")
print("="*80)

if 'vehicle_category' in df.columns:
    print("\nExisting vehicle_category distribution:")
    category_counts = df['vehicle_category'].value_counts()
    for category, count in category_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {category:30} | {count:>10,} ({pct:>5.2f}%)")
else:
    print("\nNo vehicle_category column found in dataset.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
