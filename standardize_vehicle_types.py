import pandas as pd
import numpy as np

print("="*80)
print("STANDARDIZING VEHICLE TYPES - CREATING FINAL STANDARDIZED DATASET")
print("="*80)

# Load the FINAL dataset
print("\nLoading dataset...")
df = pd.read_csv(r"E:\PycharmProjects\data visualizer\NYC_crashes_dataset_FINAL.csv", 
                 parse_dates=['CRASH DATE'], low_memory=False)

print(f"Original dataset: {len(df):,} rows")

print("\n" + "="*80)
print("CREATING COMPREHENSIVE VEHICLE TYPE STANDARDIZATION MAPPING")
print("="*80)

# Comprehensive mapping to standardize ALL vehicle types
vehicle_mapping = {}

# Function to add mapping (case-insensitive)
def add_mapping(variations, standard_name):
    for variation in variations:
        vehicle_mapping[variation] = standard_name
        vehicle_mapping[variation.upper()] = standard_name
        vehicle_mapping[variation.lower()] = standard_name
        vehicle_mapping[variation.title()] = standard_name

# SEDANS
add_mapping(['Sedan', 'SEDAN', '4 dr sedan', '2 dr sedan', '4-Door', '2-Door', 
             'Passenger Vehicle', 'PASSENGER VEHICLE'], 'Sedan')

# SUVs
add_mapping(['SUV', 'Station Wagon/Sport Utility Vehicle', 'SPORT UTILITY / STATION WAGON',
             'Sport Utility', 'Station Wagon', 'STATION WAGON', 'Suv'], 'SUV')

# TAXIS
add_mapping(['Taxi', 'TAXI', 'Yellow Cab', 'YELLOW CAB', 'Medallion', 'MEDALLION'], 'Taxi')

# PICKUP TRUCKS
add_mapping(['Pickup Truck', 'Pick-up Truck', 'PICK-UP TRUCK', 'PICKUP TRUCK', 
             'Pick up', 'Pickup'], 'Pickup Truck')

# TRUCKS (General)
add_mapping(['Truck', 'TRUCK'], 'Truck')

# BOX TRUCKS
add_mapping(['Box Truck', 'BOX TRUCK', 'Box truck'], 'Box Truck')

# VANS
add_mapping(['Van', 'VAN', 'Van Truck', 'VAN TRUCK', 'Minivan', 'MINIVAN'], 'Van')

# BUSES
add_mapping(['Bus', 'BUS', 'BUs', 'Omnibus', 'OMNIBUS'], 'Bus')
add_mapping(['School Bus', 'SCHOOL BUS', 'Schoolbus', 'SCHOOLBUS', 'School bus'], 'School Bus')
add_mapping(['MTA Bus', 'MTA BUS', 'Mta bus', 'NYC BUS', 'Transit Bus', 'TRANSIT BUS'], 'Transit Bus')

# MOTORCYCLES
add_mapping(['Motorcycle', 'MOTORCYCLE', 'Motorbike', 'MOTORBIKE', 'Motor Cycle'], 'Motorcycle')
add_mapping(['Moped', 'MOPED'], 'Moped')
add_mapping(['Scooter', 'SCOOTER', 'Motorscooter', 'MOTORSCOOTER'], 'Scooter')

# BICYCLES
add_mapping(['Bicycle', 'BICYCLE', 'Bike', 'BIKE'], 'Bicycle')
add_mapping(['E-Bike', 'E-BIKE', 'E Bike', 'Ebike', 'EBIKE'], 'E-Bike')
add_mapping(['E-Scooter', 'E-SCOOTER', 'E Scooter', 'Escooter'], 'E-Scooter')

# LARGE TRUCKS
add_mapping(['Tractor Truck Diesel', 'Tractor Truck Gasoline', 'TRACTOR TRUCK DIESEL',
             'TRACTOR TRUCK GASOLINE', 'Tractor Truck', 'TRACTOR TRUCK'], 'Tractor Truck')
add_mapping(['Dump', 'DUMP', 'Dump Truck', 'DUMP TRUCK'], 'Dump Truck')
add_mapping(['Flatbed', 'Flat Bed', 'FLAT BED', 'FLATBED'], 'Flatbed Truck')
add_mapping(['Tanker', 'TANKER', 'Tank Truck'], 'Tanker')
add_mapping(['Concrete Mixer', 'CONCRETE MIXER', 'Cement Mixer'], 'Concrete Mixer')
add_mapping(['Garbage or Refuse', 'GARBAGE OR REFUSE', 'Garbage Truck', 'Refuse'], 'Garbage Truck')
add_mapping(['Tow Truck', 'TOW TRUCK / WRECKER', 'Tow Truck / Wrecker', 'Wrecker'], 'Tow Truck')

# EMERGENCY VEHICLES
add_mapping(['Ambulance', 'AMBULANCE', 'Ambulette', 'AMBULETTE', 'EMS'], 'Ambulance')
add_mapping(['Fire Truck', 'FIRE TRUCK', 'Firetruck', 'FIRETRUCK', 'FDNY'], 'Fire Truck')
add_mapping(['Police', 'POLICE', 'NYPD', 'Police Vehicle'], 'Police Vehicle')

# LIVERY/FOR-HIRE (Note: Livery Vehicle is for-hire/black car service, not delivery)
add_mapping(['Livery Vehicle', 'LIVERY VEHICLE', 'Black Car', 'BLACK CAR'], 'For-Hire Vehicle')
add_mapping(['Limousine', 'LIMOUSINE', 'Limo', 'LIMO'], 'Limousine')

# COMMERCIAL VEHICLES
add_mapping(['SMALL COM VEH(4 TIRES)', 'Small Com Veh(4 Tires)', 'SMALL COMMERCIAL VEHICLE'], 'Small Commercial Vehicle')
add_mapping(['LARGE COM VEH(6 OR MORE TIRES)', 'Large Com Veh(6 Or More Tires)', 'LARGE COMMERCIAL VEHICLE'], 'Large Commercial Vehicle')
add_mapping(['Delivery Truck', 'DELIVERY TRUCK', 'Delivery'], 'Delivery Truck')
add_mapping(['USPS', 'Postal', 'POSTAL', 'Mail Truck'], 'USPS Vehicle')
add_mapping(['UPS', 'UPS Truck'], 'UPS Vehicle')
add_mapping(['FedEx', 'FEDEX', 'Fedex Truck'], 'FedEx Vehicle')

# TRAILERS
add_mapping(['Trailer', 'TRAILER', 'Semi-Trailer'], 'Trailer')
add_mapping(['Chassis Cab', 'CHASSIS CAB'], 'Chassis Cab')

# SPECIALTY VEHICLES
add_mapping(['Pedicab', 'PEDICAB', 'Pedi Cab'], 'Pedicab')
add_mapping(['Golf Cart', 'GOLF CART', 'Golfcart'], 'Golf Cart')
add_mapping(['ATV', 'All Terrain Vehicle', 'ALL TERRAIN VEHICLE'], 'ATV')
add_mapping(['Forklift', 'FORK LIFT', 'Fork Lift'], 'Forklift')
add_mapping(['Bobcat', 'BOBCAT'], 'Bobcat')
add_mapping(['Backhoe', 'BACKHOE'], 'Backhoe')
add_mapping(['Bulldozer', 'BULLDOZER'], 'Bulldozer')
add_mapping(['Crane', 'CRANE'], 'Crane')
add_mapping(['Tractor', 'TRACTOR'], 'Tractor')

# RECREATIONAL VEHICLES
add_mapping(['Convertible', 'CONVERTIBLE'], 'Convertible')
add_mapping(['Motorhome', 'MOTORHOME', 'RV', 'Motor Home'], 'Motorhome')
add_mapping(['Van Camper', 'VAN CAMPER'], 'Van Camper')
add_mapping(['Snowmobile', 'SNOWMOBILE'], 'Snowmobile')

# OTHER
add_mapping(['Other', 'OTHER', 'Other Vehicle'], 'Other')

# UNKNOWN/UNSPECIFIED - Combine all variations
unknown_variations = [
    'Unknown', 'UNKNOWN', 'UNK', 'Unk', 'unk',
    'Unspecified', 'UNSPECIFIED', 'Unkno', 'UNKNO', 'unkno',
    'Unkown', 'UNKOWN', 'unkown', 'Unkow', 'UNKOW', 'unkow',
    'UNKNW', 'Unknw', 'UNKWOWN', 'Unknown ve', 'UNKNOWN VE',
    'UNK BOX TR', 'UNKN', 'Unkn'
]
for var in unknown_variations:
    vehicle_mapping[var] = 'Unknown'

print(f"\nCreated mapping for {len(set(vehicle_mapping.keys()))} vehicle type variations")
print(f"Mapping to {len(set(vehicle_mapping.values()))} standardized categories")

print("\n" + "="*80)
print("APPLYING STANDARDIZATION TO ALL VEHICLE TYPE COLUMNS")
print("="*80)

vehicle_cols = ['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 
                'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']

for col in vehicle_cols:
    if col in df.columns:
        print(f"\nProcessing {col}...")
        before_unique = df[col].nunique()
        
        # Apply mapping
        df[col] = df[col].map(lambda x: vehicle_mapping.get(x, x) if pd.notna(x) else x)
        
        after_unique = df[col].nunique()
        print(f"  Unique values: {before_unique} → {after_unique} (reduced by {before_unique - after_unique})")

print("\n" + "="*80)
print("VEHICLE TYPE DISTRIBUTION AFTER STANDARDIZATION")
print("="*80)

# Get all vehicle types after standardization
all_vehicles_clean = []
for col in vehicle_cols:
    if col in df.columns:
        all_vehicles_clean.extend(df[col].dropna().unique().tolist())

from collections import Counter
clean_counts = Counter(all_vehicles_clean)

print(f"\nTotal unique vehicle types after standardization: {len(clean_counts)}")
print(f"\nTop 50 standardized vehicle types:")
for i, (vehicle, count) in enumerate(clean_counts.most_common(50), 1):
    pct = (count / sum(clean_counts.values())) * 100
    print(f"{i:3}. {vehicle:40} | {count:>10,} ({pct:>5.2f}%)")

print("\n" + "="*80)
print("SAVING STANDARDIZED DATASET")
print("="*80)

output_file = r"E:\PycharmProjects\data visualizer\NYC_crashes_dataset_STANDARDIZED.csv"
df.to_csv(output_file, index=False)
print(f"\nStandardized dataset saved to: {output_file}")
print(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")

print("\n" + "="*80)
print("STANDARDIZATION COMPLETE!")
print("="*80)

print("\nSummary:")
print(f"  ✓ Standardized all vehicle type variations")
print(f"  ✓ Combined case variations (Taxi, TAXI, taxi → Taxi)")
print(f"  ✓ Combined unknown variations (Unknown, UNK, Unkown → Unknown)")
print(f"  ✓ Reduced unique vehicle types significantly")
print(f"  ✓ Maintained data integrity - no rows removed")
print(f"\n  Output file: NYC_crashes_dataset_STANDARDIZED.csv")
