import pandas as pd
import numpy as np
from collections import Counter

# Load the original dataset
print("Loading dataset...")
df = pd.read_csv(r"E:\PycharmProjects\data visualizer\NYC_Crashes_dataset.csv", 
                 parse_dates=['CRASH DATE'], low_memory=False)

print(f"Original dataset shape: {df.shape}")
print(f"Total records: {len(df):,}")

# Analyze vehicle types
print("\n" + "="*80)
print("ANALYZING VEHICLE TYPE CATEGORIES")
print("="*80)

vehicle_cols = ['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 
                'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']

# Get all unique vehicle types
all_vehicles = []
for col in vehicle_cols:
    all_vehicles.extend(df[col].dropna().unique().tolist())

vehicle_counts = Counter(all_vehicles)
print(f"\nTotal unique vehicle type values: {len(vehicle_counts)}")
print(f"\nTop 50 vehicle types by frequency:")
for vehicle, count in vehicle_counts.most_common(50):
    print(f"  {vehicle}: {count:,}")

# Define vehicle type mapping to standardize categories
print("\n" + "="*80)
print("CREATING VEHICLE TYPE STANDARDIZATION MAPPING")
print("="*80)

vehicle_mapping = {
    # Sedans
    'SEDAN': 'Sedan',
    'Station Wagon/Sport Utility Vehicle': 'SUV',
    'SPORT UTILITY / STATION WAGON': 'SUV',
    '4 dr sedan': 'Sedan',
    '2 dr sedan': 'Sedan',
    
    # Taxis
    'TAXI': 'Taxi',
    'Taxi': 'Taxi',
    
    # Passenger vehicles
    'Passenger Vehicle': 'Passenger Vehicle',
    'PASSENGER VEHICLE': 'Passenger Vehicle',
    
    # Trucks
    'PICK-UP TRUCK': 'Pickup Truck',
    'Pick-up Truck': 'Pickup Truck',
    'TRUCK': 'Truck',
    'Truck': 'Truck',
    'Box Truck': 'Box Truck',
    'BOX TRUCK': 'Box Truck',
    'Dump': 'Dump Truck',
    'DUMP': 'Dump Truck',
    'Flat Bed': 'Flatbed Truck',
    'FLAT BED': 'Flatbed Truck',
    'Tractor Truck Diesel': 'Tractor Truck',
    'Tractor Truck Gasoline': 'Tractor Truck',
    'TRACTOR TRUCK DIESEL': 'Tractor Truck',
    'TRACTOR TRUCK GASOLINE': 'Tractor Truck',
    'Tow Truck / Wrecker': 'Tow Truck',
    'TOW TRUCK / WRECKER': 'Tow Truck',
    'Garbage or Refuse': 'Garbage Truck',
    'GARBAGE OR REFUSE': 'Garbage Truck',
    'Concrete Mixer': 'Concrete Mixer',
    'CONCRETE MIXER': 'Concrete Mixer',
    'Refrigerated Van': 'Refrigerated Van',
    'REFRIGERATED VAN': 'Refrigerated Van',
    'Carry All': 'Carry All',
    'CARRY ALL': 'Carry All',
    'Stake or Rack': 'Stake Truck',
    'STAKE OR RACK': 'Stake Truck',
    'Tanker': 'Tanker',
    'TANKER': 'Tanker',
    
    # Vans
    'VAN': 'Van',
    'Van': 'Van',
    'VAN TRUCK': 'Van',
    'Van Truck': 'Van',
    
    # Buses
    'BUS': 'Bus',
    'Bus': 'Bus',
    'SCHOOL BUS': 'School Bus',
    'School Bus': 'School Bus',
    'TRANSIT BUS': 'Transit Bus',
    'Transit Bus': 'Transit Bus',
    'OMNIBUS': 'Bus',
    'Omnibus': 'Bus',
    
    # Motorcycles/Scooters
    'MOTORCYCLE': 'Motorcycle',
    'Motorcycle': 'Motorcycle',
    'MOTORBIKE': 'Motorcycle',
    'Motorbike': 'Motorcycle',
    'MOPED': 'Moped',
    'Moped': 'Moped',
    'SCOOTER': 'Scooter',
    'Scooter': 'Scooter',
    
    # Bikes
    'BICYCLE': 'Bicycle',
    'Bicycle': 'Bicycle',
    'BIKE': 'Bicycle',
    'Bike': 'Bicycle',
    'E-Bike': 'E-Bike',
    'E-BIKE': 'E-Bike',
    'E-Scooter': 'E-Scooter',
    'E-SCOOTER': 'E-Scooter',
    
    # Emergency vehicles
    'AMBULANCE': 'Ambulance',
    'Ambulance': 'Ambulance',
    'FIRE TRUCK': 'Fire Truck',
    'Fire Truck': 'Fire Truck',
    'FIRETRUCK': 'Fire Truck',
    'Firetruck': 'Fire Truck',
    'FDNY': 'Fire Truck',
    'NYPD': 'Police Vehicle',
    'Police': 'Police Vehicle',
    'POLICE': 'Police Vehicle',
    
    # Livery/For-hire
    'LIVERY VEHICLE': 'Livery Vehicle',
    'Livery Vehicle': 'Livery Vehicle',
    'Black Car': 'Black Car',
    'BLACK CAR': 'Black Car',
    'Limo': 'Limousine',
    'LIMO': 'Limousine',
    'Limousine': 'Limousine',
    'LIMOUSINE': 'Limousine',
    
    # Other
    'OTHER': 'Other',
    'Other': 'Other',
    'UNKNOWN': 'Unknown',
    'Unknown': 'Unknown',
    'Unspecified': 'Unknown',
    'UNSPECIFIED': 'Unknown',
    
    # Construction/Utility
    'PEDICAB': 'Pedicab',
    'Pedicab': 'Pedicab',
    'FORK LIFT': 'Forklift',
    'Forklift': 'Forklift',
    'BOBCAT': 'Bobcat',
    'Bobcat': 'Bobcat',
    'Backhoe': 'Backhoe',
    'BACKHOE': 'Backhoe',
    'Bulldozer': 'Bulldozer',
    'BULLDOZER': 'Bulldozer',
    'Crane': 'Crane',
    'CRANE': 'Crane',
    'Tractor': 'Tractor',
    'TRACTOR': 'Tractor',
    
    # Trailers
    'TRAILER': 'Trailer',
    'Trailer': 'Trailer',
    'Chassis Cab': 'Chassis Cab',
    'CHASSIS CAB': 'Chassis Cab',
    
    # Utility vehicles
    'Convertible': 'Convertible',
    'CONVERTIBLE': 'Convertible',
    'AMBULETTE': 'Ambulette',
    'Ambulette': 'Ambulette',
    'VAN CAMPER': 'Van Camper',
    'Van Camper': 'Van Camper',
    'MOTORHOME': 'Motorhome',
    'Motorhome': 'Motorhome',
    'RV': 'Motorhome',
    
    # Delivery
    'USPS': 'USPS Vehicle',
    'UPS': 'UPS Vehicle',
    'FEDEX': 'FedEx Vehicle',
    'FedEx': 'FedEx Vehicle',
    'Postal': 'USPS Vehicle',
    'POSTAL': 'USPS Vehicle',
    
    # Misc
    'GOLF CART': 'Golf Cart',
    'Golf Cart': 'Golf Cart',
    'ATV': 'ATV',
    'All Terrain Vehicle': 'ATV',
    'ALL TERRAIN VEHICLE': 'ATV',
    'Snowmobile': 'Snowmobile',
    'SNOWMOBILE': 'Snowmobile',
}

# Apply standardization
print("\nApplying vehicle type standardization...")
for col in vehicle_cols:
    df[col] = df[col].map(lambda x: vehicle_mapping.get(x, x) if pd.notna(x) else x)

# Check results
print("\nVehicle types after standardization:")
all_vehicles_clean = []
for col in vehicle_cols:
    all_vehicles_clean.extend(df[col].dropna().unique().tolist())

clean_counts = Counter(all_vehicles_clean)
print(f"Total unique vehicle type values after cleaning: {len(clean_counts)}")
print(f"\nTop 30 vehicle types after cleaning:")
for vehicle, count in clean_counts.most_common(30):
    print(f"  {vehicle}: {count:,}")

# Additional data cleaning
print("\n" + "="*80)
print("ADDITIONAL DATA CLEANING")
print("="*80)

# Clean contributing factors
print("\nCleaning contributing factors...")
factor_cols = ['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2',
               'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4',
               'CONTRIBUTING FACTOR VEHICLE 5']

factor_mapping = {
    'Unspecified': 'Unspecified',
    'UNSPECIFIED': 'Unspecified',
    'Unknown': 'Unspecified',
    'UNKNOWN': 'Unspecified',
    '': 'Unspecified',
    'Driver Inattention/Distraction': 'Driver Inattention/Distraction',
    'DRIVER INATTENTION/DISTRACTION': 'Driver Inattention/Distraction',
    'Failure to Yield Right-of-Way': 'Failure to Yield Right-of-Way',
    'FAILURE TO YIELD RIGHT-OF-WAY': 'Failure to Yield Right-of-Way',
    'Following Too Closely': 'Following Too Closely',
    'FOLLOWING TOO CLOSELY': 'Following Too Closely',
    'Backing Unsafely': 'Backing Unsafely',
    'BACKING UNSAFELY': 'Backing Unsafely',
    'Passing or Lane Usage Improper': 'Passing or Lane Usage Improper',
    'PASSING OR LANE USAGE IMPROPER': 'Passing or Lane Usage Improper',
    'Passing Too Closely': 'Passing Too Closely',
    'PASSING TOO CLOSELY': 'Passing Too Closely',
    'Unsafe Speed': 'Unsafe Speed',
    'UNSAFE SPEED': 'Unsafe Speed',
    'Traffic Control Disregarded': 'Traffic Control Disregarded',
    'TRAFFIC CONTROL DISREGARDED': 'Traffic Control Disregarded',
    'Turning Improperly': 'Turning Improperly',
    'TURNING IMPROPERLY': 'Turning Improperly',
    'Other Vehicular': 'Other Vehicular',
    'OTHER VEHICULAR': 'Other Vehicular',
    'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion': 'Pedestrian/Bicyclist Error',
    'PEDESTRIAN/BICYCLIST/OTHER PEDESTRIAN ERROR/CONFUSION': 'Pedestrian/Bicyclist Error',
    'Fatigued/Drowsy': 'Fatigued/Drowsy',
    'FATIGUED/DROWSY': 'Fatigued/Drowsy',
    'Lost Consciousness': 'Lost Consciousness',
    'LOST CONSCIOUSNESS': 'Lost Consciousness',
    'Illness': 'Illness',
    'ILLNESS': 'Illness',
    'Alcohol Involvement': 'Alcohol Involvement',
    'ALCOHOL INVOLVEMENT': 'Alcohol Involvement',
    'Drugs (illegal)': 'Drugs (Illegal)',
    'DRUGS (ILLEGAL)': 'Drugs (Illegal)',
    'Prescription Medication': 'Prescription Medication',
    'PRESCRIPTION MEDICATION': 'Prescription Medication',
    'Outside Car Distraction': 'Outside Car Distraction',
    'OUTSIDE CAR DISTRACTION': 'Outside Car Distraction',
    'Passenger Distraction': 'Passenger Distraction',
    'PASSENGER DISTRACTION': 'Passenger Distraction',
    'Cell Phone (hand-held)': 'Cell Phone (Hand-Held)',
    'CELL PHONE (HAND-HELD)': 'Cell Phone (Hand-Held)',
    'Cell Phone (hands-free)': 'Cell Phone (Hands-Free)',
    'CELL PHONE (HANDS-FREE)': 'Cell Phone (Hands-Free)',
    'Texting': 'Texting',
    'TEXTING': 'Texting',
    'Using On Board Navigation Device': 'Using Navigation Device',
    'USING ON BOARD NAVIGATION DEVICE': 'Using Navigation Device',
    'Other Electronic Device': 'Other Electronic Device',
    'OTHER ELECTRONIC DEVICE': 'Other Electronic Device',
    'Eating or Drinking': 'Eating or Drinking',
    'EATING OR DRINKING': 'Eating or Drinking',
    'Glare': 'Glare',
    'GLARE': 'Glare',
    'Obstruction/Debris': 'Obstruction/Debris',
    'OBSTRUCTION/DEBRIS': 'Obstruction/Debris',
    'Animals Action': 'Animals Action',
    'ANIMALS ACTION': 'Animals Action',
    'Pavement Slippery': 'Pavement Slippery',
    'PAVEMENT SLIPPERY': 'Pavement Slippery',
    'Pavement Defective': 'Pavement Defective',
    'PAVEMENT DEFECTIVE': 'Pavement Defective',
    'View Obstructed/Limited': 'View Obstructed/Limited',
    'VIEW OBSTRUCTED/LIMITED': 'View Obstructed/Limited',
    'Lane Marking Improper/Inadequate': 'Lane Marking Improper/Inadequate',
    'LANE MARKING IMPROPER/INADEQUATE': 'Lane Marking Improper/Inadequate',
    'Traffic Control Device Improper/Non-Working': 'Traffic Control Device Improper',
    'TRAFFIC CONTROL DEVICE IMPROPER/NON-WORKING': 'Traffic Control Device Improper',
    'Shoulders Defective/Improper': 'Shoulders Defective/Improper',
    'SHOULDERS DEFECTIVE/IMPROPER': 'Shoulders Defective/Improper',
    'Brakes Defective': 'Brakes Defective',
    'BRAKES DEFECTIVE': 'Brakes Defective',
    'Steering Failure': 'Steering Failure',
    'STEERING FAILURE': 'Steering Failure',
    'Tire Failure/Inadequate': 'Tire Failure/Inadequate',
    'TIRE FAILURE/INADEQUATE': 'Tire Failure/Inadequate',
    'Accelerator Defective': 'Accelerator Defective',
    'ACCELERATOR DEFECTIVE': 'Accelerator Defective',
    'Windshield Inadequate': 'Windshield Inadequate',
    'WINDSHIELD INADEQUATE': 'Windshield Inadequate',
    'Headlights Defective': 'Headlights Defective',
    'HEADLIGHTS DEFECTIVE': 'Headlights Defective',
    'Taillights Defective': 'Taillights Defective',
    'TAILLIGHTS DEFECTIVE': 'Taillights Defective',
    'Other Lighting Defects': 'Other Lighting Defects',
    'OTHER LIGHTING DEFECTS': 'Other Lighting Defects',
    'Oversized Vehicle': 'Oversized Vehicle',
    'OVERSIZED VEHICLE': 'Oversized Vehicle',
    'Unsafe Lane Changing': 'Unsafe Lane Changing',
    'UNSAFE LANE CHANGING': 'Unsafe Lane Changing',
    'Aggressive Driving/Road Rage': 'Aggressive Driving/Road Rage',
    'AGGRESSIVE DRIVING/ROAD RAGE': 'Aggressive Driving/Road Rage',
}

for col in factor_cols:
    df[col] = df[col].map(lambda x: factor_mapping.get(x, x) if pd.notna(x) else x)

# Clean borough names
print("Cleaning borough names...")
df['BOROUGH'] = df['BOROUGH'].str.upper().str.strip()
df['BOROUGH'] = df['BOROUGH'].fillna('UNKNOWN')

# Save cleaned dataset
print("\n" + "="*80)
print("SAVING CLEANED DATASET")
print("="*80)

output_file = r"E:\PycharmProjects\data visualizer\NYC_Crashes_dataset_CLEAN.csv"
df.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to: {output_file}")
print(f"Final dataset shape: {df.shape}")
print(f"Total records: {len(df):,}")

print("\n" + "="*80)
print("DATA CLEANING COMPLETE!")
print("="*80)
print("\nSummary of changes:")
print(f"  - Standardized vehicle type categories")
print(f"  - Merged duplicate vehicle types (case variations, synonyms)")
print(f"  - Standardized contributing factor categories")
print(f"  - Cleaned borough names")
print(f"  - Output file: NYC_Crashes_dataset_CLEAN.csv")
