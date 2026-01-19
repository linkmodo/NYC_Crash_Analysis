# NYC Motor Vehicle Collisions Dashboard

An interactive data visualization dashboard for analyzing NYC motor vehicle collision data from NYC Open Data (2012-2025).

## Features

- **Overview Analysis**: Annual trends, borough distribution, and casualty breakdowns
- **Geographic Analysis**: Interactive maps (scatter, density, cluster) with street-level insights
- **Temporal Analysis**: Hourly, daily, and monthly crash patterns with rush hour analysis
- **Cause Analysis**: Contributing factors, vehicle types, and their relationships
- **Severity Analysis**: Fatal crash analysis, victim types, and severity distribution with scatter plots

## Tech Stack

- **Python 3.12**
- **Streamlit**: Interactive web framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/linkmodo/NYC_Crash_Analysis.git
cd NYC_Crash_Analysis
```

2. Install dependencies:
```bash
pip install streamlit pandas plotly numpy
```

3. The cleaned and standardized dataset is included in the repository:
   - `NYC_crashes_dataset_STANDARDIZED.csv` (stored with Git LFS)
   - Data source: [NYC Open Data - Motor Vehicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Project Structure

```
NYC_Crash_Analysis/
├── app.py                                  # Main Streamlit dashboard application
├── standardize_vehicle_types.py            # Vehicle type standardization script
├── analyze_missing_data.py                 # Missing data analysis script
├── create_final_clean_dataset.py           # Data cleaning script
├── remove_2026_data.py                     # Remove incomplete 2026 data
├── .streamlit/
│   └── config.toml                         # Streamlit configuration (light theme)
├── NYC_crashes_dataset_STANDARDIZED.csv    # Final cleaned dataset (Git LFS)
├── .gitignore
├── .gitattributes                          # Git LFS configuration
└── README.md
```

## Data Processing Pipeline

### 1. Missing Data Analysis (`analyze_missing_data.py`)
- Analyzes missing data patterns across all columns
- Identifies rows with invalid coordinates
- Recommends data removal strategies
- **Result**: Only 0.35% of data removed (invalid coordinates)

### 2. Data Cleaning (`create_final_clean_dataset.py`)
- Removes invalid location coordinates (outside NYC boundaries)
- Fills missing casualty counts with 0
- Standardizes borough names
- Fills missing contributing factors with 'Unspecified'
- Fills missing vehicle types with 'Unknown'

### 3. Vehicle Type Standardization (`standardize_vehicle_types.py`)
- Merges 367 vehicle type variations into 51 standardized categories
- Examples:
  - `Taxi`, `TAXI`, `taxi` → **Taxi**
  - `Unknown`, `UNK`, `Unkown`, `UNKOWN` → **Unknown**
  - `Sedan`, `SEDAN`, `4 dr sedan` → **Sedan**
  - `Livery Vehicle`, `Black Car` → **For-Hire Vehicle**
- Reduces duplicate categories by 380+ variations

### 4. Incomplete Year Removal (`remove_2026_data.py`)
- Removes 2,374 records from 2026 (incomplete year)
- Final dataset: **1,985,248 rows** covering 2012-2025

## Final Dataset Statistics

- **Total Records**: 1,985,248
- **Date Range**: 2012-2025 (complete years only)
- **Data Retention**: 99.65% of original data
- **Missing Data**: Minimal (<1% in critical columns)
- **Vehicle Categories**: 51 standardized types
- **Top Vehicle Types**: Sedan (31.74%), SUV (31.00%), Other (22.87%)

## Features by Page

### Overview
- Total crashes, injuries, and fatalities metrics
- Multi-year trend analysis
- Borough distribution pie chart
- Casualty breakdown by category

### Geographic Analysis
- Interactive maps with multiple view types
- Top dangerous streets analysis
- Crash density visualization
- Borough-level statistics

### Temporal Analysis
- Hourly and daily crash patterns
- Rush hour analysis
- Monthly trends and year-over-year comparison
- Day of week heatmap

### Cause Analysis
- Top contributing factors
- Vehicle types involved
- Fatal crash factor analysis
- Vehicle type vs contributing factor relationship matrix
- Deep dive analysis by vehicle type

### Severity Analysis
- Fatal crash trends
- Victim type analysis (pedestrians, cyclists, motorists)
- Borough fatality comparison
- Scatter plots with linear correlations
- Multi-fatality crash analysis

## Visualizations

- Interactive bar charts with sorting and percentage labels
- Scatter plots with OLS trendlines
- Geographic maps (scatter, density, cluster)
- Line charts for temporal trends
- Pie charts for distributions
- Heatmaps for relationship analysis

## Data Source

NYC Open Data - Motor Vehicle Collisions - Crashes
- Updated regularly by NYPD
- Contains collision data from 2012 to present
- Includes location, time, casualties, contributing factors, and vehicle information

## License

This project is for educational and analytical purposes.

## Author

Built with Streamlit & Plotly for NYC crash data analysis.
