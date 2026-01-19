# NYC Motor Vehicle Collisions Dashboard

An interactive data visualization dashboard for analyzing NYC motor vehicle collision data from NYC Open Data.

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

3. Download the NYC Crashes dataset and place it in the project directory as `NYC_Crashes_dataset.csv`
   - Data source: [NYC Open Data - Motor Vehicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)

4. Run the data cleaning script (optional but recommended):
```bash
python analyze_and_clean_data.py
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Project Structure

```
NYC_Crash_Analysis/
├── app.py                          # Main Streamlit dashboard application
├── analyze_and_clean_data.py       # Data cleaning and standardization script
├── .streamlit/
│   └── config.toml                 # Streamlit configuration (light theme)
├── NYC_Crashes_dataset.csv         # Original dataset (not included in repo)
├── NYC_Crashes_dataset_CLEAN.csv   # Cleaned dataset (not included in repo)
├── .gitignore
└── README.md
```

## Data Cleaning

The `analyze_and_clean_data.py` script performs:
- Vehicle type standardization (merges duplicates like SEDAN/Sedan, TAXI/Taxi)
- Contributing factor standardization
- Borough name cleaning
- Case normalization

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
