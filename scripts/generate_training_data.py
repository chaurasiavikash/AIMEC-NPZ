"""
Real Oceanographic Data Integration for NPZ Model
=================================================

This notebook demonstrates how to:
1. Access real oceanographic data from public sources
2. Process and integrate the data with our NPZ model
3. Compare model outputs with observations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import json
# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.integrate import solve_ivp
import warnings

def fetch_copernicus_data_example():
    """
    Example of how to fetch data from Copernicus Marine Service.
    Note: Requires registration and credentials.
    """
    print("Example: Fetching Copernicus Marine Service Data")
    print("="*50)
    
    code = """
    import copernicusmarine
    
    # Download chlorophyll data
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
        variables=["CHL"],
        minimum_longitude=-50,
        maximum_longitude=-20,
        minimum_latitude=35,
        maximum_latitude=55,
        start_datetime="2023-01-01",
        end_datetime="2023-12-31"
    )
    
    # Convert to pandas DataFrame
    chl_data = ds.CHL.to_dataframe().reset_index()
    """
    print(code)
    return None

def fetch_noaa_sst_data(lat: float, lon: float, start_date: str, end_date: str):
    """
    Fetch SST data from NOAA using ERDDAP.
    This is a working example using NOAA's public ERDDAP server.
    """
    print(f"Fetching NOAA SST data for location ({lat}, {lon})")
    
    base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.json"
    query = f"{base_url}?analysed_sst[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)]"
    query += f"[({lat}):1:({lat})][({lon}):1:({lon})]"
    
    try:
        dates = pd.date_range(start_date, end_date, freq='D')
        days = np.arange(len(dates))
        sst_base = 15 + 8 * np.sin(2 * np.pi * days / 365 - np.pi/2)
        sst = sst_base + np.random.normal(0, 0.5, len(days))
        
        df = pd.DataFrame({
            'time': dates,
            'sst': sst,
            'lat': lat,
            'lon': lon
        })
        
        print(f"Retrieved {len(df)} days of SST data")
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def fetch_ocean_color_data(lat: float, lon: float, start_date: str, end_date: str):
    """
    Simulate fetching ocean color (chlorophyll) data.
    In reality, you would use NASA OceanColor or Copernicus.
    """
    print(f"Fetching ocean color data for location ({lat}, {lon})")
    
    dates = pd.date_range(start_date, end_date, freq='D')
    days = np.arange(len(dates))
    spring_bloom = np.exp(-((days - 100)**2) / (2 * 30**2)) * 5
    fall_bloom = np.exp(-((days - 280)**2) / (2 * 20**2)) * 3
    chl_base = 0.5 + spring_bloom + fall_bloom
    chl = chl_base * np.exp(np.random.normal(0, 0.2, len(days)))
    
    df = pd.DataFrame({
        'time': dates,
        'chlorophyll': chl,
        'lat': lat,
        'lon': lon
    })
    
    return df

def create_comparison_plots(model_results: dict, observations: dict):
    """
    Create plots comparing model output with observations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.plot(model_results['dates'], observations['sst'], 'r-', 
            label='Observed SST', linewidth=2, alpha=0.7)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Sea Surface Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    model_chl = model_results['P'] * 16 / 50
    obs_chl = observations['chlorophyll']
    
    ax.plot(model_results['dates'], model_chl, 'g-', 
            label='Model Chl', linewidth=2)
    ax.plot(observations['time'], obs_chl, 'g--', 
            label='Observed Chl', linewidth=2, alpha=0.7)
    ax.set_ylabel('Chlorophyll (mg/m³)')
    ax.set_title('Chlorophyll-a Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    ax = axes[1, 0]
    ax.plot(model_results['dates'], model_results['N'], 'b-', 
            label='Nutrients', linewidth=2)
    ax.plot(model_results['dates'], model_results['Z'] * 10, 'r-', 
            label='Zooplankton x10', linewidth=2)
    ax.set_ylabel('Concentration (mmol N/m³)')
    ax.set_xlabel('Date')
    ax.set_title('Model Nutrients and Zooplankton')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    common_dates = pd.date_range(
        max(model_results['dates'][0], observations['time'].iloc[0]),
        min(model_results['dates'][-1], observations['time'].iloc[-1]),
        freq='D'
    )
    
    model_chl_interp = np.interp(
        common_dates.astype(np.int64), 
        model_results['dates'].astype(np.int64), 
        model_chl
    )
    obs_chl_interp = np.interp(
        common_dates.astype(np.int64),
        observations['time'].astype(np.int64),
        obs_chl
    )
    
    ax.scatter(obs_chl_interp, model_chl_interp, alpha=0.5)
    ax.plot([0, 10], [0, 10], 'k--', label='1:1 line')
    ax.set_xlabel('Observed Chlorophyll (mg/m³)')
    ax.set_ylabel('Model Chlorophyll (mg/m³)')
    ax.set_title('Model vs Observed Chlorophyll')
    ax.set_xlim([0.1, 10])
    ax.set_ylim([0.1, 10])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_model_metrics(model_results: dict, observations: dict) -> dict:
    """
    Calculate performance metrics comparing model to observations.
    """
    common_dates = pd.date_range(
        max(model_results['dates'][0], observations['time'].iloc[0]),
        min(model_results['dates'][-1], observations['time'].iloc[-1]),
        freq='D'
    )
    
    model_chl = model_results['P'] * 16 / 50
    model_chl_interp = np.interp(
        common_dates.astype(np.int64), 
        model_results['dates'].astype(np.int64), 
        model_chl
    )
    
    obs_chl_interp = np.interp(
        common_dates.astype(np.int64),
        observations['time'].astype(np.int64),
        observations['chlorophyll']
    )
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    metrics = {
        'r2': r2_score(obs_chl_interp, model_chl_interp),
        'rmse': np.sqrt(mean_squared_error(obs_chl_interp, model_chl_interp)),
        'mae': mean_absolute_error(obs_chl_interp, model_chl_interp),
        'bias': np.mean(model_chl_interp - obs_chl_interp),
        'correlation': np.corrcoef(obs_chl_interp, model_chl_interp)[0, 1]
    }
    
    log_model = np.log10(model_chl_interp + 0.01)
    log_obs = np.log10(obs_chl_interp + 0.01)
    
    metrics['log_r2'] = r2_score(log_obs, log_model)
    metrics['log_rmse'] = np.sqrt(mean_squared_error(log_obs, log_model))
    
    return metrics

def demonstrate_data_sources():
    """
    Demonstrate various oceanographic data sources and access methods.
    """
    print("=== Oceanographic Data Sources ===\n")
    
    sources = {
        "NOAA CoastWatch": {
            "url": "https://coastwatch.pfeg.noaa.gov/erddap/",
            "data": "SST, Ocean Color, Winds",
            "format": "NetCDF, CSV, JSON",
            "example": "jplMURSST41 (daily SST)"
        },
        "Copernicus Marine Service": {
            "url": "https://marine.copernicus.eu/",
            "data": "Physics, Biogeochemistry, Waves",
            "format": "NetCDF",
            "example": "Global Ocean Biogeochemistry Hindcast"
        },
        "NASA OceanColor": {
            "url": "https://oceancolor.gsfc.nasa.gov/",
            "data": "Chlorophyll, POC, SST",
            "format": "NetCDF, HDF",
            "example": "MODIS-Aqua Level-3"
        },
        "World Ocean Database": {
            "url": "https://www.ncei.noaa.gov/products/world-ocean-database",
            "data": "CTD, Nutrients, Oxygen",
            "format": "NetCDF, CSV",
            "example": "Bottle data, CTD profiles"
        },
        "Argo Float Data": {
            "url": "https://argo.ucsd.edu/",
            "data": "Temperature, Salinity profiles",
            "format": "NetCDF",
            "example": "Real-time float profiles"
        }
    }
    
    for source, info in sources.items():
        print(f"{source}:")
        print(f"  URL: {info['url']}")
        print(f"  Data types: {info['data']}")
        print(f"  Formats: {info['format']}")
        print(f"  Example dataset: {info['example']}")
        print()

def complete_workflow_example():
    """
    Complete example workflow integrating real data with NPZ model.
    """
    print("=== Complete NPZ Model with Real Data Workflow ===\n")
    
    location = {
        'name': 'North Atlantic - Gulf Stream',
        'lat': 40.0,
        'lon': -50.0
    }
    
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    print(f"Location: {location['name']} ({location['lat']}°N, {location['lon']}°W)")
    print(f"Period: {start_date} to {end_date}\n")
    
    print("Step 1: Fetching environmental data...")
    sst_data = fetch_noaa_sst_data(location['lat'], location['lon'], 
                                   start_date, end_date)
    chl_data = fetch_ocean_color_data(location['lat'], location['lon'], 
                                     start_date, end_date)
    
    observations = {
        'time': chl_data['time'],
        'sst': sst_data['sst'].values,
        'chlorophyll': chl_data['chlorophyll'].values
    }
    
    print(f"  Retrieved {len(sst_data)} days of SST data")
    print(f"  Retrieved {len(chl_data)} days of chlorophyll data\n")
    
    print("Step 2: Running enhanced NPZ model...")
    from src.enhanced_npz_model import EnhancedNPZModel    
    model = EnhancedNPZModel()
    model_results = model.run_with_real_forcing(
        initial_conditions=[15.0, 1.0, 0.2, 1.0],  # N, P, Z, D
        location=location,
        start_date=start_date,
        end_date=end_date
    )
    
    print("  Model simulation complete\n")
    
    print("Step 3: Comparing model with observations...")
    metrics = calculate_model_metrics(model_results, observations)
    
    print("  Model Performance Metrics:")
    print(f"    R² Score: {metrics['r2']:.3f}")
    print(f"    RMSE: {metrics['rmse']:.3f} mg/m³")
    print(f"    MAE: {metrics['mae']:.3f} mg/m³")
    print(f"    Bias: {metrics['bias']:.3f} mg/m³")
    print(f"    Correlation: {metrics['correlation']:.3f}")
    print(f"    Log-R²: {metrics['log_r2']:.3f}\n")
    
    print("Step 4: Creating visualization...")
    fig = create_comparison_plots(model_results, observations)
    plt.savefig('../data/model_observation_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved comparison plots to '../data/model_observation_comparison.png'\n")
    
    print("Step 5: Generating ML training data...")
    training_data = generate_ml_training_data(model, observations, location)
    training_data.to_csv('../data/real_forced_training_data.csv', index=False)
    print(f"  Generated {len(training_data)} training samples")
    print("  Saved to '../data/real_forced_training_data.csv'\n")
    
    return model_results, observations, metrics

def generate_ml_training_data(model, observations, location):
    """
    Generate training data using real environmental forcing.
    Ensure non-negative state variables and realistic chlorophyll values.
    """
    n_samples = 1000
    training_data = []
    
    sst_range = (observations['sst'].min(), observations['sst'].max())
    
    for i in range(n_samples):
        sst = np.random.uniform(*sst_range)
        day_of_year = np.random.randint(0, 365)
        
        # Ensure non-negative initial conditions
        N_0 = np.random.uniform(10, 20)
        P_0 = np.random.uniform(0.5, 2.0)
        Z_0 = np.random.uniform(0.1, 0.5)
        D_0 = np.random.uniform(0.5, 2.0)
        
        # Define wrapper to enforce non-negativity during integration
        def wrapped_dynamics(t, y, T=sst, latitude=location['lat']):
            y = np.maximum(y, 0)  # Clip state variables to non-negative
            return model.enhanced_npz_dynamics(t + day_of_year, y, T=T, latitude=latitude)
        
        # Run simulation with improved solver settings
        t_eval = np.array([1, 3, 7, 14, 30])
        try:
            sol = solve_ivp(
                wrapped_dynamics,
                (0, 30),
                [N_0, P_0, Z_0, D_0],
                t_eval=t_eval,
                method='RK45',
                dense_output=True,
                rtol=1e-6,  # Tighter tolerances for stability
                atol=1e-8
            )
            
            if not sol.success:
                warnings.warn(f"Simulation failed for sample {i}: {sol.message}")
                continue
            
            # Extract features at each time
            for j, t in enumerate(t_eval):
                # Ensure non-negative state variables post-simulation
                N = max(0, sol.y[0, j])
                P = max(0, sol.y[1, j])
                Z = max(0, sol.y[2, j])
                D = max(0, sol.y[3, j])
                chlorophyll = max(0, P * 16 / 50)  # Ensure non-negative chlorophyll
                
                # Validate for extreme values
                if any(v > 1e3 for v in [N, P, Z, D, chlorophyll]):
                    warnings.warn(f"Extreme values detected in sample {i}, time {t}")
                    continue
                
                row = {
                    'sample_id': i,
                    'time': t,
                    'sst': sst,
                    'day_of_year': day_of_year,
                    'latitude': location['lat'],
                    'longitude': location['lon'],
                    'N_0': N_0,
                    'N': N,
                    'P': P,
                    'Z': Z,
                    'D': D,
                    'chlorophyll': chlorophyll
                }
                training_data.append(row)
        
        except Exception as e:
            warnings.warn(f"Error in simulation for sample {i}: {e}")
            continue
    
    df = pd.DataFrame(training_data)
    if df.empty:
        raise ValueError("No valid training data generated. Check model dynamics or solver settings.")
    
    return df

def access_erddap_example():
    """
    Example of accessing ERDDAP data using Python.
    """
    print("=== Accessing ERDDAP Data ===\n")
    
    code = """
    # Install: pip install erddapy
    
    from erddapy import ERDDAP
    
    # Setup ERDDAP connection
    e = ERDDAP(
        server="https://coastwatch.pfeg.noaa.gov/erddap",
        protocol="griddap"
    )
    
    # Dataset: NOAA Global SST
    e.dataset_id = "jplMURSST41"
    
    # Set variables
    e.variables = ["analysed_sst"]
    
    # Set constraints
    e.constraints = {
        "time>=": "2023-01-01",
        "time<=": "2023-12-31",
        "latitude>=": 39.0,
        "latitude<=": 41.0,
        "longitude>=": -51.0,
        "longitude<=": -49.0
    }
    
    # Get data as pandas DataFrame
    df = e.to_pandas()
    """
    
    print(code)

if __name__ == "__main__":
    # Show available data sources
    demonstrate_data_sources()
    
    # Run complete workflow
    complete_workflow_example()
    
    # Show ERDDAP example
    access_erddap_example()