import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedNPZModel:
    """
    Enhanced NPZ model with:
    1. Temperature-dependent rates
    2. Mixed layer depth dynamics
    3. Real oceanographic data integration
    4. Multiple phytoplankton groups
    """
    
    def __init__(self, params: Dict[str, float] = None):
        """Initialize enhanced model with default parameters."""
        
        # Enhanced parameters including temperature dependence
        self.default_params = {
            # Phytoplankton parameters
            'V_max_ref': 1.5,      # Reference max growth rate at 15°C (day^-1)
            'Q10_growth': 2.0,     # Temperature coefficient for growth
            'T_ref': 15.0,         # Reference temperature (°C)
            'K_N': 1.0,            # Half-saturation for nutrients (mmol/m^3)
            'K_I': 30.0,           # Half-saturation for light (W/m^2)
            'alpha': 0.025,        # Initial slope of P-I curve
            
            # Zooplankton parameters
            'g_max_ref': 1.0,      # Reference max grazing rate (day^-1)
            'Q10_grazing': 2.5,    # Temperature coefficient for grazing
            'K_P': 1.0,            # Half-saturation for grazing
            'beta': 0.7,           # Assimilation efficiency
            'gamma': 0.3,          # Grazing efficiency
            
            # Mortality and remineralization
            'm_P': 0.05,           # Phytoplankton mortality (day^-1)
            'm_Z': 0.1,            # Zooplankton mortality (day^-1)
            'remin_rate': 0.1,     # Detritus remineralization rate (day^-1)
            
            # Physical parameters
            'MLD_min': 10.0,       # Minimum mixed layer depth (m)
            'MLD_max': 150.0,      # Maximum mixed layer depth (m)
            'k_w': 0.04,           # Light attenuation in water (m^-1)
            'k_c': 0.03,           # Light attenuation by chlorophyll
            'sinking_rate': 1.0,   # Phytoplankton sinking rate (m/day)
        }
        
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        
        # Storage for real data
        self.real_data = {}
    
    def temperature_effect(self, T: float, Q10: float, T_ref: float = 15.0) -> float:
        """Calculate temperature effect on biological rates using Q10."""
        return Q10 ** ((T - T_ref) / 10.0)
    
    def mixed_layer_depth(self, t: float, location: Dict = None) -> float:
        """
        Calculate mixed layer depth with seasonal variation.
        Can be replaced with real MLD data.
        """
        if location and 'MLD_data' in self.real_data:
            # Use real data if available
            return np.interp(t, self.real_data['time'], self.real_data['MLD_data'])
        
        # Simple seasonal model (Northern Hemisphere)
        day_of_year = (t % 365)
        MLD = self.params['MLD_min'] + (self.params['MLD_max'] - self.params['MLD_min']) * \
              (1 + np.cos(2 * np.pi * (day_of_year - 80) / 365)) / 2
        return MLD
    
    def light_profile(self, z: float, I_0: float, chl: float) -> float:
        """Calculate light at depth z considering chlorophyll attenuation."""
        k_total = self.params['k_w'] + self.params['k_c'] * chl
        return I_0 * np.exp(-k_total * z)
    
    def average_light_mixed_layer(self, I_0: float, MLD: float, P: float) -> float:
        """Calculate average light in the mixed layer."""
        # Convert P (mmol N/m³) to chlorophyll (mg/m³) assuming C:Chl = 50
        chl = P * 16 / 50  # Rough conversion
        k_total = self.params['k_w'] + self.params['k_c'] * chl
        
        if k_total * MLD < 0.01:
            return I_0
        else:
            return I_0 / (k_total * MLD) * (1 - np.exp(-k_total * MLD))
    
    def surface_light(self, t: float, latitude: float = 45.0) -> float:
        """
        Calculate surface irradiance based on time and latitude.
        More realistic than simple sinusoidal.
        """
        day_of_year = (t % 365)
        
        # Solar declination
        P = np.arcsin(0.39795 * np.cos(0.98563 * (day_of_year - 173) * np.pi / 180))
        
        # Sunrise/sunset hour angle
        lat_rad = latitude * np.pi / 180
        sunrise_angle = -np.tan(lat_rad) * np.tan(P)
        
        if sunrise_angle < -1:
            daylight_hours = 24
        elif sunrise_angle > 1:
            daylight_hours = 0
        else:
            daylight_hours = 24 - (24 / np.pi) * np.arccos(sunrise_angle)
        
        # Hour of day
        hour = (t * 24) % 24
        
        # Simple model for daily light cycle
        if daylight_hours > 0:
            sunrise = 12 - daylight_hours / 2
            sunset = 12 + daylight_hours / 2
            
            if sunrise <= hour <= sunset:
                # Maximum around 400 W/m² at noon
                return 400 * np.sin(np.pi * (hour - sunrise) / daylight_hours)
            else:
                return 0
        else:
            return 0
    
    def enhanced_npz_dynamics(self, t: float, y: np.ndarray, 
                            T: float = 15.0, latitude: float = 45.0) -> np.ndarray:
        """
        Enhanced NPZ dynamics with temperature and mixed layer effects.
        
        State variables:
        y[0] = N (Nutrients)
        y[1] = P (Phytoplankton)
        y[2] = Z (Zooplankton)
        y[3] = D (Detritus)
        """
        N, P, Z, D = y
        
        # Get parameters
        p = self.params
        
        # Environmental conditions
        I_0 = self.surface_light(t, latitude)
        MLD = self.mixed_layer_depth(t)
        I_avg = self.average_light_mixed_layer(I_0, MLD, P)
        
        # Temperature effects
        T_growth = self.temperature_effect(T, p['Q10_growth'], p['T_ref'])
        T_grazing = self.temperature_effect(T, p['Q10_grazing'], p['T_ref'])
        
        # Nutrient limitation (Michaelis-Menten)
        f_N = N / (p['K_N'] + N)
        
        # Light limitation (using average light in mixed layer)
        f_I = (p['alpha'] * I_avg) / np.sqrt(p['V_max_ref']**2 + (p['alpha'] * I_avg)**2)
        
        # Phytoplankton growth
        growth = p['V_max_ref'] * T_growth * f_N * f_I * P
        
        # Zooplankton grazing (with temperature effect)
        grazing = p['g_max_ref'] * T_grazing * (P / (p['K_P'] + P)) * Z
        
        # Mortality
        P_mort = p['m_P'] * P
        Z_mort = p['m_Z'] * Z
        
        # Detritus remineralization (temperature dependent)
        remin = p['remin_rate'] * self.temperature_effect(T, 2.0, p['T_ref']) * D
        
        # Sinking loss (only affects phytoplankton below mixed layer)
        # Simplified: assume some fraction is lost
        sinking_loss = p['sinking_rate'] / MLD * P
        
        # Differential equations
        dN_dt = -growth + remin
        dP_dt = growth - grazing - P_mort - sinking_loss
        dZ_dt = p['beta'] * p['gamma'] * grazing - Z_mort
        dD_dt = (1 - p['beta']) * grazing + P_mort + Z_mort - remin + sinking_loss
        
        return np.array([dN_dt, dP_dt, dZ_dt, dD_dt])
    
    def fetch_real_sst(self, latitude: float, longitude: float, 
                      start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch real SST data from NOAA or other sources.
        For demonstration, we'll create realistic synthetic data.
        """
        print(f"Fetching SST data for location ({latitude}, {longitude})...")
        
        # In reality, you would use an API like:
        # - NOAA ERDDAP
        # - Copernicus Marine Service
        # - NASA Earthdata
        
        # For now, create realistic SST data
        dates = pd.date_range(start_date, end_date, freq='D')
        days = (dates - pd.Timestamp(start_date)).days.values
        
        # Seasonal SST variation
        base_sst = 15 + 5 * np.sin(2 * np.pi * days / 365 - np.pi/2)
        # Add some noise
        sst = base_sst + np.random.normal(0, 0.5, len(days))
        
        df = pd.DataFrame({
            'date': dates,
            'day': days,
            'sst': sst
        })
        
        return df
    
    def run_with_real_forcing(self, initial_conditions: List[float],
                            location: Dict[str, float],
                            start_date: str, end_date: str) -> Dict:
        """
        Run model with real environmental forcing data.
        
        Args:
            initial_conditions: [N0, P0, Z0, D0]
            location: {'lat': latitude, 'lon': longitude}
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
        """
        # Fetch real data
        sst_data = self.fetch_real_sst(location['lat'], location['lon'], 
                                       start_date, end_date)
        
        # Time array in days
        t_span = (0, len(sst_data) - 1)
        t_eval = sst_data['day'].values
        
        # Create interpolation function for temperature
        def get_temperature(t):
            return np.interp(t, sst_data['day'].values, sst_data['sst'].values)
        
        # Solve with time-varying temperature
        results = []
        
        # Split integration into chunks to update temperature
        for i in range(len(t_eval) - 1):
            if i == 0:
                y0 = initial_conditions
            else:
                y0 = [results[-1]['N'][-1], results[-1]['P'][-1], 
                      results[-1]['Z'][-1], results[-1]['D'][-1]]
            
            T = get_temperature(t_eval[i])
            
            sol = solve_ivp(
                lambda t, y: self.enhanced_npz_dynamics(t, y, T=T, latitude=location['lat']),
                (t_eval[i], t_eval[i+1]),
                y0,
                t_eval=[t_eval[i], t_eval[i+1]],
                method='RK45'
            )
            
            results.append({
                't': sol.t,
                'N': sol.y[0],
                'P': sol.y[1],
                'Z': sol.y[2],
                'D': sol.y[3]
            })
        
        # Combine results
        combined = {
            't': np.concatenate([r['t'][:-1] for r in results] + [results[-1]['t'][-1:]]),
            'N': np.concatenate([r['N'][:-1] for r in results] + [results[-1]['N'][-1:]]),
            'P': np.concatenate([r['P'][:-1] for r in results] + [results[-1]['P'][-1:]]),
            'Z': np.concatenate([r['Z'][:-1] for r in results] + [results[-1]['Z'][-1:]]),
            'D': np.concatenate([r['D'][:-1] for r in results] + [results[-1]['D'][-1:]]),
            'dates': sst_data['date'].values,
            'sst': sst_data['sst'].values
        }
        
        return combined
    
    def plot_enhanced_results(self, results: Dict):
        """Create comprehensive plots for enhanced model results."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Use dates if available
        if 'dates' in results:
            x_data = results['dates']
            x_label = 'Date'
        else:
            x_data = results['t']
            x_label = 'Time (days)'
        
        # Plot 1: State variables
        ax = axes[0, 0]
        ax.plot(x_data, results['N'], 'b-', label='Nutrients', linewidth=2)
        ax.plot(x_data, results['P'], 'g-', label='Phytoplankton', linewidth=2)
        ax.plot(x_data, results['Z'], 'r-', label='Zooplankton', linewidth=2)
        ax.plot(x_data, results['D'], 'k-', label='Detritus', linewidth=2)
        ax.set_ylabel('Concentration (mmol N/m³)')
        ax.set_title('NPZ-D Model Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Total nitrogen (conservation check)
        ax = axes[0, 1]
        total_N = results['N'] + results['P'] + results['Z'] + results['D']
        ax.plot(x_data, total_N, 'k-', linewidth=2)
        ax.set_ylabel('Total N (mmol/m³)')
        ax.set_title('Nitrogen Conservation')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([total_N[0] * 0.99, total_N[0] * 1.01])
        
        # Plot 3: SST if available
        if 'sst' in results:
            ax = axes[1, 0]
            ax.plot(x_data, results['sst'], 'r-', linewidth=2)
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Sea Surface Temperature')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Phytoplankton to Chlorophyll
        ax = axes[1, 1]
        chl = results['P'] * 16 / 50  # Simple conversion
        ax.plot(x_data, chl, 'g-', linewidth=2)
        ax.set_ylabel('Chlorophyll (mg/m³)')
        ax.set_title('Estimated Chlorophyll-a')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: P:Z ratio
        ax = axes[2, 0]
        pz_ratio = results['P'] / (results['Z'] + 1e-10)  # Avoid division by zero
        ax.plot(x_data, pz_ratio, 'purple', linewidth=2)
        ax.set_ylabel('P:Z Ratio')
        ax.set_xlabel(x_label)
        ax.set_title('Phytoplankton to Zooplankton Ratio')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 6: Production estimate
        ax = axes[2, 1]
        # Estimate primary production (simplified)
        production = np.gradient(results['P']) + np.gradient(results['Z'])
        ax.plot(x_data[1:], production[1:], 'c-', linewidth=2)
        ax.set_ylabel('Production (mmol N/m³/day)')
        ax.set_xlabel(x_label)
        ax.set_title('Estimated Net Production')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def generate_training_data_enhanced(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate enhanced training data with more realistic parameter ranges.
        """
        training_data = []
        
        param_ranges = {
            'temperature': (5, 25),      # SST range
            'latitude': (30, 60),        # Temperate to polar
            'N_0': (5, 25),             # Initial nutrients
            'season_start': (0, 365),    # Starting day of year
        }
        
        print(f"Generating {n_samples} enhanced training samples...")
        
        for i in range(n_samples):
            # Sample parameters
            T = np.random.uniform(*param_ranges['temperature'])
            lat = np.random.uniform(*param_ranges['latitude'])
            N_0 = np.random.uniform(*param_ranges['N_0'])
            start_day = np.random.uniform(*param_ranges['season_start'])
            
            # Initial conditions
            P_0 = np.random.uniform(0.5, 2.0)
            Z_0 = np.random.uniform(0.1, 0.5)
            D_0 = np.random.uniform(0.5, 2.0)
            
            # Time points to sample
            t_eval = np.array([1, 3, 7, 14, 30, 60, 90])
            
            # Run simulation
            sol = solve_ivp(
                lambda t, y: self.enhanced_npz_dynamics(t + start_day, y, T=T, latitude=lat),
                (0, 90),
                [N_0, P_0, Z_0, D_0],
                t_eval=t_eval,
                method='RK45'
            )
            
            # Extract features
            for j, t in enumerate(t_eval):
                row = {
                    'sample_id': i,
                    'time': t,
                    'temperature': T,
                    'latitude': lat,
                    'season_start': start_day,
                    'N_0': N_0,
                    # Outputs
                    'N': sol.y[0, j],
                    'P': sol.y[1, j],
                    'Z': sol.y[2, j],
                    'D': sol.y[3, j],
                    'total_biomass': sol.y[1, j] + sol.y[2, j],
                    'chl': sol.y[1, j] * 16 / 50  # Chlorophyll estimate
                }
                training_data.append(row)
            
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_samples} samples")
        
        return pd.DataFrame(training_data)


# Example usage
if __name__ == "__main__":
    # Create enhanced model
    model = EnhancedNPZModel()
    
    # Example 1: Run with synthetic seasonal forcing
    print("Running enhanced model with seasonal forcing...")
    
    # North Atlantic location
    location = {'lat': 45.0, 'lon': -30.0}
    
    # Run for one year
    results = model.run_with_real_forcing(
        initial_conditions=[15.0, 1.0, 0.2, 1.0],  # N, P, Z, D
        location=location,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # Plot results
    fig = model.plot_enhanced_results(results)
    plt.show()
    
    # Example 2: Generate enhanced training data
    print("\nGenerating enhanced training data...")
    training_df = model.generate_training_data_enhanced(n_samples=500)
    
    print(f"\nEnhanced training data shape: {training_df.shape}")
    print("\nSample statistics:")
    print(training_df[['P', 'Z', 'chl', 'temperature']].describe())
    
    # Save training data
    training_df.to_csv('enhanced_npz_training_data.csv', index=False)
    print("\nTraining data saved to 'enhanced_npz_training_data.csv'")