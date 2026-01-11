"""
Synthetic Sensor Data Generator
Generates realistic industrial machine sensor data for training.
Now includes Energy Consumption and RUL (Remaining Useful Life) targets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import settings

def generate_sensor_data(n_samples: int = 10000, save_path: Path = None) -> pd.DataFrame:
    """
    Generate synthetic sensor data for predictive maintenance training.
    
    Features:
    - temperature: Machine operating temperature (°C)
    - vibration: Vibration level (mm/s)
    - pressure: Operating pressure (bar)
    - rpm: Rotational speed (revolutions per minute)
    - energy_kwh: Energy consumption (kWh) - NEW FEATURE
    
    Targets:
    - failure: Binary label (0 = normal, 1 = failure)
    - rul_days: Remaining Useful Life in days (Regression target) - NEW FEATURE
    """
    np.random.seed(settings.RANDOM_STATE)
    
    # Define segment sizes
    # 70% Healthy, 20% Warning (Degrading), 10% Failure
    n_normal = int(n_samples * 0.7)
    n_warning = int(n_samples * 0.2)
    n_failure = n_samples - n_normal - n_warning
    
    # ==========================================
    # 1. Normal Operation (Healthy State)
    # ==========================================
    # Stable readings, low energy, high RUL
    normal_temp = np.random.normal(60, 5, n_normal)
    normal_vib = np.random.normal(3, 0.5, n_normal)
    normal_pressure = np.random.normal(100, 10, n_normal)
    normal_rpm = np.random.normal(2000, 100, n_normal)
    
    # Energy: Base consumption + small noise
    # Formula: E ~ (RPM * Torque_factor) + Heat_loss
    normal_energy = (normal_rpm * 0.05) + (normal_temp * 0.1) + np.random.normal(0, 2, n_normal)
    
    # RUL: Long life remaining (50 to 200 days)
    normal_rul = np.random.randint(50, 200, n_normal)

    # ==========================================
    # 2. Warning State (Degrading)
    # ==========================================
    # Higher readings, inefficient energy, medium RUL
    warn_temp = np.random.normal(75, 5, n_warning)
    warn_vib = np.random.normal(5, 1, n_warning)
    warn_pressure = np.random.normal(115, 12, n_warning)
    warn_rpm = np.random.normal(2200, 150, n_warning)
    
    # Energy: Higher due to friction/inefficiency (+ offset)
    warn_energy = (warn_rpm * 0.06) + (warn_temp * 0.15) + np.random.normal(5, 3, n_warning)
    
    # RUL: Medium life remaining (10 to 50 days)
    warn_rul = np.random.randint(10, 50, n_warning)

    # ==========================================
    # 3. Critical Failure State
    # ==========================================
    # Extreme readings, spikes, low RUL
    fail_temp = np.random.normal(95, 8, n_failure)
    fail_vib = np.random.normal(8, 2, n_failure)
    fail_pressure = np.random.normal(130, 15, n_failure)
    fail_rpm = np.random.normal(2600, 300, n_failure)
    
    # Energy: Significant spike (high load/friction)
    fail_energy = (fail_rpm * 0.08) + (fail_temp * 0.2) + np.random.normal(10, 5, n_failure)
    
    # RUL: Near zero (0 to 10 days)
    fail_rul = np.random.randint(0, 10, n_failure)

    # ==========================================
    # Combine & Process
    # ==========================================
    temperature = np.concatenate([normal_temp, warn_temp, fail_temp])
    vibration = np.concatenate([normal_vib, warn_vib, fail_vib])
    pressure = np.concatenate([normal_pressure, warn_pressure, fail_pressure])
    rpm = np.concatenate([normal_rpm, warn_rpm, fail_rpm])
    energy = np.concatenate([normal_energy, warn_energy, fail_energy])
    rul = np.concatenate([normal_rul, warn_rul, fail_rul])
    
    # Target: 0 for Normal & Warning (technically not failed yet), 1 for Failure
    # Note: You could treat Warning as class 1 depending on strategy, 
    # but usually "Failure" implies breakdown. Here we keep 1 = Breakdown.
    failure = np.concatenate([np.zeros(n_normal), np.zeros(n_warning), np.ones(n_failure)])
    
    df = pd.DataFrame({
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'rpm': rpm,
        'energy_kwh': energy,
        'rul_days': rul,
        'failure': failure
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=settings.RANDOM_STATE).reset_index(drop=True)
    
    # Add realistic noise and clip to physical limits
    df['temperature'] = df['temperature'].clip(20, 150)
    df['vibration'] = df['vibration'].clip(0, 20)
    df['pressure'] = df['pressure'].clip(50, 250)
    df['rpm'] = df['rpm'].clip(0, 4000)
    df['energy_kwh'] = df['energy_kwh'].clip(0, 500)
    
    # Save to CSV if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Generated {n_samples} samples with Energy & RUL data at {save_path}")
    
    return df

if __name__ == "__main__":
    # Test generation when running directly
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "sensor_data.csv"
    print(f"Generating data to: {data_path}")
    
    df = generate_sensor_data(n_samples=10000, save_path=data_path)
    
    print("\nDataset Statistics:")
    print(df.describe())
    print(f"\nFailure Distribution:")
    print(df['failure'].value_counts())
    print(f"\nRUL Distribution (Head):")
    print(df['rul_days'].head())
    