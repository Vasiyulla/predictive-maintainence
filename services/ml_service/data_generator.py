"""
Synthetic Sensor Data Generator
Generates realistic industrial machine sensor data for training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import settings

def generate_sensor_data(n_samples: int = 10000, save_path: Path = None) -> pd.DataFrame:
    """
    Generate synthetic sensor data for predictive maintenance
    
    Features:
    - temperature: Machine operating temperature (°C)
    - vibration: Vibration level (mm/s)
    - pressure: Operating pressure (bar)
    - rpm: Rotational speed (revolutions per minute)
    
    Target:
    - failure: Binary label (0 = normal, 1 = failure)
    """
    np.random.seed(settings.RANDOM_STATE)
    
    # Generate normal operating conditions (80% of data)
    n_normal = int(n_samples * 0.8)
    normal_temp = np.random.normal(60, 10, n_normal)
    normal_vibration = np.random.normal(3, 1, n_normal)
    normal_pressure = np.random.normal(100, 15, n_normal)
    normal_rpm = np.random.normal(2000, 300, n_normal)
    
    # Generate failure conditions (20% of data)
    n_failure = n_samples - n_normal
    
    # Failures have extreme values
    failure_temp = np.random.normal(85, 8, n_failure)  # Higher temperature
    failure_vibration = np.random.normal(7, 1.5, n_failure)  # Higher vibration
    failure_pressure = np.random.normal(130, 10, n_failure)  # Higher pressure
    failure_rpm = np.random.normal(2500, 200, n_failure)  # Higher RPM
    
    # Combine data
    temperature = np.concatenate([normal_temp, failure_temp])
    vibration = np.concatenate([normal_vibration, failure_vibration])
    pressure = np.concatenate([normal_pressure, failure_pressure])
    rpm = np.concatenate([normal_rpm, failure_rpm])
    failure = np.concatenate([np.zeros(n_normal), np.ones(n_failure)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'rpm': rpm,
        'failure': failure
    })
    
    # Shuffle data
    df = df.sample(frac=1, random_state=settings.RANDOM_STATE).reset_index(drop=True)
    
    # Add some noise and edge cases
    df['temperature'] = df['temperature'].clip(20, 120)
    df['vibration'] = df['vibration'].clip(0, 15)
    df['pressure'] = df['pressure'].clip(50, 200)
    df['rpm'] = df['rpm'].clip(500, 3500)
    
    # Save to CSV if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Generated {n_samples} samples and saved to {save_path}")
    
    return df

if __name__ == "__main__":
    # Generate training data
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "sensor_data.csv"
    df = generate_sensor_data(n_samples=10000, save_path=data_path)
    
    print("\nDataset Statistics:")
    print(df.describe())
    print(f"\nFailure Distribution:")
    print(df['failure'].value_counts())