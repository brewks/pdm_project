import sqlite3
import numpy as np
from datetime import datetime, timedelta
import random

def generate_degrading_sensor_data(
    db_path, top_params, num_components=10, num_records=1000
):
    """
    Simple synthetic data generator:
    - simulates degrading sensor values over time
    - writes everything into the sensor_data table
    """

    # Basic thresholds for deciding if a reading is "healthy" or not
    thresholds = {
        'oil_press': 30,
        'hyd_press': 40,
        'brake_press': 50,
        'manifold_press': 25,
        'cht': 100,              # °C
        'oil_temp': 90,
        'rpm': 1000,
        'bus_voltage': 11,
        'alternator_current': 15
    }

    # Different sensors update at different rates (in seconds)
    sampling_intervals = {
        'oil_press': 60,
        'cht': 30,
        'rpm': 10,
        'bus_voltage': 60,
        'alternator_current': 30
    }

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    base_time = datetime.now()  # starting timestamp for all samples

    # Loop over synthetic components
    for comp_id in range(1, num_components + 1):
        # Give each synthetic component a random tail number
        tail_number = f"N{np.random.randint(10000, 99999)}"

        # Decide at which record index this component starts to degrade faster
        failure_point = random.randint(int(num_records * 0.5), num_records)   # failure_point could be 50, 62, 88, 100, etc.

        # For each sensor parameter we want to simulate
        for param in top_params:
            # Use the specific sampling interval for this sensor, default to 60s
            interval_sec = sampling_intervals.get(param, 60)
            time_offset = 0  # relative time offset from the base timestamp

            # Generate time-series values for this parameter
            for i in range(num_records):
                # Before failure: smooth, slow degradation
                if i < failure_point:
                    base_val = max((num_records - i) / num_records, 0) # Apply a base value calculation only for iterations before the designated failure point
                # After failure: artificially speed up the drop
                else:
                    base_val = max((num_records - i) / num_records, 0) * 0.5

                # Add a bit of random noise so it doesn't look perfectly linear
                noise = np.random.normal(0, 0.05)
                value = max(base_val + noise, 0) * 100  # scale to roughly 0–100 range

                # Choose a unit that matches the parameter type
                if param in ['oil_press', 'hyd_press', 'brake_press', 'manifold_press']:
                    unit = 'psi'
                elif param in ['cht', 'oil_temp']:
                    unit = '°C'
                elif param == 'rpm':
                    unit = 'rpm'
                elif param == 'bus_voltage':
                    unit = 'volts'
                elif param == 'alternator_current':
                    unit = 'amps'
                else:
                    unit = 'psi'  # default unit if we don't recognize the param

                # Simple health flag: 0 = healthy, 1 = unhealthy
                threshold = thresholds.get(param, 20)  # fallback threshold if missing
                sensor_health = 0 if value >= threshold else 1

                # Build a realistic timestamp based on sampling interval
                timestamp = base_time + timedelta(seconds=time_offset)
                time_offset += interval_sec

                # Insert one row into sensor_data
                cursor.execute("""
                    INSERT INTO sensor_data (
                        tail_number, component_id, parameter,
                        value, unit, timestamp, sensor_health
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    tail_number, comp_id, param, value, unit,
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    sensor_health
                ))

    # Save all inserts and close connection
    conn.commit()
    conn.close()
    print("✅ Inserted synthetic sensor records with degradation, failures, and variable sampling rates.")

