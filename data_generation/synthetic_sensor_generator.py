"""
Synthetic Sensor Data Generator for GA Predictive Maintenance System.

This module generates high-variability synthetic sensor data that simulates
realistic operational behavior including:
- Temperature drift
- Vibration spikes
- Oil pressure decay
- Gradual degradation curves
- EGT asymmetry
- RPM oscillations
- Noise bursts

The synthetic data is designed to stress-test the ML pipeline and validate
the PdM system's ability to detect degradation patterns and predict failures.

Author: Ndubuisi Chibuogwu
Date: Dec 2024 - July 2025
"""

import numpy as np  # used to generate random numbers and noise
from datetime import datetime, timedelta  # for timestamps
from typing import List, Dict, Optional  # type hinting
from pathlib import Path  # handles file paths
import random  # built-in random generator

# Import all project-level configuration values
from config.settings import (
    TOP_SENSOR_PARAMS,
    SENSOR_UNITS,
    SENSOR_HEALTH_THRESHOLDS,
    SAMPLING_INTERVALS,
    DEFAULT_NUM_COMPONENTS,
    DEFAULT_NUM_RECORDS,            # All parameters imported from /config.settings
    DEGRADATION_RATE_RANGE,
    INITIAL_HEALTH_RANGE,
    NOISE_STABLE_PARAMS,
    NOISE_PRESSURE_PARAMS,
    NOISE_DEFAULT,
    SUDDEN_DROP_PROBABILITY,
    POST_FAILURE_DEGRADATION_FACTOR,
    RANDOM_SEED
)
from config.db_utils import get_connection  # Reusable DB connection helper
from config.logging_config import setup_logger  # Logging setup

# Initialize logger
logger = setup_logger(__name__)  # Creates a named logger for this file


class SyntheticSensorDataGenerator:  # This class encapsulates all logic for generating aviation-like sensor data.
    """
    A class to generate synthetic time-series data for a sensor,
    incorporating trend, seasonality, and noise reminiscent of aircraft sensors.
    """
    # Generator for synthetic sensor data with realistic degradation patterns.
    def __init__(
        self,
        db_path: Path,
        params: Optional[List[str]] = None,
        num_components: int = DEFAULT_NUM_COMPONENTS,
        num_records: int = DEFAULT_NUM_RECORDS
    ):
        
        # Initialize the synthetic sensor data generator.
        self.db_path = db_path  # Path to SQLite database file
        self.params = params or TOP_SENSOR_PARAMS  # Default to configured top parameters
        self.num_components = num_components  # How many aircraft components to simulate
        self.num_records = num_records  # How long each component is simulated for (in datapoints)

        # Set random seed for reproducibility
        np.random.seed(RANDOM_SEED)  # Makes results the same each run
        random.seed(RANDOM_SEED)  # Same for Python's random

        logger.info(
            f"Initialized SyntheticSensorDataGenerator: "
            f"{num_components} components, {num_records} records per param"
        )

    def _get_sensor_unit(self, param: str) -> str:
        
        # Look up whether oil pressure is in psi, temp is °C, etc.
        return SENSOR_UNITS.get(param, 'psi')  # Default to psi if missing

    def _get_noise_level(self, param: str) -> float:
       
        # Some sensors fluctuate a lot, others should stay very stable.
        if param in ['rpm', 'bus_voltage']:
            return NOISE_STABLE_PARAMS  # Very low noise
        elif param in ['oil_press', 'hyd_press']:
            return NOISE_PRESSURE_PARAMS  # Higher noise for pressure systems
        else:
            return NOISE_DEFAULT  # Default noise for most sensors

    def _compute_degradation_value(
        self,
        record_index: int,  # Current record number (acts as time)
        failure_point: int,  # When the component begins failing
        initial_factor: float,  # Starts some components more healthy than others
        degradation_rate: float,  # How fast the sensor value deteriorates
        param: str  # Which sensor we're simulating
    ) -> float:
       
        # Simulate the sensor getting worse as the component ages.
        
        # Compute base degradation value (linear decay)
        if record_index < failure_point:
            # Normal gradual degradation
            base_val = max((self.num_records - record_index * degradation_rate) / self.num_records, 0)
        else:
            # Layman: “After failure begins, the sensor drops faster.         
            base_val = max(
                (self.num_records - record_index * degradation_rate) / self.num_records,
                0
            ) * POST_FAILURE_DEGRADATION_FACTOR

        # Add parameter-specific noise
        noise_std = self._get_noise_level(param)  # Layman: “How shaky this sensor is.”
        noise = np.random.normal(0, noise_std)  # Normal noise

        # Add occasional large drops for pressure sensors (simulating leaks/failures)
        if param in ['oil_press', 'hyd_press'] and np.random.rand() < SUDDEN_DROP_PROBABILITY:
            noise += -0.5  # Sudden pressure drop

        # Apply initial health factor and noise
        value = max(base_val * initial_factor + noise, 0) * 100
        # Layman: “Scale to a 0–100 range and make sure no negative values.”

        return value

    def _classify_sensor_health(self, value: float, param: str) -> int:
        """
        Classify sensor health based on threshold values.

        Layman: “Decide if the sensor reading is still healthy or unsafe.”
        """
        threshold = SENSOR_HEALTH_THRESHOLDS.get(param, 20.0)  # Each parameter has a minimum safe reading
        return 0 if value >= threshold else 1  # healthy = 0, unhealthy = 1

    def generate(self) -> None:
    
        # Generate synthetic sensor data and insert into database.
        logger.info("Starting synthetic sensor data generation...")

        try:
            conn = get_connection(self.db_path)  # Connect to SQLite
            cursor = conn.cursor()
            base_time = datetime.now()  # Start timestamp "now"

            records_inserted = 0

            for comp_id in range(1, self.num_components + 1):
                # Give each component an aircraft ID like N54321
                tail_number = f"N{np.random.randint(10000, 99999)}"

                # Randomize component degradation characteristics
                degradation_rate = np.random.uniform(*DEGRADATION_RATE_RANGE)  # How fast it degrades
                initial_factor = np.random.uniform(*INITIAL_HEALTH_RANGE)  # Start slightly better/worse

                # Randomly determine failure point (50-100% of lifecycle)
                failure_point = random.randint(
                    int(self.num_records * 0.5),
                    self.num_records
                )

                logger.debug(
                    f"Component {comp_id} ({tail_number}): "
                    f"degradation_rate={degradation_rate:.2f}, "
                    f"initial_factor={initial_factor:.2f}, "
                    f"failure_point={failure_point}"
                )

                for param in self.params:
                    # Get sampling interval for this parameter
                    interval_sec = SAMPLING_INTERVALS.get(param, 60)
                    time_offset = 0  # Keeps track of elapsed minutes/seconds

                    for i in range(self.num_records):
                        # Simulate the sensor reading every X seconds.
                        timestamp = base_time + timedelta(seconds=time_offset)
                        time_offset += interval_sec

                        # Compute degraded sensor value
                        value = self._compute_degradation_value(
                            i, failure_point, initial_factor,
                            degradation_rate, param
                        )

                        # Get unit and classify health
                        unit = self._get_sensor_unit(param)  # psi, °C, volts, etc.
                        sensor_health = self._classify_sensor_health(value, param)  # healthy or not

                        # Insert into database
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

                        records_inserted += 1

                if comp_id % 10 == 0:
                    # Layman: “Every 10 components, print progress.”
                    logger.info(f"Generated data for {comp_id}/{self.num_components} components")

            conn.commit()  # Save to DB
            conn.close()

            logger.info(
                f"Successfully inserted {records_inserted:,} synthetic sensor records "
                f"for {self.num_components} components"
            )

        except Exception as e:
            logger.error(f"Failed to generate synthetic sensor data: {e}")
            raise


def generate_synthetic_data(
    db_path: Path,
    params: Optional[List[str]] = None,
    num_components: int = DEFAULT_NUM_COMPONENTS,
    num_records: int = DEFAULT_NUM_RECORDS
) -> None:
  
    # Helper function so users don’t need to create the class manually.
    generator = SyntheticSensorDataGenerator(
        db_path=db_path,  # Path to the SQLite database
        params=params,  # List of sensors to simulate
        num_components=num_components,  # Number of components
        num_records=num_records  # Number of records per component per parameter
    )
    generator.generate()  # Start generating


if __name__ == "__main__":
    """
    Script entry point for standalone execution.

    Layman: “If you run this file directly, generate synthetic data.”
    """
    from config.settings import DATABASE_PATH

    logger.info("=== Synthetic Sensor Data Generation Script ===")

    # Generate synthetic data
    generate_synthetic_data(
        db_path=DATABASE_PATH,
        params=TOP_SENSOR_PARAMS,
        num_components=10,
        num_records=1000
    )

    logger.info("Data generation complete.")
