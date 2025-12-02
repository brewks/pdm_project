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

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import random

from config.settings import (
    TOP_SENSOR_PARAMS,
    SENSOR_UNITS,
    SENSOR_HEALTH_THRESHOLDS,
    SAMPLING_INTERVALS,
    DEFAULT_NUM_COMPONENTS,
    DEFAULT_NUM_RECORDS,            # All variables imported from /config.settings
    DEGRADATION_RATE_RANGE,
    INITIAL_HEALTH_RANGE,
    NOISE_STABLE_PARAMS,
    NOISE_PRESSURE_PARAMS,
    NOISE_DEFAULT,
    SUDDEN_DROP_PROBABILITY,
    POST_FAILURE_DEGRADATION_FACTOR,
    RANDOM_SEED
)
from config.db_utils import get_connection
from config.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class SyntheticSensorDataGenerator: # This class encapsulates all logic for generating aviation-like sensor data.   
    
    # Generator for synthetic sensor data with realistic degradation patterns.
    def __init__(
        self,
        db_path: Path,
        params: Optional[List[str]] = None,
        num_components: int = DEFAULT_NUM_COMPONENTS,
        num_records: int = DEFAULT_NUM_RECORDS
    ):
        
        # Initialize the synthetic sensor data generator.
        self.db_path = db_path # Path to the SQLite database
        self.params = params or TOP_SENSOR_PARAMS # List of sensor parameters to generate.
        self.num_components = num_components # Number of aircraft components to generate data for
        self.num_records = num_records # Number of sensor records per component per parameter

        # Set random seed for reproducibility
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        logger.info(
            f"Initialized SyntheticSensorDataGenerator: "
            f"{num_components} components, {num_records} records per param"
        )

    def _get_sensor_unit(self, param: str) -> str:
        
        # Get the unit of measurement for a sensor parameter.
        return SENSOR_UNITS.get(param, 'psi')  # Default to psi

    def _get_noise_level(self, param: str) -> float:
       
        # Determine the noise level for a given sensor parameter.
        if param in ['rpm', 'bus_voltage']:
            return NOISE_STABLE_PARAMS
        elif param in ['oil_press', 'hyd_press']:
            return NOISE_PRESSURE_PARAMS
        else:
            return NOISE_DEFAULT

    def _compute_degradation_value(
        self,
        record_index: int, # Current record number (time index)
        failure_point: int, # Record index where accelerated degradation begins
        initial_factor: float, # Component's initial health multiplier
        degradation_rate: float, # Rate of degradation over time
        param: str # Sensor parameter name
    ) -> float:
       
        # COMPUTE SENSOR VALUE WITH DEGRADATION OVER TIME.
        
        # Compute base degradation value (linear decay)
        if record_index < failure_point:
            # Normal gradual degradation
            base_val = max((self.num_records - record_index * degradation_rate) / self.num_records, 0)
        else:
            # Accelerated degradation after failure point
            base_val = max(
                (self.num_records - record_index * degradation_rate) / self.num_records,
                0
            ) * POST_FAILURE_DEGRADATION_FACTOR

        # Add parameter-specific noise
        noise_std = self._get_noise_level(param)
        noise = np.random.normal(0, noise_std)

        # Add occasional large drops for pressure sensors (simulating leaks/failures)
        if param in ['oil_press', 'hyd_press'] and np.random.rand() < SUDDEN_DROP_PROBABILITY:
            noise += -0.5  # Sudden pressure drop

        # Apply initial health factor and noise
        value = max(base_val * initial_factor + noise, 0) * 100

        return value

    def _classify_sensor_health(self, value: float, param: str) -> int:
        """
        Classify sensor health based on threshold values.

        Args:
            value: Sensor reading value
            param: Sensor parameter name

        Returns:
            int: Sensor health status (0 = healthy, 1 = unhealthy)
        """
        threshold = SENSOR_HEALTH_THRESHOLDS.get(param, 20.0)
        return 0 if value >= threshold else 1

    def generate(self) -> None:
    
        # Generate synthetic sensor data and insert into database.        
        logger.info("Starting synthetic sensor data generation...")

        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()
            base_time = datetime.now()

            records_inserted = 0

            for comp_id in range(1, self.num_components + 1):
                # Generate random tail number
                tail_number = f"N{np.random.randint(10000, 99999)}"

                # Randomize component degradation characteristics
                degradation_rate = np.random.uniform(*DEGRADATION_RATE_RANGE)
                initial_factor = np.random.uniform(*INITIAL_HEALTH_RANGE)

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
                    time_offset = 0

                    for i in range(self.num_records):
                        # Compute timestamp
                        timestamp = base_time + timedelta(seconds=time_offset)
                        time_offset += interval_sec

                        # Compute degraded sensor value
                        value = self._compute_degradation_value(
                            i, failure_point, initial_factor,
                            degradation_rate, param
                        )

                        # Get unit and classify health
                        unit = self._get_sensor_unit(param)
                        sensor_health = self._classify_sensor_health(value, param)

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
                    logger.info(f"Generated data for {comp_id}/{self.num_components} components")

            conn.commit()
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
  
    # Convenience function to generate synthetic sensor data.
    generator = SyntheticSensorDataGenerator(
        db_path=db_path, # Path to the SQLite database
        params=params, # List of sensor parameters to generate
        num_components=num_components, # Number of components to generate data for
        num_records=num_records # Number of records per component per parameter
    )
    generator.generate()


if __name__ == "__main__":
    """
    Script entry point for standalone execution.

    Usage:
        python synthetic_sensor_generator.py
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
