"""
Data Generation package for GA Predictive Maintenance System.

This package contains modules for generating synthetic sensor data and
maintenance logs to test and validate the PdM system.
"""

from data_generation.synthetic_sensor_generator import (
    SyntheticSensorDataGenerator,
    generate_synthetic_data
)

__all__ = ['SyntheticSensorDataGenerator', 'generate_synthetic_data']
