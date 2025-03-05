# Semiconductor Predictive Maintenance System

![Python](https://img.shields.io/badge/Python-3.8+-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-red)

## Overview

This project demonstrates a predictive maintenance approach for semiconductor manufacturing equipment using machine learning techniques. The system uses **simulated sensor data** to showcase how anomaly detection can identify potential equipment failures before they occur.

I've implemented a dual-model approach combining Isolation Forest and Autoencoder Neural Networks, which together provide robust detection of developing issues in complex manufacturing environments. The project focuses on time-series feature engineering to capture the temporal patterns that typically precede equipment failures.

## Approach

The system processes sensor data through several key steps:

First, it engineers time-based features from raw sensor readings to capture how measurements change over time. This is critical since equipment failures develop gradually rather than occurring instantly.

Then, it employs two complementary anomaly detection methods: Isolation Forest identifies statistical outliers in the high-dimensional sensor space, while the Autoencoder learns normal operation patterns and detects subtle deviations through reconstruction error. These models are combined in an ensemble approach to balance sensitivity and specificity.

For demonstration purposes, this implementation uses simulated data that mimics patterns observed in semiconductor equipment, including correlations between sensors and characteristic patterns that emerge when equipment begins to degrade.

## Business Context

In semiconductor manufacturing, unplanned equipment downtime can cost hundreds of thousands of dollars per hour in lost production. Traditional maintenance approaches either wait for failures (reactive) or perform unnecessary maintenance on fixed schedules (preventive).

This predictive approach enables condition-based maintenance, which can significantly reduce costs and extend equipment lifetime. When implemented with real sensor data from semiconductor fabs, similar systems have been shown to detect developing issues 8-24 hours before failure, reducing unplanned downtime by 30-40%.

## Usage

The project requires Python 3.8+ with NumPy, Pandas, scikit-learn, and TensorFlow. To run the demonstration:

```bash
pip install -r requirements.txt
python predictive_maintenance.py
