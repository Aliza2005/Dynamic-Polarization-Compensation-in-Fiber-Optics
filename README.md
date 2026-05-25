# Dynamic-Polarization-Compensation-in-Fiber-Optics
Real-time Active Disturbance Rejection and State of Polarization (SOP) Tracking System utilizing an advanced Hybrid Machine Learning &amp; Hill Climbing control framework for fiber optic networks.
# Real-Time Optical State of Polarization (SOP) Stabilization Platform

This repository provides a high-speed, multi-threaded control framework to track and stabilize the State of Polarization (SOP) in dynamic fiber optic environments. The system interfaces a Thorlabs PAX1000 Polarimeter with an Electronic Polarization Controller (EPC) to dynamically compensate for environmental drift.

---

## 🚀 System Architecture & Core Algorithms
The platform implements three distinct polarization tracking methodologies:
1. **Hybrid Optimization Model :** Utilizes Machine Learning-driven prediction models for fast coarse alignment, combined with localized Hill Climbing optimization loops for sub-degree accuracy.
2. **Deterministic PID Loop :** Implements a classical Proportional-Integral-Derivative feedback tracking controller.
3. **Gradient-Descent Hill Climb:** An incremental optimization framework for active tracking.

---

## 🛠️ Prerequisites & Driver Requirements
Before executing the core tracking scripts, you must install the official manufacturer hardware drivers on your host system:
* **Optical Sensing:** [Thorlabs Optical Power Monitor (OPM) / VISA Drivers](https://www.thorlabs.com)
* **Actuator Control:** [FTDI D2XX Direct Drivers](https://ftdichip.com/drivers/d2xx-drivers/)

### Required Python Libraries
Install the necessary package dependencies via `pip`:
```bash
pip install pyvisa ftd2xx PyQt5 numpy scipy matplotlib scikit-learn
