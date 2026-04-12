# ⚡ SmartGrid Enterprise AI: V3 Predictive Optimizer

<img width="1538" height="801" alt="chrome_fiowjXscjn" src="https://github.com/user-attachments/assets/cdb88f0c-22c1-4f10-acf4-e435de1aa85d" />
<img width="1543" height="751" alt="chrome_xZGXwhDfyL" src="https://github.com/user-attachments/assets/70dbead3-7eeb-44c7-9d6e-bf3a1978ef91" />
<img width="1540" height="768" alt="chrome_SimFoyqCCv" src="https://github.com/user-attachments/assets/9751db67-b051-487f-b9cd-344bb4585399" />
<img width="1920" height="879" alt="chrome_sq34IHRF52" src="https://github.com/user-attachments/assets/7b77fbd8-0c8c-4c7b-89de-b138de12d2aa" />
<img width="1919" height="880" alt="chrome_uP9TBuigzz" src="https://github.com/user-attachments/assets/a4cd040a-e737-4e78-8025-850f8d9e0b1a" />

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_logo.svg)](YOUR_STREAMLIT_LINK_HERE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SmartGrid Enterprise** is a commercial-grade AI SaaS platform designed to optimize energy consumption for large-scale infrastructure, telecom sites (BTS), and commercial buildings. By utilizing live meteorological telemetry and high-fidelity XGBoost models, the platform automates the decision-making process between Grid, Solar, and Battery storage to minimize OPEX.

---

## 🚀 Live Demo
**Access the Production Environment:** [👉 CLICK HERE TO OPEN THE DASHBOARD](https://smart-grid-ai-mvkzeetwszysd74j2jkwmh.streamlit.app))

---

## 🧠 The AI Engine (V3 Enterprise)
Unlike standard energy calculators that use static multipliers, this platform is powered by a **Gradient Boosted Decision Tree (XGBoost)** trained on over 870,000 data points.

### Key Capabilities:
- **Multi-Tenant Feature Engineering:** The AI natively understands the physics of house size, resident count, and HVAC loads.
- **EV Impact Forecasting:** Predicts the massive 7kW+ demand spikes from Electric Vehicle charging and automates battery discharge to protect the grid.
- **Live Weather Telemetry:** Integrates directly with Open-Meteo API to ingest real-time Temperature, Solar Radiation, and Cloud Cover.
- **Explainable AI (SHAP):** Provides a transparent "Waterfall" analysis for every prediction, showing exactly which features (e.g., Temperature vs. Time of Day) influenced the energy forecast.

## 💼 Business & ROI Features
Designed as a B2B sales tool, the platform includes:
- **20-Year Financial ROI Projection:** A dynamic Plotly engine that calculates the break-even point against a 3% annual utility inflation rate.
- **Interactive "Profit Zone" Visualization:** Visually proves the long-term value of Solar + Battery hardware.
- **ESG & Carbon Reporting:** Real-time calculation of CO2 avoided and "Trees Equivalent" metrics for corporate sustainability reporting.
- **Live Environmental Mapping:** Interactive 3D Earth globe and Windy.com radar integration for site-specific environmental monitoring.

---

## 🛠️ Technology Stack
- **Frontend:** Streamlit (Custom CSS for B2B branding)
- **AI/ML:** XGBoost, Scikit-Learn, SHAP (Explainable AI)
- **Data Science:** Pandas, NumPy, Plotly (Dynamic Financials)
- **APIs:** Open-Meteo (Weather), Streamlit-Geolocation (GPS)

---

## 📦 Local Installation
If you wish to run the development environment locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/fazaldaftan/smart-grid-ai.git](https://github.com/fazaldaftan/smart-grid-ai.git)
   cd smart-grid-ai
