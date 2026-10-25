# 🌊 EcoPredict: Predictive Environmental Analytics

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview
**EcoPredict** is a machine-learning project for exploring environmental forecasting with oceanographic measurements. Its current CalCOFI model predicts water temperature from depth, salinity, and dissolved oxygen; it does **not** currently predict microplastic or other contamination levels.

This project was developed to serve as the foundation for a comparative research analysis on the efficacy of various regression algorithms in environmental forecasting.

## 🚀 Key Features
* **Big Data Processing:** Handles and cleans large-scale oceanographic data (CalCOFI dataset with over 800,000 records).
* **Comparative Machine Learning:** Evaluates Multiple Linear Regression, Ridge, Random Forest, and Gradient Boosting algorithms.
* **Automated Metrics:** Automatically computes R², RMSE, and MAE, generating publication-ready charts.
* **Interactive Dashboard:** A Streamlit app compares regression models and makes temperature estimates from oceanographic measurements.
* **Model Artifacts:** The command-line training pipeline exports the top-performing model and scaler as `.pkl` files.

## 🛠️ Technology Stack
* **Machine Learning:** Python, Scikit-Learn, Pandas, NumPy
* **Dashboard & Visualization:** Streamlit, Matplotlib, Seaborn
* **Model Serialization:** Joblib

## 📊 Dataset Context
This model utilizes the **CalCOFI (California Cooperative Oceanic Fisheries Investigations)** dataset. The current prediction target is water temperature (`T_degC`), using `Depthm`, `Salnty`, and `O2ml_L` as input features. Microplastic forecasting is a future direction, not an implemented model capability.

## ⚙️ Installation & Usage

### 1. Clone the repository
~~~bash
git clone https://github.com/log1-codes/EcoPredict-Machine-Learning-for-Aquatic-Microplastic-Forecasting.git
~~~

### 2. Change to the repository directory
~~~bash
cd EcoPredict-Machine-Learning-for-Aquatic-Microplastic-Forecasting
~~~

### 3. Create and activate a virtual environment
~~~bash
python -m venv .venv
~~~

On Windows PowerShell:
~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

On macOS or Linux:
~~~bash
source .venv/bin/activate
~~~

### 4. Install dependencies
~~~bash
pip install -r requirements.txt
~~~

### 5. Add data and launch the dashboard
Download the CalCOFI `bottle.csv` data and either upload it in the dashboard or place it at `ML/bottle.csv`. The CSV must include `Depthm`, `Salnty`, `O2ml_L`, and `T_degC`.
~~~bash
streamlit run app.py
~~~

### 6. Train from the command line (optional)
~~~bash
cd ML
python model.py
~~~
The training script saves `model_comparison_metrics.png`, `scaler.pkl`, and `best_ecopredict_model.pkl` in the `ML` directory.

## 📝 Research Paper Abstract 
*(Draft context for academic submission)*
This project accompanies a comparative analysis titled: *"Predictive Modeling of Aquatic Contamination: Evaluating Machine Learning Approaches for Cost-Effective Environmental Monitoring."* It proposes a software-driven methodology for preliminary environmental monitoring, comparing ensemble methods against traditional linear models to establish the lowest error rate for environmental forecasting.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
