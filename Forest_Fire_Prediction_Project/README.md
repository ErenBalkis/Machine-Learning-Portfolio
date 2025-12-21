# 🌲 Forest Fire Area Prediction

This project is a Machine Learning study aiming to predict the **burned area** of forest fires in Montesinho Natural Park, Portugal. By utilizing meteorological data and Fire Weather Indices (FWI), the goal is to develop a regression model capable of handling the stochastic nature of wildfires.

## 🎯 Objectives
Forest fires are inherently difficult to predict due to their chaotic nature. This project focuses on:
* **Noise Reduction:** Cleaning and interpreting the raw dataset.
* **Feature Engineering:** Deriving new features based on fire physics to enhance model learning capacity.
* **Model Comparison:** Evaluating various regression models to minimize prediction error.

---

## 👤 My Role & Contributions: Data Engineering

I took charge of the **Data Preprocessing** and **Feature Engineering** pipelines. My primary focus was to transform the raw meteorological data into meaningful inputs that directly improved model performance.

### 🛠️ 1. Data Preprocessing
* **Log Transformation:** Applied `np.log1p` to the target variable (`area`) to reduce high skewness.
* **Categorical Encoding:** Converted `month` and `day` columns into numerical format using **One-Hot Encoding**.
* **Scaling:** Normalized the dataset using `StandardScaler` to prevent model bias derived from varying magnitudes.

### 🚀 2. Feature Engineering
I derived **5 new physics-based features** where raw data was insufficient. These features played a critical role in increasing the model's predictive power:

* **`temp_RH_ratio` (Temp/Humidity Ratio):** Modeled the compounding effect of hot and dry air on fire intensity. *(Ranked as the 5th most important feature by the model).*
* **`is_summer`:** A binary flag identifying high-risk months (July, August, September).
* **`drought_index`:** Combined DMC (Duff Moisture Code) and DC (Drought Code) to represent deep soil dryness.
* **`wind_ISI_impact`:** Modeled the multiplier effect of wind on the Initial Spread Index (ISI).
* **`high_temp_risk`:** Categorized extreme temperatures (>30°C) as a distinct risk factor.

---

## 📊 Models & Results

Four different regression models were tested with Hyperparameter Optimization using **GridSearch**. Due to the high variance and the abundance of "0" values (non-fires or small fires) in the dataset, classical models struggled, while XGBoost prevailed.

| Model | RMSE (Error) | R² Score | Status |
| :--- | :---: | :---: | :--- |
| **XGBoost** | **1.4711** | **+0.015** | ✅ **Best Performance** |
| Random Forest | 1.5071 | -0.033 | Underfitting |
| Linear Regression | 1.5154 | -0.044 | Underfitting |
| SVR (RBF Kernel) | 1.6038 | -0.170 | Poor Performance |

### 🏆 Why XGBoost?
While other models yielded negative $R^2$ scores (performing worse than a simple mean prediction), **XGBoost** was the only algorithm capable of deciphering the chaotic structure of the dataset and achieving a positive score.

---

## 📈 Key Findings
**Feature Importance:**
The custom feature `temp_RH_ratio` outperformed standard literature indices (like wind or specific drought codes), entering the top 5 most critical features. Additionally, the model assigned significant weight to **Tuesday** (`day_tue`), suggesting a potential correlation with human activity patterns.

![Feature Importance Graph](path/to/your/image.png)
*(Please replace the path above with your actual image file)*

---

## 💻 Installation & Usage

To run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    ```

2.  **Install requirements:**
    ```bash
    pip install pandas numpy scikit-learn xgboost seaborn matplotlib
    ```

3.  **Run the main script:**
    ```bash
    python forest_fire_prediction.py
    ```

## 📂 File Structure
* `forestfires.csv`: Original raw dataset.
* `forestfires_processed.csv`: Processed dataset with new engineering features.

## 📝 License & References
* **Dataset:** [UCI Machine Learning Repository - Forest Fires](https://archive.ics.uci.edu/ml/datasets/forest+fires)
* *Cortez, P., & Morais, A. (2007). A Data Mining Approach to Predict Forest Fires using Meteorological Data.*
