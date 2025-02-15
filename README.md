# 🎯 IoT Predictive Maintenance App
Welcome to the IoT Predictive Maintenance App repository! This project demonstrates the development of a predictive maintenance solution for IoT devices using machine learning. Below, you'll find an overview of the project, the tech stack used, and instructions on how to set it up and run it on your local system.

## 🌟 Project Overview
The goal of this project is to predict equipment failure in IoT devices using sensor data. Key objectives include:

Data Cleaning: Handling missing values and standardizing sensor data.

Feature Engineering: Creating new features like rolling averages and rate of change.

Machine Learning: Using XGBoost to predict equipment failure.

Visualization: Generating visualizations to understand sensor data trends and model performance.

Deployment: Building a user-friendly Streamlit app for real-time predictions.

## 💻 Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![OpenSea](https://img.shields.io/badge/OpenSea-%232081E2.svg?style=for-the-badge&logo=opensea&logoColor=white)![image](https://github.com/user-attachments/assets/c13ebe34-8873-4634-ac5f-31ec7854607b)



## 🤖 Machine Learning Algorithms
# 🌲 XGBoost (Extreme Gradient Boosting)
    XGBoost is a powerful ensemble learning algorithm used for both classification and regression tasks. In this project, it is used to predict equipment failure based on sensor data.

    Key Characteristics:

    - Handles missing values and outliers effectively.

    - Captures complex, non-linear relationships in the data.

    - Provides feature importance for interpretability.

    - Optimized for performance and scalability.

# 🔍 SMOTE (Synthetic Minority Over-sampling Technique)
    SMOTE is used to handle imbalanced datasets by generating synthetic samples of the minority class (equipment failure).

    Key Characteristics:

    - Balances the dataset by oversampling the minority class.

    - Improves model performance on imbalanced data.

# ⏳ Timeframe
    This project was completed over a period of 7 days, including the following phases:

    - Day 1: Data exploration, cleaning, and feature engineering.

    - Day 2: Model development using XGBoost.

    - Day 3: Hyperparameter tuning and evaluation.

    - Day 4: Building the Streamlit app interface.

    - Day 5: Adding visualizations and theme customization.

    - Day 6: Code optimization and documentation.

    - Day 7: Final testing and deployment.

## 🗂 Project Structure
![Diagram](https://github.com/smooth-glitch/Edgesense/blob/main/Images/project_structure.png)


## ⚙️ Setup Instructions
To set up this project on your local system, follow these steps:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/smooth-glitch/EgeSense.git
    cd Predictive analysis for IOT devices

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt

3. **Run the Streamlit App**

    ```bash
    streamlit run app.py

4. **Access the App**
   Open your browser and navigate to http://localhost:8501.
   Input sensor data and click Predict to see the results.

### 📁 Dataset
The dataset used in this project is the NASA Turbofan Engine Degradation Simulation Dataset. It contains sensor data from aircraft engines and is widely used for predictive maintenance tasks.

Dataset Source: NASA Prognostics Data Repository


### 📊 App Screenshots
![Diagram](https://github.com/smooth-glitch/Edgesense/blob/main/Images/app_sc.png)

### 🖥️ Main Interface
![Diagram](https://github.com/smooth-glitch/Edgesense/blob/main/Images/app-interface.png)

### 📈 Feature Importance
![Diagram](https://github.com/smooth-glitch/Edgesense/blob/main/Images/feature_importance.png)

### 🤝 Contribution
    If you'd like to contribute to this project, please follow these steps:
        - Fork the repository.
        - Create a new branch for your feature or bug fix.
        - Commit your changes.
        - Submit a pull request with a detailed description of your changes.

### 📜 License
    This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.

### 📧 Contact
    For any questions or feedback, feel free to reach out:

    Email: arjunsridhar445@gmail.com

### ☕ Support Me
    If you find this project useful, consider supporting me:
