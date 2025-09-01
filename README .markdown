# Customer Churn Prediction

     ## Overview
     This project predicts customer churn for a telecom company using SQL for data extraction, Python for preprocessing and modeling, and TensorFlow/scikit-learn for machine learning. I added a custom customer lifetime value (CLV) metric and visualized feature importance to identify churn drivers.

     ## Dataset
     - [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or AdventureWorks database (synthetic churn labels).

     ## Tools
     - SQL (SQLAlchemy, pyodbc)
     - Python (Pandas, NumPy, scikit-learn, TensorFlow)
     - Visualization (Matplotlib, Seaborn)

     ## Methodology
     - Extracted customer data using SQL.
     - Preprocessed data with Pandas (handled missing values, scaled features).
     - Built Random Forest and Neural Network models to predict churn.
     - Visualized feature importance and confusion matrix.

     ## Results
     - Random Forest Accuracy: ~80%
     - Neural Network Accuracy: ~78%
     - Key churn drivers: Order count, contract length (custom feature).
     - ![Feature Importance](results/feature_importance.png)

     ## Setup
     1. Install dependencies:
        ```bash
        pip install pandas numpy scikit-learn tensorflow sqlalchemy pyodbc matplotlib seaborn
        ```
     2. Run SQL query:
        ```bash
        sqlcmd -S localhost -d AdventureWorks2022 -i sql/churn_query.sql -o data/churn_data.csv
        ```
     3. Run Python script:
        ```bash
        python scripts/churn_prediction.py
        ```

     ## Folder Structure
     ```
     Customer-Churn-Prediction/
     ├── scripts/
     │   ├── churn_prediction.py
     ├── sql/
     │   ├── churn_query.sql
     ├── data/
     │   ├── churn_data.csv
     ├── results/
     │   ├── feature_importance.png
     │   ├── model_performance.csv
     ├── README.md
     ```

     ## Customizations
     - Added CLV calculation to prioritize high-value customers.
     - Created a confusion matrix to evaluate model performance.

     ## License
     MIT License