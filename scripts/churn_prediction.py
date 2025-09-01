import pandas as pd
     import numpy as np
     from sklearn.model_selection import train_test_split
     from sklearn.preprocessing import StandardScaler
     from sklearn.ensemble import RandomForestClassifier
     from tensorflow import keras
     import matplotlib.pyplot as plt
     import seaborn as sns

     # Load data
     df = pd.read_csv('data/churn_data.csv')

     # Preprocess
     df = df.dropna()
     X = df.drop(['CustomerID', 'Churn'], axis=1)
     y = df['Churn']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Scale features
     scaler = StandardScaler()
     X_train_scaled = scaler.fit_transform(X_train)
     X_test_scaled = scaler.transform(X_test)

     # Random Forest (scikit-learn)
     rf = RandomForestClassifier(random_state=42)
     rf.fit(X_train_scaled, y_train)
     rf_score = rf.score(X_test_scaled, y_test)

     # Neural Network (TensorFlow)
     model = keras.Sequential([
         keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
         keras.layers.Dense(32, activation='relu'),
         keras.layers.Dense(1, activation='sigmoid')
     ])
     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
     model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
     tf_score = model.evaluate(X_test_scaled, y_test)[1]

     # Visualize feature importance
     importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
     plt.figure(figsize=(10, 6))
     sns.barplot(x='Importance', y='Feature', data=importance.sort_values('Importance', ascending=False))
     plt.title('Feature Importance in Churn Prediction')
     plt.savefig('results/feature_importance.png')

     # Save results
     pd.DataFrame({'Model': ['Random Forest', 'Neural Network'], 'Accuracy': [rf_score, tf_score]}).to_csv('results/model_performance.csv')