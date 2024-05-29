import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Load the generated data from the Excel file
df = pd.read_excel('2024.xlsx')

# Encode the internship categories into integers
internship_mapping = {'CLOUD': 1, 'WEB_DEVELOPMENT': 2, 'DEVOPS': 3, 'CYBER_SECURITY': 4}
df['INTERNSHIP'] = df['INTERNSHIP'].map(internship_mapping)

# Normalize the DSA, DBMS, CNS, CGPA, and Internship values using Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']])
df[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']] = scaled_data

# Define the user's priorities and corresponding weights
priorities = {
    'DSA': 1.0,
    'DBMS': 0.0,
    'CNS': 0.0,
    'CGPA': 0.0,
    'INTERNSHIP': 0.0
}

# Calculate the weighted scores for each subject using dot product of feature vector and weight vector
weights_vector = [priorities['DSA'], priorities['DBMS'], priorities['CNS'], priorities['CGPA'], priorities['INTERNSHIP']]
df['WEIGHTED_SCORE'] = df[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']].dot(weights_vector)

# Separate features and target variable
X = df[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']]
y = df['WEIGHTED_SCORE']

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)


# Predict the weighted scores for all students using the Random Forest model
df['PREDICTED_WEIGHTED_SCORE'] = rf_model.predict(X)
y_true = y
y_predict_rf = df['PREDICTED_WEIGHTED_SCORE']

# Sort the DataFrame based on predicted weighted scores
sorted_df_rf = df.sort_values(by='PREDICTED_WEIGHTED_SCORE', ascending=False)

# Inverse transform the actual scores
actual_scores = scaler.inverse_transform(sorted_df_rf[['DSA', 'DBMS', 'CNS', 'CGPA', 'INTERNSHIP']].values)

# Add the actual scores to the DataFrame
sorted_df_rf[['DSA_ACTUAL', 'DBMS_ACTUAL', 'CNS_ACTUAL', 'CGPA_ACTUAL', 'INTERNSHIP_ACTUAL']] = actual_scores

# Print the top 10 students based on actual weighted scores
top_students_rf = sorted_df_rf.head(10)
print("Top students based on actual weighted scores using Random Forest:")
print(top_students_rf[['DSA_ACTUAL', 'DBMS_ACTUAL', 'CNS_ACTUAL', 'CGPA_ACTUAL', 'INTERNSHIP_ACTUAL']])