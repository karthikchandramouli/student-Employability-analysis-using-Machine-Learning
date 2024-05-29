import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def read_data(file_path):
    return pd.read_excel(file_path)

def train_model(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def predict_scores(models, X):
    return {role: model.predict(X) for role, model in models.items()}

def display_students_with_higher_marks(subject, predicted_scores, students_df, job_role):
    print("For {} job role:".format(job_role))
    print("Students with higher marks:")
    print(students_df[students_df[subject] > predicted_scores[job_role][0]])



def main():
    # Read data from Excel file
    df = read_data('data.xlsx')

    # Define job roles and their corresponding prioritized subjects
    job_roles = {
        "Cloud": "CNS",
        "DevOps": "DSA",
        "Cybersecurity": "CNS",
        "Web Development": "DBMS"
    }

    # Train models for each job role
    models = {}
    for job_role, subject in job_roles.items():
        X = df[['CGPA', 'LEET_RANK', 'GraduatedYear']]
        y = df[subject]
        models[job_role] = train_model(X, y)

    # Predict scores for all job roles at once
    predicted_scores = predict_scores(models, [[0, 0, 0]])

    # Display students for each job role
    for job_role, subject in job_roles.items():
        display_students_with_higher_marks(subject, predicted_scores, df, job_role)

if __name__ == "__main__":
    main()
