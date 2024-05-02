import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('RandomForestClassifierr_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Define the mapping of encoded labels to career aspirations
career_aspirations_mapping = {
    9: 'Lawyer', 6: 'Doctor', 8: 'Government Officer', 1: 'Artist', 
    15: 'Unknown', 12: 'Software Engineer', 14: 'Teacher', 3: 'Business Owner', 
    11: 'Scientist', 2: 'Banker', 16: 'Writer', 0: 'Accountant', 
    5: 'Designer', 4: 'Construction Engineer', 7: 'Game Developer', 
    13: 'Stock Investor', 10: 'Real Estate Developer'
}

# Define the feature names
feature_names = ['part_time_job_encoded', 'absence_days',
                 'extracurricular_activities','weekly_self_study_hours',
                 'math_score','history_score', 'physics_score', 'chemistry_score',
                 'biology_score','english_score', 'geography_score']

# Function to get user input
def get_user_input():
    user_input = {}
    for feature in feature_names:
        if feature == 'part_time_job_encoded':
            value = st.radio(f"Does the student have a part-time job?", ('Yes', 'No'))
            value = 1 if value.lower() == 'yes' else 0  # Convert boolean to numeric

        elif feature == 'extracurricular_activities':
            value = st.radio("Does the student participate in extracurricular activities?", ('Yes', 'No'))
            value = 1 if value.lower() == 'yes' else 0  # Convert boolean to numeric

        elif feature in ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']:
            value = st.slider(f"Enter value for {feature}:", min_value=0, max_value=100, step=1)

        else:
            value = st.number_input(f"Enter value for {feature}:")

        user_input[feature] = value

    return user_input

# Streamlit app
def main():
    st.title("Career Aspiration Predictor")
    st.write("This app predicts the career aspirations of students based on their input.")

    # Get user input
    st.sidebar.header("User Input")
    user_input = get_user_input()

    # Submit button
    if st.button("Submit"):
        # Convert user input to DataFrame
        user_df = pd.DataFrame(user_input, index=[0])

        # Ensure the feature names are in the same order as during model fitting
        user_df = user_df[feature_names]

        # Calculate total score
        user_df['total_score'] = user_df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].sum(axis=1)

        # Calculate average score
        user_df['average_score'] = user_df['total_score'] / 7

        # Make predictions
        predictions = rf_model.predict(user_df)

        # Convert the predicted labels to their string representations
        predicted_career_aspirations = [career_aspirations_mapping[prediction] for prediction in predictions]

        # Extracting total score and average score as scalars
        total_score = user_df['total_score'].values[0]
        average_score = user_df['average_score'].values[0]

        # Display results
        st.header("Prediction Results")
        st.write(f"Total Score: {total_score}")
        st.write(f"Average Score: {average_score}")
        st.write(f"Predicted Career Aspiration: {predicted_career_aspirations[0]}")

if __name__ == "__main__":
    main()
