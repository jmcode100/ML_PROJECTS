import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load(r"C:\Users\jiten\Desktop\Projectml1\model_joblib_test")

# Streamlit title
st.title("Insurance Cost Prediction")

# User input fields
st.header("Enter the following details:")

# Inputs for age, sex, bmi, children, smoker, and region
age = st.number_input("Enter Your Age", min_value=0, max_value=100, value=40)
sex = st.selectbox("Male Or Female [1/0]", options=[0, 1], index=1)  # 1: Male, 0: Female
bmi = st.number_input("Enter Your BMI Value", min_value=10.0, max_value=60.0, value=40.3)
children = st.number_input("Enter Number of Children", min_value=0, max_value=10, value=4)
smoker = st.selectbox("Smoker Yes/No [1/0]", options=[0, 1], index=1)  # 1: Smoker, 0: Non-Smoker
region = st.selectbox("Region [1-4]", options=[1, 2, 3, 4], index=2)  # Example region selection

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Create a DataFrame with the user inputs
    columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    df_new = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=columns)

    # Make the prediction
    result = model.predict(df_new)

    # Display the result
    st.subheader(f"Predicted Insurance Cost: ${result[0]:,.2f}")

