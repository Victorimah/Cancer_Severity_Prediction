import streamlit as st
import pickle
import numpy as np
import sklearn as sklearn

# Load the machine learning model
def load_model():
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Function to predict cancer severity
def predict_severity(age, shape, margin, density):
    # Create a feature vector from the user inputs
    input_data = np.array([[age, shape, margin, density]])

    # Make a prediction
    prediction = model.predict(input_data)

    return prediction[0]

# Streamlit UI
def main():
    st.title('Predict Cancer Severity')

    st.write("""### Please provide the following information:""")

    # Input fields for age, shape, margin, and density
    age = st.slider("Age (20-100)", 20, 100)
    shape = st.slider("Shape (1-5)", 1, 5)
    margin = st.slider("Margin (1-5)", 1, 5)
    density = st.slider("Density (1-5)", 1, 5)

    if st.button("Predict"):
        # Make a prediction
        prediction = predict_severity(age, shape, margin, density)
        
        # Display the result
        if prediction == 0:
            st.write("The cancer severity is benign.")
        else:
            st.write("The cancer severity is malignant.")

if __name__ == "__main__":
    main()
