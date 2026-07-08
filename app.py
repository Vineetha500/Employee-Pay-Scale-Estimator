import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("best_model.pkl")


st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered"
)


st.title("💼 Employee Salary Classification App")
st.markdown(
    "Predict whether an employee earns >50K or ≤50K based on input features."
)


# Get model feature information
expected_features = model.n_features_in_


st.sidebar.header("Input Employee Details")


# User inputs

age = st.sidebar.slider(
    "Age",
    18,
    65,
    30
)


education = st.sidebar.selectbox(
    "Education Level",
    [
        "Bachelors",
        "Masters",
        "PhD",
        "HS-grad",
        "Assoc",
        "Some-college"
    ]
)


occupation = st.sidebar.selectbox(
    "Job Role",
    [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces"
    ]
)


hours_per_week = st.sidebar.slider(
    "Hours per week",
    1,
    80,
    40
)


experience = st.sidebar.slider(
    "Years of Experience",
    0,
    40,
    5
)



# Create dataframe

input_df = pd.DataFrame(
    {
        "age": [age],
        "education": [education],
        "occupation": [occupation],
        "hours-per-week": [hours_per_week],
        "experience": [experience]
    }
)


st.write("### 🔎 Input Data")
st.write(input_df)



# Preprocessing function

def preprocess_data(df):

    df = df.copy()


    # One hot encoding
    df = pd.get_dummies(df)


    # If model expects more features
    if df.shape[1] < expected_features:

        missing_features = expected_features - df.shape[1]

        for i in range(missing_features):
            df[f"missing_feature_{i}"] = 0


    # If extra features exist
    elif df.shape[1] > expected_features:

        df = df.iloc[:, :expected_features]


    return df



# Single prediction

if st.button("Predict Salary Class"):

    processed_input = preprocess_data(input_df)


    prediction = model.predict(
        processed_input
    )


    if prediction[0] == 1:
        result = ">50K"
    else:
        result = "<=50K"


    st.success(
        f"✅ Prediction: {result}"
    )



# Batch prediction

st.markdown("---")

st.subheader("📂 Batch Prediction")


uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    batch_data = pd.read_csv(uploaded_file)


    st.write(
        "Uploaded data preview:",
        batch_data.head()
    )


    processed_batch = preprocess_data(
        batch_data
    )


    batch_preds = model.predict(
        processed_batch
    )


    batch_data["PredictedClass"] = batch_preds


    batch_data["PredictedClass"] = batch_data[
        "PredictedClass"
    ].replace(
        {
            1: ">50K",
            0: "<=50K"
        }
    )


    st.write(
        "✅ Prediction Results:"
    )

    st.write(
        batch_data.head()
    )


    csv = batch_data.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "Download Predictions CSV",
        csv,
        file_name="salary_predictions.csv",
        mime="text/csv"
    )
