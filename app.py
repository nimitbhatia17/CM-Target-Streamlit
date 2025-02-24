import streamlit as st
import pandas as pd
import numpy as np
import joblib as joblib
import keras
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

# scaling functions
def scaler_function(dataframe):
    standard_scaler = joblib.load('./scalers/standard_scaler.bin')
    dataframe = standard_scaler.transform(dataframe)
    return dataframe

# loading the model and making predictions
def load_predict(dataframe):
    modelX = keras.saving.load_model("./models/model-x.h5")
    modelY = keras.saving.load_model("./models/model-y.h5")
    modelZ = keras.saving.load_model("./models/model-z.h5")
    predictionX = modelX.predict(dataframe)
    predictionY = modelY.predict(dataframe)
    predictionZ = modelZ.predict(dataframe)
    return predictionX, predictionY, predictionZ

# actual website
st.title('CM Coordinates Predictor')

with st.form('Coordinate Details: '):
    PL_L_x = st.number_input('PL_L_x', min_value=0.0, max_value=100.0, value=0.0)
    PL_L_y = st.number_input('PL_L_y', min_value=0.0, max_value=100.0, value=0.0)
    PL_R_x = st.number_input('PL_R_x', min_value=0.0, max_value=100.0, value=0.0)
    PL_R_y = st.number_input('PL_R_y', min_value=0.0, max_value=100.0, value=0.0)
    PM_x = st.number_input('PM_x', min_value=0.0, max_value=100.0, value=0.0)
    PM_y = st.number_input('PM_y', min_value=0.0, max_value=100.0, value=0.0)
    PP_x = st.number_input('PP_x', min_value=0.0, max_value=100.0, value=0.0)

    submitted = st.form_submit_button("Predict")
    if submitted:
        dataframe = pd.DataFrame({
            'PL_L_x': [PL_L_x],
            'PL_L_y': [PL_L_y],
            'PL_R_x': [PL_R_x],
            'PL_R_y': [PL_R_y],
            'PM_x': [PM_x],
            'PM_y': [PM_y],
            'PP_x': [PP_x]
        })
        dataframe = scaler_function(dataframe)
        predictionX, predictionY, predictionZ = load_predict(dataframe)
        st.write(f"X: {predictionX}")
        st.write(f"Y: {predictionY}")
        st.write(f"Z: {predictionZ}")