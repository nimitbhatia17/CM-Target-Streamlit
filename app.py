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
st.title('Stereotactic coordinates of centroid of CM nucleus of the thalamus')
st.write('Coordinates of centroid (centre of mass) of Centromedian nucleus from easily identifiable anatomical landmarks from AC-PC transformed T1W MRI image. Ensure the origin (point 0, 0, 0) is the mid-commissural point (MCP). From the axial image at the level of the AC-PC plane, provide the coordinates of the following points')

with st.form('Coordinate Details: '):
    st.markdown('**LEFT postero-lateral corner of the third ventricle**')
    st.image('./resources/images/1.png')
    PL_L_x = st.number_input('x coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 1)
    PL_L_y = st.number_input('y coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 2)

    st.markdown('**RIGHT postero-lateral corner of the third ventricle**')
    st.image('./resources/images/2.png')
    PL_R_x = st.number_input('x coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 3)
    PL_R_y = st.number_input('y coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 4)

    st.markdown('**LEFT medial limit of the putamen**')
    st.image('./resources/images/3.png')
    PM_L_x = st.number_input('x coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 5)
    PM_L_y = st.number_input('y coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 6)

    st.markdown('**RIGHT medial limit of the putamen**')
    st.image('./resources/images/4.png')
    PM_R_x = st.number_input('x coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 7)
    PM_R_y = st.number_input('y coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 8)

    st.markdown('**LEFT posterior limit of the putamen**')
    st.image('./resources/images/5.png')
    PP_L_x = st.number_input('x coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 9)
    PP_L_y = st.number_input('y coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 10)

    st.markdown('**RIGHT posterior limit of the putamen**')
    st.image('./resources/images/6.png')
    PP_R_x = st.number_input('x coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 11)
    PP_R_y = st.number_input('y coordinate (mm)', min_value=-100.0, max_value=100.0, value=0.0, key = 12)

    submitted = st.form_submit_button("Predict")
    if submitted:
        leftDataframe = pd.DataFrame({
            'PL_L_x': [PL_L_x],
            'PL_L_y': [PL_L_y],
            'PL_R_x': [PL_R_x],
            'PL_R_y': [PL_R_y],
            'PM_x': [PM_L_x],
            'PM_y': [PM_L_y],
            'PP_x': [PP_L_x]
        })
        rightDataframe = pd.DataFrame({
            'PL_L_x': [PL_R_x],
            'PL_L_y': [PL_R_y],
            'PL_R_x': [PL_L_x],
            'PL_R_y': [PL_L_y],
            'PM_x': [PM_R_x],
            'PM_y': [PM_R_y],
            'PP_x': [PP_R_x]
        })
        leftDataframe = scaler_function(leftDataframe)
        rightDataframe = scaler_function(rightDataframe)
        leftPredictionX, leftPredictionY, leftPredictionZ = load_predict(leftDataframe)
        rightPredictionX, rightPredictionY, rightPredictionZ = load_predict(rightDataframe)
        st.write(f"Left Hemisphere")
        st.write(f"X: {leftPredictionX}")
        st.write(f"Y: {leftPredictionY}")
        st.write(f"Z: {leftPredictionZ}")
        st.write(f"Right Hemisphere")
        st.write(f"X: {rightPredictionX}")
        st.write(f"Y: {rightPredictionY}")
        st.write(f"Z: {rightPredictionZ}")
    
st.markdown('#### Abstract')
st.markdown('**Objective**: Deep brain stimulation (DBS) of the centromedian nucleus (CM) of the thalamus is a promising treatment for drug-resistant epilepsy, Tourette syndrome, pain and disorders of consciousness, particularly when other surgical options are not feasible. However, the CM is challenging to visualize on routine MRI and atlas based targeting often results in inaccurate electrode placement, affecting surgical outcomes. Inability to visualize and directly target the CM is a barrier to entry for CM-DBS in a resource limited setting.')
st.markdown('**Methods**: We developed and compared accuracy of different machine learning (ML) models to predict the stereotactic coordinates of the CM using input features, which were and coordinates of readily identifiable points from T1-weighted MRI images. Four ML models—Linear Regression (LR), K-Nearest Neighbor (KNN), Support Vector Regression (SVR), and Deep Neural Networks (DNN)—were trained and optimized using 100 MRI scans of healthy subjects and validated in a separate dataset of 20 patients with generalized epilepsy, an indication for CM-DBS. Models were trained to predict coordinates of the centroid of the CM.')
st.markdown('**Results**: DNNs demonstrated the highest accuracy in predicting CM coordinates, with mean Euclidean error of 0.88 ± 0.41 mm in the healthy subjects, and 1.12 ± 0.44 mm in the epilepsy dataset. The LR, SVR, and KNN models all performed similarly, although with higher error rates.')
st.markdown('**Conclusions**: Our study indicates that ML models, particularly Djson, can accurately predict CM coordinates using standard T1-weighted MRI images. This approach reduces the dependency on advanced imaging techniques, making CM-DBS more accessible.')
