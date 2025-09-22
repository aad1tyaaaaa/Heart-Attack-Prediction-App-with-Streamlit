import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import altair as alt

# Load the trained model
with open('model.sav', 'rb') as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(page_title="Heart Attack Prediction", page_icon="❤️", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    .low-risk {
        background-color: #D4EDDA;
        color: #155724;
    }
    .high-risk {
        background-color: #F8D7DA;
        color: #721C24;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">❤️ Heart Attack Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Predict Your Heart Disease Risk</h2>', unsafe_allow_html=True)

st.write("This app uses machine learning to predict the likelihood of heart disease based on your health parameters. Please fill in the form below with accurate information.")

# Sidebar for additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This prediction is based on a machine learning model trained on the Cleveland Heart Disease dataset.")
    st.write("**Disclaimer:** This is not a substitute for professional medical advice. Consult a healthcare provider for accurate diagnosis.")
    
    st.header("📊 Model Info")
    st.write("Algorithm: Random Forest Classifier")
    st.write("Accuracy: ~85% on test data")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.slider('Age', min_value=1, max_value=120, value=50, help="Your age in years")
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male', help="Your biological sex")
    
    st.subheader("Chest Pain")
    cp = st.selectbox('Chest Pain Type', options=[1, 2, 3, 4], format_func=lambda x: {
        1: 'Typical Angina',
        2: 'Atypical Angina',
        3: 'Non-anginal Pain',
        4: 'Asymptomatic'
    }[x], help="Type of chest pain experienced")

with col2:
    st.subheader("Vital Signs")
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120, help="Resting blood pressure")
    chol = st.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200, help="Serum cholesterol level")
    thalach = st.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150, help="Maximum heart rate during exercise")

st.subheader("Additional Health Parameters")
col3, col4, col5 = st.columns(3)

with col3:
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', help="Fasting blood sugar level")
    restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2], format_func=lambda x: {
        0: 'Normal',
        1: 'ST-T wave abnormality',
        2: 'Left ventricular hypertrophy'
    }[x], help="Resting electrocardiographic results")

with col4:
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', help="Angina induced by exercise")
    oldpeak = st.slider('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression induced by exercise")

with col5:
    slope = st.selectbox('Slope of ST Segment', options=[1, 2, 3], format_func=lambda x: {
        1: 'Upsloping',
        2: 'Flat',
        3: 'Downsloping'
    }[x], help="Slope of the peak exercise ST segment")
    ca = st.selectbox('Major Vessels Colored', options=[0, 1, 2, 3, 4], help="Number of major vessels colored by fluoroscopy")
    thal = st.selectbox('Thalassemia', options=[3, 6, 7], format_func=lambda x: {
        3: 'Normal',
        6: 'Fixed Defect',
        7: 'Reversable Defect'
    }[x], help="Thalassemia type")

# Prediction button
if st.button('🔍 Predict Heart Disease Risk', type='primary'):
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    input_data = features.reshape(1, -1)
    
    # Convert input_data to DataFrame with feature names to avoid warning
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0]
    
    if prediction[0] == 1:
        st.markdown('<div class="prediction-result high-risk">⚠️ High Risk of Heart Disease Detected</div>', unsafe_allow_html=True)
        st.write(f"Probability of heart disease: {prediction_proba[1]*100:.1f}%")
        st.warning("Please consult a healthcare professional for further evaluation.")
    else:
        st.markdown('<div class="prediction-result low-risk">✅ Low Risk of Heart Disease Detected</div>', unsafe_allow_html=True)
        st.write(f"Probability of heart disease: {prediction_proba[1]*100:.1f}%")
        st.success("Keep up the healthy lifestyle!")
    
    # Interactive Charts Section
    st.header("📊 Risk Analysis Dashboard")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Gauge", "Feature Importance", "Risk Factors", "Health Insights"])

    with tab1:
        st.subheader("Heart Disease Risk Gauge")
        # Gauge chart for risk probability
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_proba[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        st.subheader("Feature Importance Analysis")
        # Interactive bar chart for feature importance
        feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting BS', 'Resting ECG', 'Max HR', 'Exercise Angina', 'ST Depression', 'Slope', 'Major Vessels', 'Thalassemia']
        importance = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)

        fig_bar = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance in Heart Disease Prediction',
            color='Importance',
            color_continuous_scale='RdYlGn_r'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Your Risk Factors")
        # Radar chart for user's risk factors
        user_values = [age/120, sex, cp/4, trestbps/250, chol/600, fbs, restecg/2, thalach/220, exang, oldpeak/10, slope/3, ca/4, thal/7]
        categories = ['Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol', 'Blood Sugar', 'ECG', 'Heart Rate', 'Angina', 'ST Depression', 'Slope', 'Vessels', 'Thalassemia']

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_values,
            theta=categories,
            fill='toself',
            name='Your Values',
            line_color='red'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Your Health Parameters (Normalized)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab4:
        st.subheader("Health Insights")
        # Create health insights based on user's data
        insights = []

        if age > 50:
            insights.append("⚠️ Age is a significant risk factor for heart disease")
        if chol > 240:
            insights.append("⚠️ High cholesterol levels detected")
        if trestbps > 140:
            insights.append("⚠️ Elevated blood pressure")
        if thalach < 120:
            insights.append("⚠️ Low maximum heart rate may indicate risk")
        if cp == 4:
            insights.append("⚠️ Asymptomatic chest pain is concerning")
        if exang == 1:
            insights.append("⚠️ Exercise-induced angina detected")

        if not insights:
            insights.append("✅ Your parameters are within normal ranges")

        for insight in insights:
            st.write(insight)

        # Altair chart for risk distribution
        risk_data = pd.DataFrame({
            'Risk Level': ['Low Risk', 'Moderate Risk', 'High Risk'],
            'Percentage': [100 - prediction_proba[1]*100, prediction_proba[1]*100/2, prediction_proba[1]*100/2]
        })

        chart = alt.Chart(risk_data).mark_arc().encode(
            theta=alt.Theta(field="Percentage", type="quantitative"),
            color=alt.Color(field="Risk Level", type="nominal",
                          scale=alt.Scale(domain=['Low Risk', 'Moderate Risk', 'High Risk'],
                                        range=['green', 'yellow', 'red'])),
            tooltip=['Risk Level', 'Percentage']
        ).properties(title="Risk Distribution")

        st.altair_chart(chart, use_container_width=True)
