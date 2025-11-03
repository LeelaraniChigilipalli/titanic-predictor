import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Load model and scaler ---
# Note: Renamed pickle import to joblib (as used in the logs)
@st.cache_resource
def load_files():
    # Attempt to load model and scaler (using uploaded names)
    model = joblib.load('logistic_model (1).pkl')
    scaler = joblib.load('scaler (1).pkl')
    return model, scaler

model, scaler = load_files()

# --- Feature Mapping and Constants ---
SEX_MAP = {'Male': 1, 'Female': 0}

# The definitive list of features the model was trained on (adjusted to include Deck_T)
FEATURE_NAMES = [
    'Age', 'Fare', 'Family Size', 'Pclass_2', 'Pclass_3', 'Sex_male', 
    'Embarked_Q', 'Embarked_S', 'Deck_B', 'Deck_C', 'Deck_D', 
    'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T'
]

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Probability Predictor")
st.markdown("This application predicts passenger survival based on your trained Logistic Regression model.")
st.markdown("---")

# --- User Input Sidebar ---
with st.sidebar:
    st.header("Passenger Details")
    pclass = st.selectbox("1. Passenger Class (Pclass)", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
    sex = st.selectbox("2. Sex", options=['Male', 'Female'])
    age = st.slider("3. Age", min_value=1, max_value=80, value=30, step=1)
    sibsp = st.number_input("4. Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
    parch = st.number_input("5. Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0)
    fare = st.number_input("6. Fare Paid ($)", min_value=0.0, max_value=512.0, value=30.0, step=5.0)
    deck_option = st.selectbox("7. Deck Location (Cabin Letter)", options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T'], index=7) # Added 'T' as an option
    embarked_option = st.selectbox("8. Port of Embarkation", options=['Southampton (S)', 'Cherbourg (C)', 'Queenstown (Q)'], index=0)

    predict_button = st.button("Predict Survival Probability", type="primary")

# --- Prediction Logic ---

if predict_button:
    # 1. Calculate engineered feature
    family_size = sibsp + parch + 1
    
    # 2. Prepare raw input data dictionary
    raw_input_dict = {name: [0] for name in FEATURE_NAMES}

    # 3. Map user inputs to the exact feature names the model expects
    
    # Map Continuous Features
    raw_input_dict['Age'][0] = age
    raw_input_dict['Fare'][0] = fare
    raw_input_dict['Family Size'][0] = family_size

    # Map Binary/Categorical Features
    if pclass == 2: raw_input_dict['Pclass_2'][0] = 1
    if pclass == 3: raw_input_dict['Pclass_3'][0] = 1
    
    raw_input_dict['Sex_male'][0] = SEX_MAP[sex]
    
    # Embarked
    if 'Q' in embarked_option: raw_input_dict['Embarked_Q'][0] = 1
    if 'S' in embarked_option: raw_input_dict['Embarked_S'][0] = 1
    
    # Deck
    if deck_option != 'M': # M is baseline (all zeros), so only set 1 for others
        deck_col = f'Deck_{deck_option}'
        if deck_col in raw_input_dict:
            raw_input_dict[deck_col][0] = 1

    # Create the DataFrame
    input_df = pd.DataFrame.from_dict(raw_input_dict)
    
    # 4. Apply Scaling (CRITICAL: Only on Age, Fare, Family Size)
    SCALED_COLS = ['Age', 'Fare', 'Family Size']
    input_df[SCALED_COLS] = scaler.transform(input_df[SCALED_COLS])

    # 5. Predict
    # Ensure the DataFrame column order matches the model's expected order 
    # (The simple dict creation should already enforce alphabetical order, but explicit column selection is safest)
    input_df = input_df[FEATURE_NAMES]
    
    probability = model.predict_proba(input_df)[0][1] # Probability of class 1 (Survival)
    prediction = model.predict(input_df)[0]
    
    # --- Display Results ---
    st.subheader("Prediction Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Survival Probability", f"{probability:.2%}")
        
    with col2:
        if prediction == 1:
            st.success("✅ Prediction: PASSENGER SURVIVED")
            st.balloons()
        else:
            st.error("❌ Prediction: PASSENGER DID NOT SURVIVE")
            
    st.markdown("---")
    st.caption("The prediction is based on your trained Logistic Regression model and its learned historical patterns.")
