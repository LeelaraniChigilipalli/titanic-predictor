import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢")

# Load model and scaler
@st.cache_resource
def load_files():
    model = pickle.load(open('logistic_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_files()

# Title
st.title("🚢 Titanic Survival Predictor")
st.write("Predict passenger survival using Machine Learning")
st.markdown("---")

# Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Passenger Info")
    pclass = st.selectbox("Class", [1, 2, 3])
    age = st.slider("Age", 0, 80, 30)
    sex = st.radio("Gender", ["Male", "Female"])

with col2:
    st.subheader("🎫 Travel Info")
    fare = st.number_input("Fare (£)", 0.0, 520.0, 32.0)
    embarked = st.selectbox("Port", ["Southampton", "Cherbourg", "Queenstown"])
    deck = st.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "M"], index=7)

st.subheader("👨‍👩‍👧‍👦 Family")
col3, col4 = st.columns(2)
sibsp = col3.number_input("Siblings/Spouses", 0, 8, 0)
parch = col4.number_input("Parents/Children", 0, 6, 0)

# Predict button
if st.button("🔮 PREDICT", type="primary"):
    # Calculate family size
    family_size = sibsp + parch + 1
    
    # Encoding
    sex_male = 1 if sex == "Male" else 0
    embarked_Q = 1 if embarked == "Queenstown" else 0
    embarked_S = 1 if embarked == "Southampton" else 0
    
    # Create Deck dummy variables (IMPORTANT: Keep exact same order as training!)
    deck_B = 1 if deck == "B" else 0
    deck_C = 1 if deck == "C" else 0
    deck_D = 1 if deck == "D" else 0
    deck_E = 1 if deck == "E" else 0
    deck_F = 1 if deck == "F" else 0
    deck_G = 1 if deck == "G" else 0
    deck_M = 1 if deck == "M" else 0
    
    # Create DataFrame with EXACT column names and order from training
    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare],
        'Family Size': [family_size],
        'Sex_male': [sex_male],
        'Embarked_Q': [embarked_Q],
        'Embarked_S': [embarked_S],
        'Deck_B': [deck_B],
        'Deck_C': [deck_C],
        'Deck_D': [deck_D],
        'Deck_E': [deck_E],
        'Deck_F': [deck_F],
        'Deck_G': [deck_G],
        'Deck_M': [deck_M]
    })
    
    # Scale only the numeric features that were scaled during training
    input_df[['Age', 'Fare', 'Family Size']] = scaler.transform(
        input_df[['Age', 'Fare', 'Family Size']]
    )
    
    # Make prediction
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Results")
    
    col5, col6 = st.columns(2)
    with col5:
        if pred == 1:
            st.success("### ✅ SURVIVED")
            st.balloons()
        else:
            st.error("### ❌ DID NOT SURVIVE")
    
    with col6:
        st.metric("Survival Chance", f"{proba[1]:.1%}")
    
    # Probability chart
    st.bar_chart(pd.DataFrame({
        'Outcome': ['Not Survived', 'Survived'],
        'Probability': proba
    }).set_index('Outcome'))

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("**Model:** Logistic Regression")
    st.write("**Accuracy:** ~80.5%")
    st.markdown("---")
    st.write("**Developer:** Leelarani Chigilipalli")
