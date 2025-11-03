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
    sex_male = 1 if sex == "Male" else 0

with col2:
    st.subheader("🎫 Travel Info")
    fare = st.number_input("Fare (£)", 0.0, 520.0, 32.0)
    embarked = st.selectbox("Port", ["Southampton", "Cherbourg", "Queenstown"])
    deck = st.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "M"], index=7)

st.subheader("👨‍👩‍👧‍👦 Family")
col3, col4 = st.columns(2)
sibsp = col3.number_input("Siblings/Spouses", 0, 8, 0)
parch = col4.number_input("Parents/Children", 0, 6, 0)
family_size = sibsp + parch + 1

# Encoding
embarked_Q = 1 if embarked == "Queenstown" else 0
embarked_S = 1 if embarked == "Southampton" else 0
deck_dict = {d: 1 if deck == d else 0 for d in ['B', 'C', 'D', 'E', 'F', 'G', 'M']}

# Predict button
if st.button("🔮 PREDICT", type="primary"):
    # Prepare input
    input_df = pd.DataFrame({
        'Pclass': [pclass], 'Age': [age], 'Fare': [fare],
        'Family Size': [family_size], 'Sex_male': [sex_male],
        'Embarked_Q': [embarked_Q], 'Embarked_S': [embarked_S],
        **{f'Deck_{k}': [v] for k, v in deck_dict.items()}
    })
    
    # Scale
    input_df[['Age', 'Fare', 'Family Size']] = scaler.transform(
        input_df[['Age', 'Fare', 'Family Size']]
    )
    
    # Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    # Results
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
    
    # Chart
    st.bar_chart(pd.DataFrame({
        'Outcome': ['Not Survived', 'Survived'],
        'Probability': proba
    }).set_index('Outcome'))

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("**Model:** Logistic Regression")
    st.write("**Accuracy:** ~80.5%")
    st.write("**ROC AUC:** ~87.8%")
    st.markdown("---")
    st.write("**Developer:** [Your Name]")
