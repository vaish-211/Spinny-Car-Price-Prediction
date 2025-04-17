import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Set page config for wide layout and title
st.set_page_config(
    page_title="App1 Spinny Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsiveness and styling
st.markdown("""
    <style>
    .main {
        background-image: url('assets/showroom_bg.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        padding: 20px;
    }
    .main::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.2); /* Slight whitish overlay for text visibility */
        z-index: -1;
    }
    .reportview-container .main .block-container {
        background: transparent;
    }
    .stButton>button {
        background-color: #1e90ff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #104e8b;
    }
    .stNumberInput, .stSelectbox {
        background-color: #000000;
        border-radius: 5px;
        border: 1px solid #d1d1d1;
        padding: 5px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .sidebar img {
        border-radius: 10px;
    }
    @media (max-width: 600px) {
        .stNumberInput, .stSelectbox {
            font-size: 14px;
        }
        .stButton>button {
            width: 100%;
            font-size: 14px;
        }
        img {
            width: 100%;
        }
    }
    @media (min-width: 601px) and (max-width: 1024px) {
        .stNumberInput, .stSelectbox {
            font-size: 16px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load preprocessed data
data = pd.read_csv("data/preprocessed_car_details.csv")

# Sidebar
with st.sidebar:
    st.image("assets/car_logo.png", use_column_width=True)
    st.markdown("<h2 style='color: white;'>Spinny Car Price Predictor</h2>", unsafe_allow_html=True)
    st.markdown("Navigate the app:")
    st.markdown("- **Predict Price**: Estimate your car's value.")
    st.markdown("- **Data Analysis**: Explore car price trends.")
    st.markdown("<p style='color: white;'>Built with ❤️ by Vaish</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🚗 Predict Price", "📊 Data Analysis"])

# Predict Price tab
with tab1:
    st.markdown("### Predict Your Car's Price with ML Magic 🧠✨")

    col1, col2 = st.columns([1, 1])
    with col1:
        year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, step=5000, value=30000)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])

    with col2:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)

    age = 2025 - year

    mileage_median = data['mileage'].median()
    engine_median = data['engine'].median()
    max_power_median = data['max_power'].median()

    numeric_inputs = np.array([[km_driven, age, engine_median, max_power_median, mileage_median]])
    numeric_scaled = scaler.transform(numeric_inputs)

    trans_encoded = 0 if transmission == "Manual" else 1

    owner_map = {
        'First Owner': 0,
        'Second Owner': 1,
        'Third Owner': 2,
        'Fourth & Above Owner': 3
    }
    owner_encoded = owner_map[owner]

    fuel_dict = {
        'Petrol': [0, 0, 1],
        'Diesel': [1, 0, 0],
        'CNG': [0, 0, 0],
        'LPG': [0, 1, 0]
    }
    fuel_encoded = fuel_dict[fuel_type]

    seller_encoded = [1, 0]  # Default to Individual

    brand_cols = [
        'brand_Ashok', 'brand_Audi', 'brand_BMW', 'brand_Chevrolet', 'brand_Daewoo', 'brand_Datsun',
        'brand_Fiat', 'brand_Force', 'brand_Ford', 'brand_Honda', 'brand_Hyundai', 'brand_Isuzu',
        'brand_Jaguar', 'brand_Jeep', 'brand_Kia', 'brand_Land', 'brand_Lexus', 'brand_MG',
        'brand_Mahindra', 'brand_Maruti', 'brand_Mercedes-Benz', 'brand_Mitsubishi', 'brand_Nissan',
        'brand_Opel', 'brand_Renault', 'brand_Skoda', 'brand_Tata', 'brand_Toyota', 'brand_Volkswagen',
        'brand_Volvo'
    ]
    brand_encoded = [0] * len(brand_cols)
    brand_encoded[brand_cols.index('brand_Maruti')] = 1  # Default to Maruti

    final_input = np.concatenate([
        numeric_scaled[0],
        [trans_encoded],
        [owner_encoded],
        [seats],
        fuel_encoded,
        seller_encoded,
        brand_encoded
    ]).reshape(1, -1)

    if st.button("Predict Price 💰"):
        price = model.predict(final_input)[0]
        st.success(f"Estimated Price: ₹ {price:,.2f}")
        st.balloons()

# Data Analysis tab
with tab2:
    st.markdown("### Explore Car Price Trends 📈")
    st.write("Understand the factors affecting used car prices with these insights.")

    st.subheader("Price Distribution")
    st.image("visuals/price_distribution.png", caption="Distribution of car prices in the dataset.", use_column_width=True)
    st.write("Most cars are priced between ₹1L–₹10L, with a right skew indicating luxury cars.")

    st.subheader("Price vs. Year")
    st.image("visuals/price_vs_year.png", caption="How car age affects price.", use_column_width=True)
    st.write("Newer cars generally have higher prices, with some older models retaining value.")

    st.subheader("Correlation Matrix")
    st.image("visuals/correlation_matrix.png", caption="Relationships between numerical features.", use_column_width=True)
    st.write("Strong correlations exist between price, engine size, and max power.")