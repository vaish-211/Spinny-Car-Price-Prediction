import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Spinny Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Import Poppins font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* Main content */
    .main {
        font-family: 'Poppins', sans-serif;
        color: #FFFFFF;
        position: relative;
        
    }
    .main-background {
        background-image: url('assets/showroom_bg.jpg');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -2;
    }
    .main-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.4); /* Overlay for text visibility */
        z-index: -1;
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .sidebar img {
        border-radius: 10px;
    }
    /* Buttons */
    .stButton>button {
        background-color: #561381;
        color: #FFFFFF;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2E054E;
    }
    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #2E054E ;
        border: 1px solid #561381;
        border-radius: 8px;
        padding: 10px;
        font-size: 18px;
        color: #FFFFFF;
    }
    .stNumberInput label, .stSelectbox label {
        color: #FFFFFF;
        font-weight: 500;
        font-size: 20px;
    }
    /* Cards */
    .card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        padding: 2px;
        margin: 2px 0;
    }
    /* Headings */
    h1, h2, h3 {
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    h1 {
        font-size: 36px;
    }
    h2 {
        font-size: 28px;
    }
    /* Tabs */
    .stTabs [role="tab"] {
        font-size: 20px !important;
        font-weight: bold;
        color: #FFFFFF;
        background-color: #2E054E;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #561381;
        font-size: 20px !important;
        font-weight: bold;
    }
    
    /* Tableau iframe */
    .tableau-iframe {
        width: 100%;
        height: 800px;
        border: none;
        border-radius: 12px;
        margin: 20px 0;
    }
    
    /* Footer */
    .footer {
        background-color: #2E054E;
        color: #FFFFFF;
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        font-size: 14px;
    }
    .footer a {
        color: #561381;
        text-decoration: none;
    }
    /* Responsive adjustments */
    @media (max-width: 600px) {
        .stButton>button {
            width: 100%;
            font-size: 20px;
        }
        .card {
            padding: 15px;
            margin: 10px;
        }
        h1 {
            font-size: 28px;
        }
        h2 {
            font-size: 20px;
        }
        .tableau-iframe {
            height: 600px;
        }
        .sidebar .sidebar-content {
            padding: 10px;
        }
    }
    @media (min-width: 601px) and (max-width: 1024px) {
        h1 {
            font-size: 32px;
        }
        h2 {
            font-size: 24px;
        }
        .tableau-iframe {
            height: 700px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Background divs for main content (not sidebar)
st.markdown("<div class='main-background'></div>", unsafe_allow_html=True)
st.markdown("<div class='main-overlay'></div>", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load preprocessed data
data = pd.read_csv("data/preprocessed_car_details.csv")

# Sidebar
with st.sidebar:
    if os.path.exists("assets/car_logo.png"):
        st.image("assets/car_logo.png", use_container_width=True)
    else:
        st.markdown("<p style='color: white;'>Logo Missing</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white;'>Spinny Car Price Predictor</h2>", unsafe_allow_html=True)
    st.markdown("Navigate the app:")
    st.markdown("- **Predict Price**: Estimate your car's value.")
    st.markdown("- **Analytics**: Explore price trends.")
    st.markdown("<p style='color: white;'>Built with ❤️ by Vaish</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🚗 Predict Price", "📊 Analytics"])

# Predict Price tab
with tab1:
    st.markdown("<h1>🚗Spinny Car Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("#### Predict Your Car's Price with ML Magic 🧠✨")
    st.markdown("Estimate your car's value with our ML-powered tool!")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
            km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, step=5000, value=30000)
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
        
        with col2:
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            owner = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
            seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)
        
        # Calculate age
        age = 2025 - year
        
        # Median values
        mileage_median = data['mileage'].median()
        engine_median = data['engine'].median()
        max_power_median = data['max_power'].median()
        
        # Numerical features (scaled)
        numeric_inputs = np.array([[km_driven, age, engine_median, max_power_median, mileage_median]])
        numeric_scaled = scaler.transform(numeric_inputs)
        
        # Encode categorical inputs
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
        brand_encoded[brand_cols.index('brand_Maruti')] = 1
        
        # Combine features
        final_input = np.concatenate([
            numeric_scaled[0],
            [trans_encoded],
            [owner_encoded],
            [seats],
            fuel_encoded,
            seller_encoded,
            brand_encoded
        ]).reshape(1, -1)
        
        # Predict button
        if st.button("Predict Price"):
            price = model.predict(final_input)[0]
            st.success(f"Estimated Price: ₹ {price:,.2f}")
            st.balloons()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='footer'>Made with ❤️ by Vaish | Spinny Car Price Predictor</div>", unsafe_allow_html=True)

# Analytics tab
with tab2:
    st.markdown("<h1>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("#### Explore Visual Insights from the Spinny Car Dataset")

    # Dashboard Image instead of Tableau
    st.markdown("### 📈 Overall Dashboard")
    dashboard_path = "assets/Spinny_Car_Dashboard.png"
    if os.path.exists(dashboard_path):
        st.image(dashboard_path, use_container_width=True, caption="Spinny Car Dashboard Overview")
    else:
        st.warning("Dashboard image not found in assets folder!")

    st.markdown("---")

    st.markdown("### 🧩 Additional Visuals")

    # Pie Charts Row
    pie1, pie2, pie3, pie4 = st.columns(4)
    
    with pie1:
        st.image("visuals/fuel_distribution.png", use_container_width=True, caption="Fuel Type Distribution")
        st.markdown("🚗 Diesel and Petrol cars dominate")

    with pie2:
        st.image("visuals/transmission_distribution.png", use_container_width=True, caption="Transmission Types")
        st.markdown("⚙️ Manual is more common")

    with pie3:
        st.image("visuals/owner_distribution.png", use_container_width=True, caption="Ownership Breakdown")
        st.markdown("👤 Most cars are 1st hand")

    with pie4:
        st.image("visuals/seller_type_distribution.png", use_container_width=True, caption="Car Brands Share")
        st.markdown("🏁 Individual Sellers lead")
    
    st.markdown("---")
    
    # Scatter Charts Row
    scatter1, scatter2, scatter3 = st.columns(3)

    with scatter1:
        st.image("visuals/Selling Price vs Mileage.png", use_container_width=True, caption="Mileage matters, but it’s not the only game in town.")
        st.markdown("🚗 Most cars have mileage between 15–25 kmpl, but selling prices vary a lot. Some high-priced outliers exist.")

    with scatter2:
        st.image("visuals/Selling Price vs Engine Size.png", use_container_width=True, caption="Engine size fuels the price rise.")
        st.markdown("🛠️ Bigger engines = higher selling prices. The relationship holds well, with some luxury outliers.")

    with scatter3:
        st.image("visuals/Selling Price vs Max Power.png", use_container_width=True, caption="More power? More price.")
        st.markdown("⚡ Clear positive trend—cars with more max power usually cost more. A few power-packed outliers too.")
    
    st.markdown("---")
    
    # Line Charts Row
    line1, line2, line3 = st.columns(3)

    with line1:
        st.image("visuals/Average Selling Price by Year.png", use_container_width=True, caption="Price trends over time.")
        st.markdown("📈 Average selling prices rise steadily from 2006 to 2020, reflecting newer car values.")

    with line2:
        st.image("visuals/Total KM Driven by Year.png", use_container_width=True, caption="KM usage over years.")
        st.markdown("🚗 Total kilometers driven peak in recent years, indicating higher usage or data volume.")

    with line3:
        st.image("visuals/Average Mileage by Year.png", use_container_width=True, caption="Mileage evolution by year.")
        st.markdown("⛽ Average mileage fluctuates, with slight improvements in efficiency over time.")

    st.markdown("---")

    # Bar Charts and Heatmap Row
    bar1, bar2, heat1 = st.columns(3)

    with bar1:
        st.image("visuals/Average Selling Price by Fuel Type.png", use_container_width=True, caption="Price by fuel type.")
        st.markdown("💧 Diesel and Petrol show higher average selling prices than CNG or LPG.")

    with bar2:
        st.image("visuals/Average Max Power by Owner Type.png", use_container_width=True, caption="Power by ownership.")
        st.markdown("🏁 Test Drive Cars & First owners tend to have cars with higher average max power.")

    with heat1:
        st.image("visuals/correlation_matrix.png", use_container_width=True, caption="Feature correlations.")
        st.markdown("🔗 Strong positive correlations between selling price, engine, and max power.")

    st.markdown("---")
    st.markdown("<div class='footer'>Made with ❤️ by Vaish | Spinny Car Price Predictor</div>", unsafe_allow_html=True)
