import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Crop & Fertiliser Recommendation",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop Yield & Fertiliser Recommendation")

st.write(
    "A farmer-friendly machine learning system "
    "for crop and fertiliser recommendation."
)

st.divider()


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("crop_recommendation.csv")


df = load_data()

features = [
    "N",
    "P",
    "K",
    "pH",
    "rainfall",
    "temperature"
]

X = df[features]
y = df["crop"]


# ---------------- TRAIN MODEL ----------------

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    return model, accuracy


model, accuracy = train_model(X, y)


# ---------------- MODE ----------------

st.subheader("👨‍🌾 Choose Your Mode")

mode = st.radio(
    "How would you like to provide farm information?",
    [
        "🌱 Simple Farmer Mode",
        "🧪 Soil Test Mode"
    ]
)


# =================================================
# SIMPLE FARMER MODE
# =================================================

if mode == "🌱 Simple Farmer Mode":

    st.info(
        "You don't need to know N, P, K or pH. "
        "Just answer the simple questions."
    )

    st.subheader("🌾 About Your Farm")

    soil_type = st.selectbox(
        "1. What type of soil do you have?",
        [
            "Black Soil",
            "Red Soil",
            "Sandy Soil",
            "Loamy Soil",
            "Clay Soil",
            "I Don't Know"
        ]
    )

    water = st.selectbox(
        "2. How much water is available?",
        [
            "Low",
            "Medium",
            "High",
            "Rain-fed"
        ]
    )

    season = st.selectbox(
        "3. Which season are you growing in?",
        [
            "Kharif",
            "Rabi",
            "Summer"
        ]
    )

    rainfall_level = st.selectbox(
        "4. How much rainfall does your area usually receive?",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    temperature_level = st.selectbox(
        "5. What is the usual temperature?",
        [
            "Cool",
            "Moderate",
            "Hot"
        ]
    )

    location = st.text_input(
        "📍 Village / District / City",
        placeholder="Example: Anantapur"
    )

    st.divider()

    recommend = st.button(
        "🌾 RECOMMEND CROP & FERTILISER",
        use_container_width=True
    )

    if recommend:

        # Representative values for demonstration

        soil_values = {
            "Black Soil": [80, 40, 40, 7.0],
            "Red Soil": [55, 35, 30, 6.2],
            "Sandy Soil": [40, 25, 25, 6.5],
            "Loamy Soil": [70, 40, 40, 6.8],
            "Clay Soil": [75, 45, 45, 7.0],
            "I Don't Know": [60, 35, 35, 6.5]
        }

        N, P, K, pH = soil_values[soil_type]

        rainfall_values = {
            "Low": 60,
            "Medium": 150,
            "High": 250
        }

        rainfall = rainfall_values[rainfall_level]

        temperature_values = {
            "Cool": 18,
            "Moderate": 25,
            "Hot": 32
        }

        temperature = temperature_values[temperature_level]

        user_data = pd.DataFrame(
            [[
                N,
                P,
                K,
                pH,
                rainfall,
                temperature
            ]],
            columns=features
        )

        # Crop prediction

        predicted_crop = model.predict(user_data)[0]

        # Confidence

        probabilities = model.predict_proba(user_data)[0]
        confidence = max(probabilities) * 100

        # ---------------- FERTILISER RECOMMENDATION ----------------

        crop_data = df[df["crop"] == predicted_crop]

        crop_N = crop_data["N"].median()
        crop_P = crop_data["P"].median()
        crop_K = crop_data["K"].median()

        deficiencies = {
            "Nitrogen": crop_N - N,
            "Phosphorus": crop_P - P,
            "Potassium": crop_K - K
        }

        highest_deficiency = max(
            deficiencies,
            key=deficiencies.get
        )

        if deficiencies[highest_deficiency] <= 0:
            fertilizer = "Balanced NPK fertilizer"

        elif highest_deficiency == "Nitrogen":
            fertilizer = "Nitrogen-rich fertilizer"

        elif highest_deficiency == "Phosphorus":
            fertilizer = "Phosphorus-rich fertilizer"

        else:
            fertilizer = "Potassium-rich fertilizer"

        # ---------------- RESULTS ----------------

        st.success("🌱 Recommendation Generated!")

        st.subheader("🌾 Recommended Crop")

        st.markdown(
            f"# {predicted_crop.title()}"
        )

        st.subheader("🧪 Fertiliser Recommendation")

        st.info(
            f"Recommended: **{fertilizer}**"
        )

        st.subheader("📊 Model Confidence")

        st.progress(
            min(int(confidence), 100)
        )

        st.write(
            f"Model confidence: **{confidence:.1f}%**"
        )

        st.subheader("📋 Your Farm Information")

        result = pd.DataFrame({
            "Information": [
                "Soil Type",
                "Water Availability",
                "Season",
                "Rainfall",
                "Temperature",
                "Location"
            ],
            "Your Input": [
                soil_type,
                water,
                season,
                rainfall_level,
                temperature_level,
                location if location else "Not provided"
            ]
        })

        st.table(result)

        st.warning(
            "Simple Farmer Mode uses representative values "
            "for demonstration. The current training dataset "
            "does not contain soil type, season or water "
            "availability as separate features. For real farm "
            "use, actual soil-test values are recommended."
        )


# =================================================
# SOIL TEST MODE
# =================================================

else:

    st.info(
        "Use this mode if you have a soil test report "
        "or Soil Health Card."
    )

    st.subheader("🧪 Enter Soil Test Values")

    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        value=60.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        value=35.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        value=35.0
    )

    pH = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

    rainfall = st.number_input(
        "Rainfall",
        min_value=0.0,
        value=150.0
    )

    temperature = st.number_input(
        "Temperature",
        value=25.0
    )

    recommend_test = st.button(
        "🌾 RECOMMEND",
        use_container_width=True
    )

    if recommend_test:

        user_data = pd.DataFrame(
            [[
                N,
                P,
                K,
                pH,
                rainfall,
                temperature
            ]],
            columns=features
        )

        predicted_crop = model.predict(
            user_data
        )[0]

        probabilities = model.predict_proba(
            user_data
        )[0]

        confidence = max(probabilities) * 100

        # Fertiliser recommendation

        crop_data = df[
            df["crop"] == predicted_crop
        ]

        crop_N = crop_data["N"].median()
        crop_P = crop_data["P"].median()
        crop_K = crop_data["K"].median()

        deficiencies = {
            "Nitrogen": crop_N - N,
            "Phosphorus": crop_P - P,
            "Potassium": crop_K - K
        }

        highest_deficiency = max(
            deficiencies,
            key=deficiencies.get
        )

        if deficiencies[highest_deficiency] <= 0:
            fertilizer = "Balanced NPK fertilizer"

        elif highest_deficiency == "Nitrogen":
            fertilizer = "Nitrogen-rich fertilizer"

        elif highest_deficiency == "Phosphorus":
            fertilizer = "Phosphorus-rich fertilizer"

        else:
            fertilizer = "Potassium-rich fertilizer"

        # Results

        st.success(
            "🌱 Recommendation Generated!"
        )

        st.subheader("🌾 Recommended Crop")

        st.markdown(
            f"# {predicted_crop.title()}"
        )

        st.subheader("🧪 Recommended Fertiliser")

        st.info(
            f"**{fertilizer}**"
        )

        st.subheader("📊 Model Confidence")

        st.progress(
            min(int(confidence), 100)
        )

        st.write(
            f"{confidence:.1f}%"
        )


# =================================================
# MODEL INFORMATION
# =================================================

with st.expander("ℹ️ About the ML Model"):

    st.write(
        "The system uses a Random Forest classifier "
        "trained on the crop recommendation dataset."
    )

    st.write(
        f"Test accuracy: **{accuracy * 100:.2f}%**"
    )

    st.write(
        "Features used: N, P, K, soil pH, rainfall "
        "and temperature."
    )
