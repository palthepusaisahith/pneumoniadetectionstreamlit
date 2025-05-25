# pneumonia_app_streamlit.py

import json
from PIL import Image
import io
import os
import numpy as np
import streamlit as st
import base64
import hashlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

@st.cache_resource
def load_prediction_model():
    return load_model("my_model.h5")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_database(json_file_path="data.json"):
    if not os.path.exists(json_file_path):
        with open(json_file_path, "w") as f:
            json.dump({"users": []}, f)

def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if not name or not email or not password:
                st.error("Please fill out all fields.")
                return
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                st.session_state["logged_in"] = True
                st.session_state["user_info"] = user
            else:
                st.error("Passwords do not match.")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
        for user in data["users"]:
            if user["email"] == username and user["password"] == hash_password(password):
                st.session_state["logged_in"] = True
                st.session_state["user_info"] = user
                st.success("Login successful!")
                render_dashboard(user)
                return user
        st.error("Invalid credentials.")
    except Exception as e:
        st.error(f"Login error: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        data = {"users": []} if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0 else json.load(open(json_file_path))

        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": hash_password(password),
            "report": None,
            "precautions": None,
            "xray": None
        }
        data["users"].append(user_info)
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)
        st.success("Account created successfully!")
        return user_info
    except Exception as e:
        st.error(f"Signup error: {e}")

def predict(model, image_path):
    IMAGE_SIZE = 128
    img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return "Normal" if np.argmax(predictions) == 0 else "Pneumonia"

def generate_medical_report(predicted_label):
    info = {
        "Normal": {
            "report": "Great news! The patient's chest X-ray appears normal with no signs of pneumonia. It's important to maintain a healthy lifestyle and continue with regular check-ups to ensure overall well-being.",
            "preventative_measures": [
                "Continue practicing good hygiene habits",
                "Maintain a healthy diet and lifestyle",
                "Regular exercise can further support respiratory health",
            ],
            "precautionary_measures": [
                "Keep up with routine health screenings",
                "Consider scheduling periodic chest X-rays to monitor any changes",
            ],
        },
        "Pneumonia": {
            "report": "It seems like the patient is showing signs of pneumonia in the chest X-ray. Prompt medical attention and treatment are necessary to address the infection and prevent complications. Pneumonia is a serious respiratory condition that can affect lung function, and early intervention is crucial. It's commonly caused by bacteria, viruses, or fungi and may present symptoms such as cough, fever, chills, and difficulty breathing. Depending on severity, patients may require antibiotics, oxygen therapy, or hospitalization.",
            "preventative_measures": [
                "Follow the prescribed treatment plan diligently",
                "Get plenty of rest and stay hydrated",
                "Avoid exposure to smoke and pollutants",
                "Maintain proper hand hygiene to prevent secondary infections",
                "Ensure a balanced diet to support immune function"
            ],
            "precautionary_measures": [
                "Monitor symptoms closely and seek medical help if they worsen",
                "Consider follow-up chest X-rays to track recovery progress",
            ],
        },
    }
    r = info[predicted_label]
    report = f"Medical Report:\n\n{r['report']}\n\nPreventative Measures:\n- " + ",\n- ".join(r["preventative_measures"]) + "\n\nPrecautionary Measures:\n- " + ",\n- ".join(r["precautionary_measures"])
    return report, r["precautionary_measures"]

def save_xray_image(image_file, json_file_path="data.json"):
    try:
        if not image_file or not st.session_state.get("logged_in"):
            return
        with open(json_file_path, "r") as f:
            data = json.load(f)
        for user in data["users"]:
            if user["email"] == st.session_state["user_info"]["email"]:
                img = Image.open(image_file).convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                user["xray"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
                with open(json_file_path, "w") as f:
                    json.dump(data, f, indent=4)
                st.session_state["user_info"]["xray"] = user["xray"]
                break
    except Exception as e:
        st.error(f"Error saving image: {e}")

def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome {user_info['name']}!")
        st.write(f"Sex: {user_info['sex']}, Age: {user_info['age']}")
        if user_info.get("xray"):
            st.image(Image.open(io.BytesIO(base64.b64decode(user_info["xray"]))), caption="Uploaded X-ray")
        if isinstance(user_info.get("precautions"), list):
            st.subheader("Precautions:")
            for precaution in user_info["precautions"]:
                st.write("-", precaution)
        else:
            st.info("Upload a chest X-ray to get a medical report.")
    except Exception as e:
        st.error(f"Dashboard error: {e}")

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    if st.button("Login"):
        check_login(username, password, json_file_path)

def main(json_file_path="data.json"):
    st.sidebar.title("Pneumonia Prediction System")
    page = st.sidebar.radio("Go to", ("Signup/Login", "Dashboard", "Upload Chest X-ray", "View Reports"))

    if page == "Signup/Login":
        if st.radio("Select", ("Login", "Signup")) == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if st.session_state.get("logged_in"):
            render_dashboard(st.session_state["user_info"])
        else:
            st.warning("Please login/signup first.")

    elif page == "Upload Chest X-ray":
        if st.session_state.get("logged_in"):
            st.title("Upload Chest X-ray")
            uploaded_image = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
            if st.button("Upload") and uploaded_image:
                st.image(uploaded_image, use_column_width=True)
                save_xray_image(uploaded_image, json_file_path)
                model = load_prediction_model()
                condition = predict(model, uploaded_image)
                report, precautions = generate_medical_report(condition)
                with open(json_file_path, "r+") as f:
                    data = json.load(f)
                    for user in data["users"]:
                        if user["email"] == st.session_state["user_info"]["email"]:
                            user["report"] = report
                            user["precautions"] = precautions
                            st.session_state["user_info"] = user
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
                st.success(f"Prediction: {condition}")
                st.write(report)
        else:
            st.warning("Please login/signup first.")

    elif page == "View Reports":
        if st.session_state.get("logged_in"):
            user_info = st.session_state["user_info"]
            if user_info.get("report"):
                st.subheader("Report")
                st.write(user_info["report"])
            else:
                st.warning("No report found.")
        else:
            st.warning("Please login/signup first.")

if __name__ == "__main__":
    initialize_database()
    main()
