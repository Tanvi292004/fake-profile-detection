import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load(r'C:\Users\tanvi\OneDrive\Desktop\Fake Profile Detection\model\xgboost_model.pkl')

st.title("📸 Fake Profile Detection App")
st.markdown("Enter profile details below to check if it's fake or not:")

# Input fields
followers = st.number_input("Followers Count", min_value=0)
following = st.number_input("Following Count", min_value=0)
posts = st.number_input("Number of Posts", min_value=0)
has_profile_pic = st.selectbox("Has Profile Picture?", ["Yes", "No"])
has_bio = st.selectbox("Has Bio?", ["Yes", "No"])
has_external_url = st.selectbox("Has External URL?", ["Yes", "No"])
private = st.selectbox("Is the account private?", ["Yes", "No"])
description_length = st.number_input("Bio Description Length", min_value=0)

# Convert Yes/No to 1/0
has_profile_pic = 1 if has_profile_pic == "Yes" else 0
has_bio = 1 if has_bio == "Yes" else 0
has_external_url = 1 if has_external_url == "Yes" else 0
private = 1 if private == "Yes" else 0

# Create input DataFrame
input_df = pd.DataFrame([[
    has_profile_pic,  # profile pic
    0.27,             # nums/length username (static for now, you can add logic to calculate)
    0,                # fullname words (static for now)
    0,                # nums/length fullname
    0,                # name==username
    description_length,  # description length
    has_external_url, # external URL
    private,          # private account
    posts,            # number of posts
    followers,        # number of followers
    following         # number of follows
]], columns=[
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private',
    '#posts', '#followers', '#follows'
])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 This might be a **FAKE PROFILE**!")
    else:
        st.success("✅ This appears to be a **REAL PROFILE**.")
