# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(page_title='Diet Recommendation using Facial Images')
st.title("Diet Recommendation using Facial Images")
st.write("Upload an image to predict Body Mass Index(BMI) and Diet subsequently")
from deepface import DeepFace
import face_recognition
import numpy as np
def get_face_encoding(image_):
	image_data = np.asarray(image_)
	face_encoding = face_recognition.face_encodings(image_data)
	if not face_encoding:
		print("No face detected")
		return np.zeros(128).tolist()
	return face_encoding[0].tolist()

import joblib
from PIL import Image
BMI_model = 'bmi_mlp.model'
Gender_model = 'gender_rfc.model'
#joblib.dump(BMI_MLP_model,BMI_model)

BMI_model = joblib.load(BMI_model)
Gender_model = joblib.load(Gender_model)
st.markdown("""
<style>
body {
    color: #000;
    background-color: #A0A0A0;
}
</style>
    """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded Image', use_column_width=True)
	prediction = DeepFace.analyze(image)
	test_array = np.expand_dims(np.array(get_face_encoding(image)),axis=0)
	BMI = round(np.asscalar(np.exp(BMI_model.predict(test_array))),2)
	Gender = age_model.predict(test_array)
	if Gender==0:
        	st.title("Gender: Male")
	else:
	    	st.title("Gender: Female")
	if BMI <14:
		st.title("Please upload a proper facial image")
	elif BMI>19.5 and BMI <=25:
		st.success("NORMAL BMI")
		st.write("BMI is : ",BMI)
	elif BMI>25 and BMI<=30:
		st.info("OVERWEIGHT BMI")
		st.write("BMI is : ",BMI)
	elif BMI>30:
		st.error("OBESE BMI")
		st.write("BMI is : ",BMI)
	else:
		st.warning("UNDERWEIGHT BMI")
		st.write("BMI is : ",BMI)
	st.write("Age:",prediction['age'])
	st.write("Ethnicity:",prediction['dominant_race'])


