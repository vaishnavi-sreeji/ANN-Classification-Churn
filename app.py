import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
from tensorflow.keras.models import load_model

model=load_model("model.h5")

with open("le_gender.pkl","rb") as file:
    le_gender=pickle.load(file)
with open("ohe.pkl" ,"rb") as file:
    ohe=pickle.load(file)
with open("scaler.pkl" ,"rb") as file:
    scaler=pickle.load(file)

st.title("Customer Churn Prediction")

geography=st.selectbox("Geography",ohe.categories_[0])
gender=st.selectbox("Gender",le_gender.classes_)
age=st.slider("Age",0,100,25)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit_score")
estimated_salary=st.number_input("Estimated_salary")
tenure=st.slider("tenure",0,10)
num_of_products=st.slider("Number of Products",11,4)
has_cr_card=st.selectbox("has Credit Card",[0,1])
is_active_number=st.selectbox("Is Active Member",[0,1])


input_data=pd.DataFrame({
    "Credit_score":[credit_score],
    "Gender":[le_gender.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumofProducts":[num_of_products],
    "Hascrcard":[has_cr_card],
    "IsActiveMember":[is_active_number],
    "EstimatedSalary":[estimated_salary]


})

geo_encoded=ohe.transform([[geography]]).toarray()
ge_encoded_df=pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(["Geography"]))
# input_data=pd.concat([input_data.drop('Geography',axis=1),ge_encoded_df],axis=1)
input_data = pd.concat([input_data.reset_index(drop=True), ge_encoded_df], axis=1)



expected_columns = scaler.feature_names_in_
input_data = input_data.reindex(columns=expected_columns)

input_scaled=scaler.transform(input_data)

prediction=model.predict(input_scaled)

prediction_proba=prediction[0][0]
st.write("churn Probability:{}".format(prediction_proba))
if prediction_proba <0.5:
    st.write("customer is likely to churn")
else:
    st.write("customer is not likely to churn")


