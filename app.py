import pickle
import pandas as pd 
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import streamlit as st

with open('labelencoder_gender.pkl','rb') as file:
    lb_gender = pickle.load(file)

with open('OneHotEncoder_geo.pkl','rb') as file:
    encoder_geo = pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar = pickle.load(file)

model = load_model('model.h5')

st.title('Churn Prediction system')

geography = st.selectbox('Geography', encoder_geo.categories_[0])
gender = st.selectbox('Gender', lb_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'Geography': [geography],
    'CreditScore': [credit_score],
    'Gender': [lb_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

df = input_data.copy()

geo_array = encoder_geo.transform(df[['Geography']])
geo_df = pd.DataFrame(
    geo_array.toarray(),
    columns=encoder_geo.get_feature_names_out()
)

geo_df = geo_df.reset_index(drop=True)
df = df.reset_index(drop=True)

new_df = pd.concat([df, geo_df], axis=1)

new_df.drop('Geography', axis=1, inplace=True)

new_df = new_df[scalar.feature_names_in_]

new_df = scalar.transform(new_df)

pred = model.predict(new_df)
pred_prob = pred[0][0]
print(pred)
if pred_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')