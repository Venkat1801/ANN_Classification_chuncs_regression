import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


#load the trained model
model=tf.keras.models.load_model('regression_model.h5')

#Load the encoders and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gen=pickle.load(file)
with open('onehotencoder_geo.pkl','rb') as file:
    onehotencoder_geo=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

    #user input
#user input
geography=st.selectbox('Geography',onehotencoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gen.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
exited=st.selectbox('Exited',[0,1])
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Has credit card',[0,1])
is_active_member=st.selectbox('Is Active Number',[0,1])

#prepare the input input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gen.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveNumber':[is_active_member],
    'Exited':[exited]
})

#one-hot encode Geography
geo_encoded=onehotencoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehotencoder_geo.get_feature_names_out(['Geography']))

#combined one-hot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled=scaler.trasform(input_data)

##predict churn 
prediction=model.predict(input_data_scaled)
predicted_salary=prediction[0][0]

st.write(f'Predicted estimated salary:${predicted_salary:.2f}')

