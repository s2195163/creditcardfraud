import streamlit as st
import random
import pickle
import pandas as pd
from datetime import datetime

if 'error_text' not in st.session_state:
    st.session_state.error_text = ''

if 'prediction' not in st.session_state:
    st.session_state.prediction = False

features_list = ['V1', 'V2', 'V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16']
for feature in features_list:
    if f'feature_{feature}' not in st.session_state:
        st.session_state[f'feature_{feature}'] = ''
        st.session_state[f'predict_{feature}'] = ''

model_paths = {
    'Logistic Regression': 'logisticregression.pkl',
    'Random Forest': 'randomforest.pkl',
    'SVM': 'supportvector.pkl',
    'XGBoost': 'xgboost.pkl'
}

# Function to handle transaction click
def handle_transaction():
    for i,feature in enumerate(features_list):
        try:
            st.session_state[f'predict_{feature}'] = float(st.session_state[f'feature_{feature}'])
            st.session_state.error_text = ''
        except ValueError:
            st.session_state.error_text = f"{feature} value '{st.session_state[f'feature_{feature}']}' cannot be converted to float."
            return
    curr_data = {}
    for i,feature in enumerate(features_list):
        curr_data[feature] = st.session_state[f'predict_{feature}']
    df = pd.DataFrame.from_dict(curr_data, orient='index').T
    with open(model_paths[st.session_state.selected_model], 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(df)
    df['datetime'] = datetime.now().strftime("%d-%b-%y %H:%M:%S")
    df['prediction'] = prediction
    if len(st.session_state.mydataframe) == 0:
        st.session_state.mydataframe = df
    else:
        st.session_state.mydataframe = pd.concat([st.session_state.mydataframe, df], ignore_index=True)
    st.rerun()

def handle_random_transaction():
    for i,feature in enumerate(features_list):
        rand_num = float(random.randint(-5, 5))
        st.session_state[f'predict_{feature}'] = rand_num
        # st.session_state[f'feature_{feature}'] = rand_num
    curr_data = {}
    for i,feature in enumerate(features_list):
        curr_data[feature] = st.session_state[f'predict_{feature}']
    df = pd.DataFrame.from_dict(curr_data, orient='index').T
    with open('logisticregression.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(df)
    df['datetime'] = datetime.now().strftime("%d-%b-%y %H:%M:%S")
    df['prediction'] = prediction
    if len(st.session_state.mydataframe) == 0:
        st.session_state.mydataframe = df
    else:
        st.session_state.mydataframe = pd.concat([st.session_state.mydataframe, df], ignore_index=True)
    st.rerun()

st.title("Simulate Transaction")

# Create two columns
col1, col2 = st.columns(2)

st.write(f"Total Transactions: {len(st.session_state.mydataframe)}")

# Input for amount
for i,feature in enumerate(features_list):
    if i % 2 == 1:
        with col2:
            st.text_input(feature, key=f'feature_{feature}')
    else:
        with col1:
            st.text_input(feature, key=f'feature_{feature}')

# Transaction button
if st.button('Make Transaction'):
    handle_transaction()
if st.button('Make Random Transaction'):
    handle_random_transaction()

st.write(st.session_state.error_text)