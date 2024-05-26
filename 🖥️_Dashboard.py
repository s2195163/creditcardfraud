import streamlit as st
import plotly.express as px
import pandas as pd

if 'mydataframe' not in st.session_state:
    st.session_state.mydataframe = pd.DataFrame()

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'Logistic Regression'

features_list = ['V1', 'V2', 'V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16']

# Initialize session state variables
if 'fraud_list' not in st.session_state:
    st.session_state.fraud_list = ''

st.title("CC Fraud Monitoring System")

model_paths = {
    'Logistic Regression': 'logisticregression.pkl',
    'Random Forest': 'randomforest.pkl',
    'SVM': 'supportvector.pkl',
    'XGBoost': 'xgboost.pkl'
}

st.session_state.selected_model = st.selectbox('Choose a model to use:', list(model_paths.keys()))

# Display transaction count
first_row = st.container(border=True)
r1col1, r1col2 = st.columns(2)
second_row = st.container(border=True)
r2col1, r2col2 = st.columns([0.3,0.7])

if len(st.session_state.mydataframe)>0:
    with first_row:
        with r1col1:
            st.write(f"Total Transactions: {len(st.session_state.mydataframe)}")
        with r1col2:
            value_counts = st.session_state.mydataframe['prediction'].value_counts()
            count_1 = value_counts.get(1, 0)  # Default to 0 if the value is not found
            st.write(f"Total Fraud Transactions: {count_1}")
    with second_row:
        with r2col1:
            # Count the occurrences of each unique value in the 'prediction' column
            prediction_counts = st.session_state.mydataframe['prediction'].value_counts().reset_index()
            prediction_counts.columns = ['prediction', 'count']
            prediction_counts['prediction'] = prediction_counts['prediction'].apply(lambda x: 'Fraud' if x==1 else 'Normal')
            fig = px.pie(prediction_counts, values='count', names='prediction', title='Fraud Distribution')
            fig.update_layout(
                height=300  # Set the height of the figure
            )
            st.plotly_chart(fig, use_container_width=True)
        with r2col2:
            linechart_df = st.session_state.mydataframe[['prediction']]
            linechart_df = linechart_df['prediction'].apply(lambda x: 'Fraud' if x==1 else 'Normal')
            fig = px.line(linechart_df, y='prediction', title='Fraud Line Chart', markers=True)
            fig.update_layout(
                xaxis_title='Transaction Number',  # Set the x-axis title
                yaxis_title='Prediction',  # Set the y-axis title
                height=300  # Set the height of the figure
            )
            st.plotly_chart(fig, use_container_width=True)

    cols = ['datetime','prediction'] + [col for col in st.session_state.mydataframe if col[0]=='V']
    st.session_state.mydataframe = st.session_state.mydataframe[cols]
    st.dataframe(st.session_state.mydataframe)
else:
    st.write('Monitoring transactions...')