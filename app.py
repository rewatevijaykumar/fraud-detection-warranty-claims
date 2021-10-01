from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle
import seaborn as sns
import matplotlib.pyplot as plt




def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversion
    href = f'<a href="data:/file/csv;base64,{b64}" download="warranty_claims.csv">Download CSV File</a>'
    return href

def main():
    # allow user to download train data
    df = pd.read_csv('warranty_claims.csv')

    st.title('Warranty Claim - Fraud Detection')
    st.sidebar.title('Input Parameters')
    st.sidebar.markdown('Please input features to predict whether warranty claim is Genuine or Fraud')
    st.markdown(filedownload(df),unsafe_allow_html=True)




    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)

    else:
        def user_input_features():
            region_list = sorted(df['Region'].unique().tolist())
            region = st.sidebar.selectbox('Customer Region',region_list)
            state_list = sorted(df['State'].unique().tolist())
            state = st.sidebar.selectbox('Current location of customer',state_list)
            area_list = sorted(df['Area'].unique().tolist())
            area = st.sidebar.selectbox('Area',area_list)
            city_list = sorted(df['City'].unique().tolist())
            city = st.sidebar.selectbox('Customers current located city',city_list)
            consumer_profile_list = sorted(df['Consumer_profile'].unique().tolist())
            consumer_profile = st.sidebar.selectbox("Customer's work profile",consumer_profile_list)
            product_category_list = sorted(df['Product_category'].unique().tolist())
            product_category = st.sidebar.selectbox('Product Category',product_category_list)
            product_type_list = sorted(df['Product_type'].unique().tolist())
            product_type = st.sidebar.selectbox('Type of the product',product_type_list)
            st.sidebar.markdown("Note: For AC / TV Issue - '0' means to replace the component, '1' means partial damage of the component and with servicing component good work and '2' no issue with the component.")
            ac_1001_issue_list = sorted(df['AC_1001_Issue'].unique().tolist())
            ac_1001_issue = st.sidebar.selectbox('AC_1001_Issue - Compressor',ac_1001_issue_list)
            ac_1002_issue_list = sorted(df['AC_1002_Issue'].unique().tolist())
            ac_1002_issue = st.sidebar.selectbox('AC_1002_Issue - Condenser',ac_1002_issue_list)
            ac_1003_issue_list = sorted(df['AC_1003_Issue'].unique().tolist())
            ac_1003_issue = st.sidebar.selectbox('AC_1003_Issue - Evaporator',ac_1003_issue_list)
            tv_2001_issue_list = sorted(df['TV_2001_Issue'].unique().tolist())
            tv_2001_issue = st.sidebar.selectbox('TV_2001_Issue - Power Supply',tv_2001_issue_list)
            tv_2002_issue_list = sorted(df['TV_2002_Issue'].unique().tolist())
            tv_2002_issue = st.sidebar.selectbox('TV_2002_Issue - Inverter',tv_2002_issue_list)
            tv_2003_issue_list = sorted(df['TV_2003_Issue'].unique().tolist())
            tv_2003_issue = st.sidebar.selectbox('TV_2003_Issue - Motherboard',tv_2003_issue_list)
            claim_value = st.sidebar.slider("Customer's claim amount in Rupees",0.00,50000.00,0.00)
            service_centre_list = sorted(df['Service_Centre'].unique().tolist())
            service_centre = st.sidebar.selectbox('Service Centre',service_centre_list)
            product_age = st.sidebar.slider('Duration of the product purchased by customer',0,1000,0)
            purchased_from_list = sorted(df['Purchased_from'].unique().tolist())
            purchased_from = st.sidebar.selectbox('From where product is purchased',purchased_from_list)
            call_details = st.sidebar.slider('Call Duration in mins',0.5,30.0,0.5)
            purpose_list = sorted(df['Purpose'].unique().tolist())
            purpose = st.sidebar.selectbox('Purpose',purpose_list)
            data = {
                'Region': [region],
                'State': [state],
                'Area': [area],
                'City': [city],
                'Consumer_profile': [consumer_profile],
                'Product_category': [product_category],
                'Product_type': [product_type],
                'AC_1001_Issue': [ac_1001_issue],
                'AC_1002_Issue': [ac_1002_issue],
                'AC_1003_Issue': [ac_1003_issue],
                'TV_2001_Issue': [tv_2001_issue],
                'TV_2002_Issue': [tv_2002_issue],
                'TV_2003_Issue': [tv_2003_issue],
                'Claim_Value': [claim_value],
                'Service_Centre': [service_centre],
                'Product_Age': [product_age],
                'Purchased_from': [purchased_from],
                'Call_details': [call_details],
                'Purpose': [purpose],
            }
            features = pd.DataFrame(data)
            return features

        input_df = user_input_features()

    print(input_df)
    fraud_raw = pd.read_csv('warranty_claims.csv')

    # drop Unnamed column
    fraud_raw.drop(['Unnamed: 0','Fraud'], axis=1, inplace=True)

    fraud_raw.fillna(0, inplace=True)

    fraud = fraud_raw
    # fraud = fraud_raw.drop(columns=['Fraud'])

    pred_df = pd.concat([input_df,fraud],axis=0)
    
    # encode
    pred_df = pd.get_dummies(pred_df)

    pred_df = pred_df[:1] #select only the first row (user input data)

    pred_df.fillna(0, inplace=True)

    # Displays the user input features

    st.subheader('User Input features')

    if uploaded_file is not None:

        st.write(input_df)
        # st.write(pred_df)

    else:

        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')

        st.write(input_df)
        # st.write(pred_df)

    load_clf = pickle.load(open('models/QuadraticDiscriminantAnalysis.pkl', 'rb'))


    prediction = load_clf.predict(pred_df)

    prediction_proba = load_clf.predict_proba(pred_df)

    fraud_labels = np.array(['Genuine','Fraud'])

    st.write(fraud_labels[prediction])

    st.subheader('Prediction Probability')

    st.write(prediction_proba)  

    # result =""
    # if st.button("Predict"):
    #     result = fraud_labels[prediction]
    # st.success('The output is {}'.format(result))

if __name__ == '__main__':
    main()