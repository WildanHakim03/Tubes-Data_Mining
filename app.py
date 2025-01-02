import pickle
import streamlit as st
from sklearn.tree import plot_tree
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = 'dataset.csv'
if os.path.exists(dataset_path):
    data = pd.read_csv(dataset_path)    
else:
    st.error(f"Error: Dataset '{dataset_path}' not found.")
    st.stop()


columns_to_drop = [
    'CLIENTNUM',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]
if all(col in data.columns for col in columns_to_drop):
    data = data.drop(columns=columns_to_drop, axis=1)
else:
    st.error("Some columns to drop are missing from the dataset.")
    st.stop()

numerical_columns = data.select_dtypes(include=['float64', 'int64'])
categorical_columns = ['Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

if all(col in data.columns for col in categorical_columns):
    encoded_columns = pd.get_dummies(data[categorical_columns], drop_first=True)

target_column = 'Attrition_Flag'
if target_column in data.columns:
    data = pd.concat([data[target_column], numerical_columns, encoded_columns], axis=1)
else:
    st.error(f"Target column '{target_column}' is missing.")
    st.stop()

X = data.drop(columns=[target_column])

model_path = 'decision_tree_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error(f"Error: Model file '{model_path}' not found.")
    st.stop()

st.sidebar.title("MENU")
page = st.sidebar.radio("Halaman", ["Home", "Prediksi", "Visualisasi"])

if page == "Home":
    st.title("SELAMAT DATANG DI APLIKASI PREDIKSI SEGMENTASI NASABAH")
    st.write("""
    Aplikasi ini menggunakan Decision Tree untuk memprediksi segmentasi nasabah (attriated/existing) berdasarkan data yang diberikan. 
    Anda dapat menggunakan menu navigasi untuk:
    - Membaca deskripsi aplikasi pada halaman *Home*.
    - Melakukan prediksi pada halaman *Prediksi*.
    - Melihat visualisasi data pada halaman *Visualisasi*.
    
    Disusun Oleh
    - Reykeisha Ridhan Taruna (1202223106)
    - Nico Viogi Pratama (1202223015)
    - Muhammad Fadli Mannguluang (1202223209)
    - Wildan Baihaqi Hakim (1202223346)
    """)

elif page == "Prediksi":
    st.title("Prediksi Segmentasi Nasabah")

    st.header("Informasi Nasabah")

    customer_data = {}
    customer_data['Customer_Age'] = st.number_input('Customer Age (26 - 65)')
    customer_data['Dependent_count'] = st.number_input('Dependent Count (1 - 4)')
    customer_data['Months_on_book'] = st.number_input('Months on Book (20 - 52)')
    customer_data['Total_Relationship_Count'] = st.number_input('Total Relationship Count (1 - 6)')
    customer_data['Months_Inactive_12_mon'] = st.number_input('Months Inactive 12 Mon (1 - 4)')
    customer_data['Contacts_Count_12_mon'] = st.number_input('Contacts Count 12 Mon (1 - 4)')
    customer_data['Credit_Limit'] = st.number_input('Credit Limit (1438.3 - 10836)')
    customer_data['Total_Revolving_Bal'] = st.number_input('Total Revolving Balance (0 - 2517)')
    customer_data['Total_Trans_Amt'] = st.number_input('Total Transaction Amount (510 - 8109)')
    customer_data['Total_Trans_Ct'] = st.number_input('Total Transaction Count (10-106)')

    st.write('---')

    selected_education = st.radio(
        "Select Education Level",
        options=["None", "Doctorate", "Graduate", "High School", "Post-Graduate", "Uneducated", "Unknown"]
    )
    customer_data['Education_Level_Doctorate'] = int(selected_education == "Doctorate")
    customer_data['Education_Level_Graduate'] = int(selected_education == "Graduate")
    customer_data['Education_Level_High School'] = int(selected_education == "High School")
    customer_data['Education_Level_Post-Graduate'] = int(selected_education == "Post-Graduate")
    customer_data['Education_Level_Uneducated'] = int(selected_education == "Uneducated")
    customer_data['Education_Level_Unknown'] = int(selected_education == "Unknown")

    st.write('---')

    selected_marital_status = st.radio(
        "Select Marital Status",
        options=["None", "Married", "Single", "Unknown"]
    )
    customer_data['Marital_Status_Married'] = int(selected_marital_status == "Married")
    customer_data['Marital_Status_Single'] = int(selected_marital_status == "Single")
    customer_data['Marital_Status_Unknown'] = int(selected_marital_status == "Unknown")

    st.write('---')

    selected_income_category = st.radio(
        "Select Income Category",
        options=["None", "$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K", "Unknown"]
    )
    customer_data['Income_Category_$40K - $60K'] = int(selected_income_category == "$40K - $60K")
    customer_data['Income_Category_$60K - $80K'] = int(selected_income_category == "$60K - $80K")
    customer_data['Income_Category_$80K - $120K'] = int(selected_income_category == "$80K - $120K")
    customer_data['Income_Category_Less than $40K'] = int(selected_income_category == "Less than $40K")
    customer_data['Income_Category_Unknown'] = int(selected_income_category == "Unknown")

    st.write('---')

    selected_card_category = st.radio(
        "Select Card Category",
        options=["None", "Gold", "Silver"]
    )
    customer_data['Card_Category_Gold'] = int(selected_card_category == "Gold")
    customer_data['Card_Category_Silver'] = int(selected_card_category == "Silver")

    customer_input = pd.DataFrame([customer_data])

    if st.button('Prediksi'):
        try:
            prediction = model.predict(customer_input)
            st.write(f'Prediksi: {prediction[0]}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif page == "Visualisasi":
    st.title("Visualisasi Data")
    
    st.subheader("Overview")
    st.dataframe(data.head())
    
    st.subheader("Pohon Keputusan")
    
    try:
        fig, ax = plt.subplots(figsize=(60, 15))
        plot_tree(
            model, 
            feature_names=X.columns,
            class_names=model.classes_,
            filled=True,
            rounded=True,
            proportion=True,
            fontsize=10,
            max_depth=4
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating decision tree visualization: {e}")