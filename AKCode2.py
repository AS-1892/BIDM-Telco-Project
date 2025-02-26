# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:12:05 2025

@author: akank
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Custom Styling to match the color scheme from the banner (red, yellow, orange)
st.markdown("""
    <style>
        /* App background and font colors */
        body, .stApp { background-color: #ffffff; color: #333333; }
        .sidebar .sidebar-content { background-color: #FFEB3B; }
        
        /* Header */
        h1, h2, h3, h4, h5, h6 {
            color: #F44336 !important; /* Red */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
        }

        /* Sidebar Title */
        [data-testid="stSidebar"] h2 {
            color: #FF9800 !important; /* Orange */
        }

        /* Radio Buttons */
        div[data-testid="stSidebar"] div[role="radiogroup"] label {
            background-color: transparent !important;
            color: #F44336 !important; /* Red */
            border: 2px solid #FF9800 !important; /* Orange */
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 5px;
            transition: background 0.3s ease-in-out;
        }

        /* Hover Effect */
        div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #FFEB3B !important; /* Yellow */
            color: #F44336 !important; /* Red */
        }

        /* Selected Radio Button */
        div[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
            background-color: #FF9800 !important; /* Orange */
            color: #ffffff !important;
            border-color: #FF9800 !important; /* Orange */
        }

        /* Buttons */
        button {
            background-color: #FF9800 !important; /* Orange */
            color: #ffffff !important;
            border: 2px solid #F44336 !important; /* Red */
            border-radius: 8px;
            padding: 10px 15px;
            transition: background 0.3s ease-in-out;
        }

        /* Button Hover */
        button:hover {
            background-color: #F44336 !important; /* Red */
            color: #ffffff !important;
            border-color: #FF9800 !important; /* Orange */
        }

        /* Title and Header Customization for Visualizations */
        .stTextInput input {
            color: #F44336 !important; /* Red */
        }

        /* Custom header color for visualizations */
        .stHeader {
            color: #FF9800 !important; /* Orange */
        }

        /* Adjust other text elements */
        .stMetric {
            background-color: #FFEB3B; /* Yellow */
            color: #F44336 !important; /* Red */
        }

        /* Adjust Dataframe header and text */
        div[data-testid="stDataFrame"] th {
            color: #FF9800 !important; /* Orange */
        }
        
        /* Custom styling for Graphs */
        .stPlotlyChart {
            color: #FFEB3B !important; /* Yellow for text inside charts */
        }

    </style>
""", unsafe_allow_html=True)

# App Header with a gradient banner and image height equal to banner height
col1, col2 = st.columns([2, 6])
with col1:
    st.image("CuCh.jpg", width=150, use_column_width=False)  # Adjusting image width for better visual alignment
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # Adding space between the image and the banner
with col2:
    st.markdown("""
        <div style="text-align: left; background: linear-gradient(90deg, #FFEB3B, #FF9800, #F44336); padding: 30px; border-radius: 10px; margin-bottom: 15px; height: 200px; width: 100%; display: flex; align-items: center;">
            <h1 style="color: #ffffff; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 45px; font-weight: 600; line-height: 1.2;">
                Customer Retention Analyzer
            </h1>
        </div>
        """, unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

df = load_data()

# Preprocessing function
def preprocess_data(data):
    data = data.copy()
    # Drop customerID column if it exists
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
   
    # Replace inconsistent service strings
    data.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
   
    # Convert TotalCharges to numeric and fill missing values with the median
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
   
    # Encode binary categorical variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x in ['Male', 'Yes'] else 0)
   
    # Function to encode three-option services
    def encode_service(x):
        if x == "Yes":
            return 2
        elif x == "No":
            return 0
        else:
            return 1  # Covers any non-standard response
   
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        data[col] = data[col].apply(encode_service)
   
    # Encode Contract: Month-to-month -> 0, One year -> 1, Two year -> 2
    data['Contract'] = data['Contract'].apply(lambda x: 0 if x=="Month-to-month" else 1 if x=="One year" else 2)
   
    # Encode PaymentMethod: Electronic check -> 0, Mailed check -> 1, Bank transfer (automatic) -> 2, Credit card (automatic) -> 3
    data['PaymentMethod'] = data['PaymentMethod'].apply(
        lambda x: 0 if x=="Electronic check" else 1 if x=="Mailed check"
        else 2 if x=="Bank transfer (automatic)" else 3
    )
   
    # Ensure target variable (Churn) is numeric
    if data['Churn'].dtype == 'object':
        data['Churn'] = data['Churn'].apply(lambda x: 1 if x=="Yes" else 0)
   
    return data

df_processed = preprocess_data(df)

# Train-Test Split and Model Training
def train_model(data):
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    X = data[features]
    y = data['Churn']
   
    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
   
    # Get feature importance
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
   
    return model, acc, cm, feature_importance, X_train, X_test, y_train, y_test

model, acc, cm, feature_importance, X_train, X_test, y_train, y_test = train_model(df_processed)

# Define the function properly before calling it
def visualize_data(df):
   plt.style.use('dark_background')

   # Customer Demographics Section
   st.subheader("📊 Customer Demographics")

   col1, col2 = st.columns(2)

   with col1:
       st.write("### Tenure Distribution")
       fig, ax = plt.subplots()
       sns.histplot(df['tenure'], ax=ax, color='cyan', bins=30)
       ax.set_xlabel("Tenure (Months)")
       ax.set_ylabel("Count")
       st.pyplot(fig)

   with col2:
       st.write("### Monthly Charges by Churn")
       fig, ax = plt.subplots()
       sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax, palette="coolwarm")
       ax.set_xlabel("Churn (0 = No, 1 = Yes)")
       ax.set_ylabel("Monthly Charges ($)")
       st.pyplot(fig)

   # Service Usage Patterns
   st.subheader("📡 Service Usage Patterns")
   st.write("### Monthly Charges vs. Total Charges")

   fig, ax = plt.subplots()
   sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df, ax=ax, palette="coolwarm", alpha=0.7)
   ax.set_xlabel("Monthly Charges ($)")
   ax.set_ylabel("Total Charges ($)")
   ax.set_title("Monthly Charges vs. Total Charges (Colored by Churn)")
   st.pyplot(fig)


# Sidebar Navigation
st.sidebar.title("Navigation")
#Sidebar options for navigation
option = st.sidebar.radio("🔍 Choose Analysis", 
    ["📊 Dataset Overview", "📈 Visualizations", "🤖 Churn Prediction", "📈 Model Evaluation", "Churn Risk Assessment"])
#option = st.sidebar.radio("Go to", ["Customer Profile Analysis", "Visualizations", "Churn Risk Assessment", "Interactive Data Exploration", "Churn Prediction"])


# Customer Profile Analysis
if option == "📊 Dataset Overview":
    # Page Header
    st.markdown("<h1 style='text-align: center; color: #2a3f5f;'>Customer Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Quick Stats Row
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df), help="Total number of customers in dataset")
    with col2:
        avg_tenure = df['tenure'].mean()
        st.metric("Average Tenure", f"{avg_tenure:.1f} months", delta="+2% MoM")
    with col3:
        churn_rate = (df['Churn'].value_counts(normalize=True)[1] * 100)
        st.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-1.2% MoM", delta_color="inverse")
    with col4:
        avg_clv = (df['MonthlyCharges'] * df['tenure']).mean()
        st.metric("Avg CLV", f"${avg_clv:,.0f}", help="Average Customer Lifetime Value")
    
    st.markdown("---")
    
    # Main Content
    st.subheader("🔍 Data Exploration")
    
    # Create container for better visual separation
    with st.container():
        st.markdown("### 📋 Data Preview")
        # Display dataframe with better styling
        st.dataframe(
            df.head(20),
            height=300,
            use_container_width=True,
            column_config={
                "CustomerID": "ID",
                "tenure": "Tenure (months)",
                "MonthlyCharges": st.column_config.NumberColumn(
                    "Monthly Charges",
                    format="$%.2f"
                )
            }
        )
    
    # Columns with improved spacing
    col_left, col_right = st.columns([6, 7])

    with col_left:
        with st.container():
            st.markdown("### 📊 Churn Distribution by Key Features")
        
            # Select a feature to analyze churn distribution
            feature_to_analyze = st.selectbox(
                "Select a Feature to Analyze Churn",
                options=['Contract', 'PaymentMethod', 'InternetService', 'SeniorCitizen', 'Dependents'],
                index=0  # Default to 'Contract'
            )
        
            # Map numerical values back to their original labels for better visualization
            if feature_to_analyze == 'Contract':
                feature_labels = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
            elif feature_to_analyze == 'PaymentMethod':
                feature_labels = {0: 'Electronic check', 1: 'Mailed check', 2: 'Bank transfer', 3: 'Credit card'}
            elif feature_to_analyze == 'InternetService':
                feature_labels = {0: 'No', 1: 'DSL', 2: 'Fiber optic'}
            elif feature_to_analyze == 'SeniorCitizen':
                feature_labels = {0: 'No', 1: 'Yes'}
            elif feature_to_analyze == 'Dependents':
                feature_labels = {0: 'No', 1: 'Yes'}
        
            # Create a DataFrame for visualization
            churn_distribution = df_processed.groupby([feature_to_analyze, 'Churn']).size().unstack(fill_value=0)
            churn_distribution.index = churn_distribution.index.map(feature_labels)
        
            # Plot the churn distribution
            fig = px.bar(
                churn_distribution,
                x=churn_distribution.index,
                y=[0, 1],  # 0: No Churn, 1: Churn
                barmode='group',
                labels={'x': feature_to_analyze, 'y': 'Count', 'color': 'Churn'},
                title=f"<b>Churn Distribution by {feature_to_analyze}</b>",
                color_discrete_sequence=['#FFEB3B', '#F44336']  # Yellow: No Churn, Red: Churn
            )
        
            fig.update_layout(
                xaxis_title=feature_to_analyze,
                yaxis_title="Number of Customers",
                legend_title="Churn",
                hovermode="x unified",
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif", size=12),
                plot_bgcolor='#ffffff',  # White background
                paper_bgcolor='#ffffff',  # White background
                margin=dict(l=20, r=20, t=40, b=20)
            )
        
            st.plotly_chart(fig, use_container_width=True)

    # Add spacing between the two visualizations
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)
    
    with col_right:
        with st.container():
            st.markdown("### 💰 Customer Lifetime Value (CLV) by Churn Status")
        
            # Calculate CLV
            df_processed['CLV'] = df_processed['MonthlyCharges'] * df_processed['tenure']
        
            # Map Churn values to labels for better visualization
            churn_labels = {0: 'Not Churned', 1: 'Churned'}
            df_processed['Churn_Label'] = df_processed['Churn'].map(churn_labels)
        
            # Create a Violin Plot to show CLV distribution by Churn Status
            fig = px.violin(
                df_processed,
                x='Churn_Label',
                y='CLV',
                color='Churn_Label',
                box=True,  # Show box plot inside the violin
                points="all",  # Show all data points
                title="<b>Customer Lifetime Value (CLV) Distribution by Churn Status</b>",
                labels={'Churn_Label': 'Churn Status', 'CLV': 'Customer Lifetime Value ($)'},
                color_discrete_map={'Not Churned': '#FFEB3B', 'Churned': '#F44336'}  # Yellow: Not Churned, Red: Churned
            )
        
            # Update layout for better readability
            fig.update_layout(
                xaxis_title="Churn Status",
                yaxis_title="Customer Lifetime Value ($)",
                legend_title="Churn Status",
                hovermode="x unified",
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif", size=12),
                plot_bgcolor='#ffffff',  # White background
                paper_bgcolor='#ffffff',  # White background
                margin=dict(l=20, r=20, t=40, b=20)
            )
        
            st.plotly_chart(fig, use_container_width=True)

    # Add custom CSS styling
    st.markdown("""
    <style>
        .stContainer {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stMetric {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2a3f5f;
        }
        label {
            color: #ff00ff !important;  /* Neon Pink */
            font-weight: bold;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

  # Page 2: Visualizations
elif option == "📈 Visualizations":
    # Section 1: Key Metrics
    st.subheader("📊 Key Business Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df_processed))
    
    # Convert 'Churn' column from strings to numeric (1 for Yes, 0 for No)
    if df_processed['Churn'].dtype == 'object':
        df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    
    with col2:
        churn_rate = df_processed['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
    with col3:
        st.metric("Avg Monthly Charges", f"${df_processed['MonthlyCharges'].mean():.2f}")
        
    with col4:
        st.metric("Avg Tenure", f"{df_processed['tenure'].mean():.1f} months")

    # Increase width of col1 (1:1 ratio)
    col1, col2 = st.columns([1, 1])  

    with col1:
        fig = px.pie(df_processed, names='gender', title='Gender Distribution',
                     color_discrete_sequence=px.colors.sequential.Aggrnyl)
    
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(width=400, height=500)
    
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig = px.histogram(df_processed, x='tenure', nbins=20, title='Tenure Distribution',
                           color_discrete_sequence=['#00F3FF'])
        fig.update_layout(bargap=0.1)
        fig.update_layout(width=1000, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.header("🔌 Service Usage Patterns")
    service_cols = ['PhoneService', 'InternetService', 'StreamingTV', 'TechSupport']
    selected_service = st.selectbox("Select Service", service_cols)

    fig = px.bar(df_processed, x=selected_service, color='Churn', barmode='group',
                 title=f'{selected_service} vs Churn',
                 color_discrete_sequence=['#FF00FF', '#00FF00'])

    st.plotly_chart(fig, use_container_width=True)
    #visualize_data(df_processed)  # Call the function after defining it



# Churn Prediction
elif option == "🤖 Churn Prediction":
    # st.image("Churn.png", use_container_width=True)
    st.write("## 🔮 Churn Prediction")
    
    # Apply custom styling to enhance visibility on black background
    st.markdown("""
    <style>
        label {
            color: #ff00ff !important;  /* Neon Pink */
            font-weight: bold;
            font-size: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    # User input for features
    tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0)

    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    # Encode inputs consistently with training
    gender_encoded = 1 if gender == "Male" else 0
    partner_encoded = 1 if partner == "Yes" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    phone_service_encoded = 1 if phone_service == "Yes" else 0

    def encode_service_input(x):
        if x == "Yes":
            return 2
        elif x == "No":
            return 0
        else:
            return 1

    multiple_lines_encoded = encode_service_input(multiple_lines)
    online_security_encoded = encode_service_input(online_security)
    online_backup_encoded = encode_service_input(online_backup)
    device_protection_encoded = encode_service_input(device_protection)
    tech_support_encoded = encode_service_input(tech_support)
    streaming_tv_encoded = encode_service_input(streaming_tv)
    streaming_movies_encoded = encode_service_input(streaming_movies)

    contract_encoded = 0 if contract == "Month-to-month" else 1 if contract == "One year" else 2
    paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
    payment_method_encoded = (
        0 if payment_method == "Electronic check" 
        else 1 if payment_method == "Mailed check" 
        else 2 if payment_method == "Bank transfer (automatic)" 
        else 3
    )

    # Create input data DataFrame
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender': [gender_encoded],
        'Partner': [partner_encoded],
        'Dependents': [dependents_encoded],
        'PhoneService': [phone_service_encoded],
        'MultipleLines': [multiple_lines_encoded],
        'OnlineSecurity': [online_security_encoded],
        'OnlineBackup': [online_backup_encoded],
        'DeviceProtection': [device_protection_encoded],
        'TechSupport': [tech_support_encoded],
        'StreamingTV': [streaming_tv_encoded],
        'StreamingMovies': [streaming_movies_encoded],
        'Contract': [contract_encoded],
        'PaperlessBilling': [paperless_billing_encoded],
        'PaymentMethod': [payment_method_encoded]
    })

    if st.button("Predict Churn"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][prediction]
        result = "🔥 Customer is likely to churn!" if prediction == 1 else "✅ Customer is likely to stay!"
        #st.subheader("Prediction Result")
        #st.write(f"Confidence: {prob*100:.2f}%")
        st.markdown(f"<h3 style='color:{'#f0f' if prediction == 1 else '#0ff'}'>{result}</h3>", unsafe_allow_html=True)

# Page 4: Model Evaluation
elif option == "📈 Model Evaluation":
    st.write("## 📊 Model Evaluation Metrics")
     
    st.write(f"### **Model Accuracy: {acc:.2f}**")

    # Create two columns for horizontal layout
    col1, col2 = st.columns(2)

    # Confusion Matrix (Left Column)
    with col1:
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    # Feature Importance (Right Column)
    with col2:
        st.write("### Feature Importance")
        fig, ax = plt.subplots()
        feature_importance.plot(kind='bar', ax=ax, color='cyan')
        st.pyplot(fig)

    # Correlation Matrix (Full Width Below)
    st.write("### Correlation Matrix")
    numeric_df = df_processed.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
# Churn Risk Assessment
elif option == "Churn Risk Assessment":
    st.title("📌 Churn Risk Assessment")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df_processed,ax=ax, palette=['#f0f', '#ff0'])
    st.pyplot(fig)


# Service Bundle Analysis
    st.write("### Top 10 Service Bundles Used by Customers")
    service_features = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'StreamingTV']
    df_processed['ServiceBundle'] = df_processed[service_features].astype(str).agg('-'.join, axis=1)
    bundle_counts = df_processed['ServiceBundle'].value_counts().reset_index()
    bundle_counts.columns = ['ServiceBundle', 'Count']
    fig = px.bar(bundle_counts[:10], x='ServiceBundle', y='Count', title="Top 10 Service Bundles")
    st.plotly_chart(fig)