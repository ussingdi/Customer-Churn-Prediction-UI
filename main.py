import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
from streamlit.config import set_option
import utils as ut

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ['GROQ_API_KEY'])


def explain_prediction(probability, input_dict, surname, df):
  prompt = f"""
  You are a data science expert at a bank, specializing in explaining customer churn predictions in simple, actionable terms.

  The bank has identified a customer, {surname}, and analyzed their likelihood of churning based on the following details:  
  **Customer Information:**  
  {input_dict}  

  **Top Factors Influencing Churn Prediction:**  

  | Feature           | Importance |
  |-------------------|------------|
  | NumOfProducts     | 0.323888   |
  | IsActiveMember    | 0.164146   |
  | Age               | 0.109550   |
  | Geography_Germany | 0.091373   |
  | Balance           | 0.052786   |
  | Geography_France  | 0.046463   |
  | Gender_Female     | 0.045283   |
  | Geography_Spain   | 0.036855   |
  | CreditScore       | 0.035005   |
  | EstimatedSalary   | 0.032655   |
  | HasCrCard         | 0.031940   |
  | Tenure            | 0.030054   |
  | Gender_Male       | 0.000000   |

  **Customer Behavior Comparisons:**  
  - **Churned Customers:**  
  {df[df['Exited']==1].describe()}  
  - **Non-Churned Customers:**  
  {df[df['Exited']==0].describe()}  

  ---

  ### Task:  
  Based on the customer's profile and the key features, provide a **3-sentence explanation** about their churn risk:  
  Customer will be  at High risk , when {probability} is greater than 40, 
  Customer will be at  Low Risk when {probability} is lesser than 40.
  - If the customer shows a high risk explain the specific traits contributing to their risk and how they compare with churned customers.  
  - If the customer shows a low risk explain why their profile aligns more closely with non-churned customers.  

  **Guidelines:**  
  1. Avoid mentioning specific churn probabilities, machine learning models, or feature importance scores directly. 
  2. Use simple, professional language to highlight the customer's behavior, engagement, and how it impacts their likelihood of churning.  
  3. Focus on actionable insights to help the bank improve the customer’s experience and retain their loyalty.
  4.Do not mention anything like "3-sentence explanation".
  5.Seperate First Line Mention in Bold that Churn Risk is low,high,medium,very low, very high. 
  Your explanation should be concise, insightful, and strictly three sentences.
  """

  print({round(probability*100),1})

  
  print("EXPLANATION_PROMPT", prompt)

  raw_response = client.chat.completions.create(
     model="llama-3.2-3b-preview",
     #model="llama-3.1-8b-instant",
      messages=[{
          "role": "user",
          "content": prompt
      }],
  )
  return raw_response.choices[0].message.content
  
def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""
  You are Manager at ABC Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.

  You noticed that a customer, {surname}, has a  {round(probability*100,1)} % probability of churning.

  Here is the Customer's Information: {input_dict}

  Here is some explanation as to why the customer might be at risk of churning: {explanation} 

  You need to Generate a personalized email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

  Make sure to list out set of incentives to stay based on their information , in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer or anything such like your churn risk is low, high , medium or alike.

  
  """

  raw_response = client.chat.completions.create(
     #model="llama-3.2-3b-preview",
     model="llama-3.1-8b-instant",
      messages=[{
          "role": "user",
          "content": prompt
      }],
  )

  print("\n\n Email PROMOPT",prompt)
  return raw_response.choices[0].message.content
  

def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)


xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
#random_forest_model = load_model('rf_model.pkl')

def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_of_products, has_credit_card, is_active_member,
                  estimated_salary):
  input_dict = {
      'CreditScore': credit_score,
      'Age': age,
      'Tenure': tenure,
      'Balance': balance,
      'NumOfProducts': num_of_products,
      'HasCrCard': int(has_credit_card),
      'IsActiveMember': int(is_active_member),
      'EstimatedSalary': estimated_salary,
      'Geography_France': 1 if location == 'France' else 0,
      'Geography_Germany': 1 if location == 'Germany' else 0,
      'Geography_Spain': 1 if location == 'Spain' else 0,
      'Gender_Female': 1 if gender == 'Female' else 0,
      'Gender_Male': 1 if gender == 'Male' else 0,
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict


def predict(input_df, input_dict, key_suffix=""):
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        #'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'Gaussian Naive Bayes ': naive_bayes_model.predict_proba(input_df)[0][1],
    }
    print(probabilities)
    avg_probability = np.mean(list(probabilities.values()))
    formatted_prob = "{:.1f}".format(avg_probability * 100)

    col1, col2 = st.columns(2, gap="medium")
    with col1:  # First column
        fig = ut.generate_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True, key=f"gauge_chart_{key_suffix}")
        st.write(f"The Customer has a {formatted_prob}% probability of churning", key=f"prob_text_{key_suffix}")
    with col2:  # Second column
        fig_probs = ut.create_model_proba_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True, key=f"model_proba_chart_{key_suffix}")
    return float(formatted_prob)


    


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

#creating customer list - dropdown
customer_list = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

#display list on dropdown menu
selected_customer_option = st.selectbox("Select a customer", customer_list)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print(selected_customer_id)
  selected_surname = selected_customer_option.split(" - ")[1]
  print(selected_surname)

  selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]
  print(selected_customer)

col1, col2 = st.columns(2)

with col1:
    
  credit_score = st.number_input("Credit Score",
                                 min_value=300,
                                 max_value=800,
                                 value=int(selected_customer["CreditScore"]),
                                 key="credit_score_input")

  location = st.selectbox("Location", 
                          ["Spain", "France", "Germany"],
                          index=["Spain", "France", "Germany"].index(selected_customer["Geography"]),
                          key="location_select")

  gender = st.radio("Gender", 
                    ["Male", "Female"],
                    index=0 if selected_customer["Gender"] == "Male" else 1,
                    key="gender_radio")

  age = st.number_input("Age",
                        min_value=18,
                        max_value=100,
                        value=int(selected_customer["Age"]),
                        key="age_input")

  tenure = st.number_input("Tenure (years)",
                           min_value=1,
                           max_value=100,
                           value=int(selected_customer["Tenure"]),
                           key="tenure_input")

with col2:
  balance = st.number_input("Balance",
                            min_value=0.0,
                            value=float(selected_customer["Balance"]),
                            key="balance_input")

  no_of_products = st.number_input("No. of Products",
                                   min_value=0,
                                   value=int(selected_customer["NumOfProducts"]),
                                   key="products_input")

  has_credit_card = st.checkbox("Has Credit Card",
                                value=bool(selected_customer["HasCrCard"]),
                                key="credit_card_checkbox")

  is_active_member = st.checkbox("Is Active Member",
                                 value=bool(selected_customer["IsActiveMember"]),
                                 key="active_member_checkbox")

  estimated_salary = st.number_input("Estimated Salary",
                                     min_value=0.0,
                                     value=float(selected_customer["EstimatedSalary"]),
                                     key="salary_input")

  input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                       tenure, balance, no_of_products,
                                       has_credit_card, is_active_member,
                                       estimated_salary)

# Add custom CSS with updated color scheme
st.markdown("""
    <style>
    /* Main page styling */
    .main {
        padding: 20px;
        background-color: #ffffff;
    }
    
    /* Title styling */
    .title {
        color: #FF6B6B;  /* Coral pink */
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 30px;
        padding: 20px;
        border-bottom: 2px solid #f0f0f0;
    }
    
    /* Button styling */
    div.stButton > button {
        width: 250px;
        height: 60px;
        font-size: 20px;
        font-weight: 500;
        color: white;
        background: linear-gradient(to right, #FF6B6B, #FF8E8E);  /* Coral gradient */
        border-radius: 30px;
        border: none;
        box-shadow: 0 4px 6px rgba(255, 107, 107, 0.2);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(255, 107, 107, 0.3);
    }
    
    /* Info message styling */
    .stInfo {
        background-color: rgba(255, 107, 107, 0.1);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    
    /* Subheader styling */
    .css-10trblm {
        color: #FF6B6B;
        font-size: 24px;
        margin-top: 30px;
    }
    
    /* Separator styling */
    hr {
        margin: 30px 0;
        border: none;
        border-top: 2px solid #f0f0f0;
    }
    
    /* Updated footer styling */
    .footer {
        background-color: #DCE4C9;
        color: #FF6B6B;
        text-align: center;
        padding: 15px;
        font-size: 24px;
        font-style: italic;
        border-top: 1px solid #f0f0f0;
        margin-top: 50px;  /* Add space above footer */
    }
    
    /* Add padding to main content to prevent overlap with footer */
    .main {
        padding-bottom: 60px;
    }
    
    /* Container styling */
    .css-1d391kg {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #FF6B6B !important;
    }

    /* Input fields styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border-color: #f0f0f0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF6B6B;
        box-shadow: 0 0 0 1px #FF6B6B;
    }

    /* Selectbox styling */
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border-color: #f0f0f0;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False
if 'show_email' not in st.session_state:
    st.session_state.show_email = False
if 'avg_prob' not in st.session_state:
    st.session_state.avg_prob = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None

# Show initial instruction
st.info("👇 Please click 'Let's Predict and Analyse' to see the prediction results", icon="ℹ️")

# First button for prediction and analysis
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Let's Predict and Analyse", key="predict_button"):
        st.session_state.show_prediction = True

# Show prediction and analysis results
if st.session_state.show_prediction and not st.session_state.avg_prob:
    with st.spinner("Generating prediction and analysis..."):
        st.session_state.avg_prob = predict(input_df, input_dict, key_suffix="single")
        st.session_state.explanation = explain_prediction(st.session_state.avg_prob, input_dict,
                                                       selected_customer['Surname'], df)
        st.markdown("-----------------")
        st.subheader("Explanation of Prediction")
        st.markdown(st.session_state.explanation)
    
    st.info("👇 Click 'Generate Personalized Email' to create customer email", icon="ℹ️")

# Email button
if st.session_state.show_prediction:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Generate Personalized Email", key="email_button"):
            st.session_state.show_email = True

# Show email if generated
if st.session_state.show_email:
    with st.spinner("Generating personalized email..."):
        email = generate_email(st.session_state.avg_prob, input_dict, 
                            st.session_state.explanation,
                            selected_customer['Surname'])
        st.markdown("-----------------")
        st.subheader("Personalized Email for Customer")
        st.markdown(email)

# Reset button at the bottom if needed
if st.session_state.show_prediction:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Reset All", key="reset_button"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# Add some spacing before footer
st.markdown("<br><br>", unsafe_allow_html=True)

# Non-sticky footer at the bottom
st.markdown("""
    <div class="footer">
        Made by Uvesh @ Headstarter AI
    </div>
""", unsafe_allow_html=True)
