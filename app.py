# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:20:29 2023

@author: sanjy
"""

import streamlit as st
import requests
import joblib

def main():
    st.title("Streamlit Django App")
    st.write("Welcome to the Model App!")

    # Streamlit form input fields
    recency = st.number_input("Recency")
    total_expenses = st.number_input("Total Expenses")
    income = st.number_input("Income")
    total_acc_cmp = st.number_input("Total Acc Cmp")
    total_purchases = st.number_input("Total Purchases")

    # Submit button
    if st.button("Submit"):
        # Load the trained model
        model = joblib.load(open("dec1.pkl", "rb"))  # or joblib.load()

        # Make predictions
        user_input = [[recency, total_purchases, total_expenses, income, total_acc_cmp]]
        prediction = model.predict(user_input)

        # Display the predicted cluster
        st.write("Predicted Cluster:", prediction[0])

if __name__ == "__main__":
    main()
