from mistralai import Mistral
import streamlit as st




# Initialize Mistral client
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
