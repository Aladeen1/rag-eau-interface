import streamlit as st
from mistralai import Mistral, UserMessage
import time

st.title("Mistral Chatbot")

# Initialize Mistral client
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])

# Set default model
if "mistral_model" not in st.session_state:
    st.session_state["mistral_model"] = "mistral-large-latest"

# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me something!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    # Call Mistral API


    with st.chat_message("assistant", avatar="images/logoRageau.jpg"):
        stream_response = client.chat.stream(
            model=st.session_state["mistral_model"],
            messages=messages,
        )
        full_response = ""
        placeholder = st.empty()  # Crée un espace réservé pour la réponse
        for chunk in stream_response:
            content = chunk.data.choices[0].delta.content
            if content:
                full_response += content
                placeholder.markdown(full_response + "▌", unsafe_allow_html=True)  # Affiche la réponse partielle
                time.sleep(0.05)  # Petite pause pour rendre l'affichage plus naturel
        placeholder.markdown(full_response, unsafe_allow_html=True)  # Affiche la réponse complète

    st.session_state.messages.append({"role": "assistant", "content": full_response})
