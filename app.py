from typing import List, Dict, Any
import time

from mistralai import Mistral, UserMessage
import psycopg2
import streamlit as st

from utils import retrieve_context, vectorize_query, format_chunks_with_bullets, system_prompt

db_connection_string = st.secrets['SUPABASE_PG_URL']
conn = psycopg2.connect(db_connection_string)

st.title("Le Ragueauteur")

st.text("Le Rageauteur vous permet de poser n'importe quelle question en lien avec le système de gestion des données des eaux souterraines.")

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
if prompt := st.chat_input("Besoin de renseignement ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    input_vector = vectorize_query(prompt, client)

    context = retrieve_context(conn, input_vector, 10)
    context = format_chunks_with_bullets(context, prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    system_message = {"role": "system", "content": f"{system_prompt}\n\nContext: {context}"}
    messages = [system_message] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

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
