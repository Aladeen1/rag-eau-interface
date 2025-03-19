from typing import List, Dict, Any
import time

from gcloud import storage

from mistralai import Mistral, UserMessage
import psycopg2
import streamlit as st

from s3_client import upload_to_gcloud
from utils import retrieve_context, vectorize_query, format_chunks_with_bullets, system_prompt

db_connection_string = st.secrets['SUPABASE_PG_URL']
supabase_document_table = st.secrets['SUPABASE_TABLE']
conn = psycopg2.connect(db_connection_string)

st.title("Le Ragueauteur")

st.text("Le Ragueauteur vous permet de poser n'importe quelle question en lien avec le système de gestion des données des eaux souterraines.")

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

    context = retrieve_context(conn, supabase_document_table, input_vector, 10)
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


st.subheader("Télécharger un fichier sur Google Cloud Storage")

# Téléchargez le fichier
uploaded_file = st.file_uploader("Glissez et déposez votre fichier", type=["pdf", "md"])

# Si un fichier est téléchargé
if uploaded_file is not None:
    st.write(f"Fichier téléchargé : {uploaded_file.name}")

    # Déterminer le bucket en fonction du type de fichier
    file_extension = uploaded_file.name.split('.')[-1].lower()  # Récupère l'extension du fichier

    if file_extension == "pdf":
        bucket_name = st.secrets["bucket_name_pdf"]
        st.write("Le fichier sera uploadé dans le bucket PDF.")
    elif file_extension == "md":
        bucket_name = st.secrets["bucket_name_markdown"]
        st.write("Le fichier sera uploadé dans le bucket Markdown.")
    else:
        st.error("Le fichier n'est ni un PDF ni un fichier Markdown.")
        bucket_name = None

    # Affiche un bouton pour valider l'upload
    if bucket_name is not None and st.button("Valider l'upload"):
        # Assurez-vous que les informations du bucket sont présentes dans `st.secrets`
        try:
            result = upload_to_gcloud(uploaded_file, bucket_name)
            st.success(result)
        except Exception as e:
            st.error(f"Erreur lors de l'upload du fichier : {str(e)}")
