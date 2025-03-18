from typing import List, Dict, Any
import time


from minio import Minio
import io

from dotenv import load_dotenv


from mistralai import Mistral, UserMessage
import psycopg2
import streamlit as st
import subprocess

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

# FONCTION DRAG AND DROP

# Configuration de MinIO
minio_client = Minio(
    st.secrets['MINIO_INSTANCE_URL'],  # Remplacez par l'URL de votre instance MinIO
    access_key= st.secrets['MINIO_ACCESS_KEY'],
    secret_key=st.secrets["MINIO_SECRET_KEY"],
    secure=False  # Choisissez True si vous utilisez HTTPS
)

# Nom du bucket sur MinIO
bucket_name = "pdf"

def upload_to_minio(file):
    # Lire le contenu du fichier téléchargé
    file_content = file.read()

    # Crée un flux en mémoire pour MinIO
    file_stream = io.BytesIO(file_content)

    # Nom du fichier (tu peux ajouter un timestamp pour éviter les collisions)
    file_name = file.name

    try:
        # Upload sur MinIO
        minio_client.put_object(
            bucket_name,  # Le nom du bucket
            file_name,    # Le nom du fichier sur MinIO
            file_stream,  # Le flux du fichier
            len(file_content)  # La taille du fichier
        )
        st.success(f"Fichier {file_name} téléchargé avec succès sur MinIO.")
    except Exception as e:
        st.error(f"Erreur lors de l'upload sur MinIO: {e}")

# Interface Streamlit pour drag and drop
uploaded_file = st.file_uploader("Glissez et déposez votre fichier", type=["pdf"])


    # Si un fichier est téléchargé
if uploaded_file is not None:
    st.write(f"Fichier téléchargé : {uploaded_file.name}")
    load_dotenv()
    # Affiche un bouton pour valider l'upload
    if st.button("Valider l'upload sur MinIO"):
        # Appel de la fonction pour uploader le fichier sur MinIO
        file_name = upload_to_minio(uploaded_file)
