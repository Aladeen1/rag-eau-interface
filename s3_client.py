import io
from google.cloud import storage

from minio import Minio
import streamlit as st

# Configuration de MinIO
minio_client = Minio(
    st.secrets['MINIO_INSTANCE_URL'],  # Remplacez par l'URL de votre instance MinIO
    access_key= st.secrets['MINIO_ACCESS_KEY'],
    secret_key=st.secrets["MINIO_SECRET_KEY"],
    secure=False  # Choisissez True si vous utilisez HTTPS
)

# Nom du bucket sur MinIO
bucket_name = 'st.secrets["MINIO_BUCKET_NAME"]'

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


def upload_to_gcloud(file, bucket_name):

    # Créez un client de stockage Google Cloud
    client = storage.Client()

    # Récupère le bucket
    bucket = client.get_bucket(bucket_name)

    # Crée un objet blob dans le bucket (le fichier sera enregistré sous ce nom)
    blob = bucket.blob(file.name)

    # Crée un flux en mémoire avec le contenu du fichier téléchargé
    file_content = file.read()
    file_stream = io.BytesIO(file_content)

    # Upload le fichier vers le bucket Google Cloud Storage
    blob.upload_from_file(file_stream)

    # Retourner un message de confirmation
    return f"Fichier '{file.name}' téléchargé avec succès vers gs://{bucket_name}/{file.name}"
