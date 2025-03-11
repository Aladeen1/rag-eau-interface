import os
from mistralai import Mistral
from typing import List

def vectorize_query(query: str, api_key) -> List[float]:
    """
    Transforme une requête texte en vecteur d'embedding via l'API Mistral.

    Args:
        query: La requête utilisateur en texte

    Returns:
        Le vecteur d'embedding correspondant
    """
    # Récupérer la clé API de l'environnement
    if not api_key:
        raise ValueError("La variable d'environnement MISTRAL_API_KEY n'est pas définie")

    # Initialiser le client Mistral
    client = Mistral(api_key=api_key)

    # Modèle d'embedding
    model = "mistral-embed"

    # Obtenir l'embedding
    response = client.embeddings.create(
        model=model,
        inputs=[query],
    )

    # Retourner le vecteur d'embedding
    return response.data[0].embedding
