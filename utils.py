import os
from typing import List, Dict, Any

import psycopg2
from mistralai import Mistral

def vectorize_query(query: str, client: Mistral) -> List[float]:
    """
    Transforme une requête texte en vecteur d'embedding via l'API Mistral.

    Args:
        query: La requête utilisateur en texte

    Returns:
        Le vecteur d'embedding correspondant
    """
    # Modèle d'embedding
    model = "mistral-embed"

    # Obtenir l'embedding
    response = client.embeddings.create(
        model=model,
        inputs=[query],
    )

    # Retourner le vecteur d'embedding
    return response.data[0].embedding


def retrieve_context(connection, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    cursor = connection.cursor()
    # Conversion du vecteur Python en format PostgreSQL
    vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"
    
    cursor.execute(
        "SELECT id, id_doc, content, similarity FROM match_mvp_docs(%s::vector(1024), %s)",
        (vector_str, 5)
    )
    
    results = []
    for row in cursor.fetchall():
        results.append({
            "id": row[0],
            "id_doc": row[1],
            "content": row[2],
            "similarity": row[3]
        })
    
    cursor.close()
    return results
