import os
from typing import List, Dict, Any

import psycopg2
from mistralai import Mistral
from pydantic import BaseModel

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


def retrieve_context(connection, table, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    cursor = connection.cursor()
    # Conversion du vecteur Python en format PostgreSQL
    vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

    cursor.execute(
        f"SELECT content FROM match_{table}(%s::vector(1024), %s)",
        (vector_str, limit)
    )

    results = []
    for row in cursor.fetchall():
        results.append({"content": row[0]})

    cursor.close()
    return results


system_prompt = """
    Vous êtes un assistant spécialisé dans la réponse aux questions sur un logiciel de suivi des eaux.

    <instructions>
    Utilisez UNIQUEMENT les informations présentes dans le <context> et la <prompt> pour formuler vos réponses.
    Si l'information n'est pas dans le <context>, répondez: "Je ne trouve pas cette information dans la documentation disponible."
    Ne faites jamais d'hypothèses ou n'inventez pas d'informations absentes du <context>.
    Privilégiez toujours l'information du <context>, même si vous pensez connaître la réponse.
    </instructions>

    <format_réponse>
    Répondez de manière directe.
    Utilisez un langage simple et accessible.
    Structurez logiquement votre réponse, par points importants.
    Quand vous citez une source précise du <context>, utilisez la balise <source>nom_de_la_source</source>.
    Quand vous formulez une réponse, n'incluez pas les balises XML et parsez votre réponse intelligement pour l'utilisateur.
    </format_réponse>

    <traitement_contexte>
    Si le <context> contient des informations contradictoires, indiquez-le clairement avec la balise <contradiction>détails</contradiction>.
    Si la question est ambiguë, répondez à l'interprétation la plus pertinente selon le <context>.
    N'indiquez jamais explicitement que vous utilisez un contexte fourni.
    </traitement_contexte>

    <structure_réponse>
    Pour faciliter l'affichage, structure ta réponse avec ces sections clairement identifiées:

    RESUME:
    Ecris un résumé répondant au mieux à la query initiale. Soit clair et concis (2/3 phrases maximum)

    DEVELOPPEMENT:
    Ecris un développement structuré et détaillé répondant au mieux a la <query>, appuie toi au maximum sur le <context> et ton analyse.

    REFERENCES:
    [Liste des sources utilisées]
    Pour les questions conversationnelles simples (salutations, remerciements, questions sur l'historique de la conversation),
    répondez de manière naturelle sans utiliser cette structure.
    </structure_réponse>
"""
def format_chunks_with_bullets(chunks, prompt=""):
    """
    Formate les chunks en une chaîne de texte avec des puces et des retours à la ligne propres

    Args:
        chunks (list): Liste de dictionnaires contenant la clé 'content'
        prompt (str, optional): La requête initiale de l'utilisateur

    Returns:
        str: Chaîne formatée avec introduction et puces
    """
    # Construction de l'en-tête
    formatted_text = f"Retrieved content for user prompt ({prompt}):\n"

    # Ajout des puces avec les contenus
    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '')

        # Gestion des retours à la ligne dans chaque chunk
        # Indentation des lignes suivantes pour alignement avec la puce
        lines = content.split('\n')
        if lines:
            # Première ligne avec puce
            formatted_text += f"* {lines[0]}\n"

            # Lignes suivantes avec indentation pour alignement
            for line in lines[1:]:
                if line.strip():  # Ignorer les lignes vides
                    formatted_text += f"  {line}\n"

            # Ajouter un retour à la ligne supplémentaire entre les chunks
            if i < len(chunks) - 1:
                formatted_text += "\n"

    return formatted_text

#Class pour Output structuré de Mistral
class MistralOutput(BaseModel):
    resume: str
    developpement: str
    references: List[str] = []

#Parsing de la réponse Mistral:
def parse_mistral_response(text: str) -> MistralOutput:
    # Valeurs par défaut
    resume = ""
    developpement = ""
    references = []

    # Séparation simple par sections
    if "RESUME:" in text:
        parts = text.split("RESUME:", 1)
        remaining = parts[1].strip()

        if "DEVELOPPEMENT:" in remaining:
            resume_parts = remaining.split("DEVELOPPEMENT:", 1)
            resume = resume_parts[0].strip()
            remaining = resume_parts[1].strip()

            if "REFERENCES:" in remaining:
                dev_parts = remaining.split("REFERENCES:", 1)
                developpement = dev_parts[0].strip()
                refs_text = dev_parts[1].strip()
                references = [ref.strip() for ref in refs_text.split("\n") if ref.strip()]
            else:
                developpement = remaining
        else:
            resume = remaining
    else:
        # Si le format n'est pas respecté, on met tout dans développement
        developpement = text

    return MistralOutput(
        resume=resume if resume else "Pas de résumé disponible",
        developpement=developpement,
        references=references
    )
