import os
import json
import asyncio
import time
from typing import List, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.mistral import MistralModel
from asyncpg import Pool, create_pool
import streamlit as st
import nest_asyncio

from utils import vectorize_query
from clients import client

# Active les boucles asyncio imbriquées
nest_asyncio.apply()

# Modèles de données simplifiés
class Document(BaseModel):
    """Document récupéré de la base de données"""
    id: str
    content: str
    relevance_score: float = None
    from_query: str = None  # Trace la requête source

class RefinedQuery(BaseModel):
    """Requête raffinée générée à partir des premiers résultats"""
    text: str
    reason: str = Field(description="Justification de cette requête")

class SearchResult(BaseModel):
    """Résultat de la recherche documentaire"""
    documents: List[Document]
    is_sufficient: bool = True
    feedback: str = None

@dataclass
class RetrievalDeps:
    """Dépendances pour l'agent de récupération"""
    conn: Pool
    cache: Dict[str, Any] = None

# Configuration des modèles LLM
API_KEY = st.secrets.get('MISTRAL_API_KEY', '')
model = MistralModel(model_name='mistral-small-latest', api_key=API_KEY)

# Agent de récupération
retrieval_agent = Agent(
    model=model,
    deps_type=RetrievalDeps,
    result_type=SearchResult,
    system_prompt="""Tu es un agent de récupération d'informations pertinentes.
Évalue si les résultats sont suffisants pour répondre à la question."""
)

@retrieval_agent.tool
async def search_docs(ctx: RunContext[RetrievalDeps], query: str, limit: int = 5) -> List[Document]:
    """Recherche des documents pertinents"""
    cache_key = f"{query}:{limit}"
    if ctx.deps.cache and cache_key in ctx.deps.cache:
        return ctx.deps.cache[cache_key]

    try:
        # Vectorisation de la requête
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(None, vectorize_query, query, client)
        pg_vector = json.dumps(query_embedding)

        # Exécution de la requête vectorielle
        rows = await ctx.deps.conn.fetch(
            f"""
            SELECT id, id_doc, content, metadata, similarity
            FROM match_{st.secrets['SUPABASE_TABLE']}($1, $2, $3, $4::jsonb, NULL)
            """,
            pg_vector, limit, 0.5, '{}'
        )

        documents = [
            Document(
                id=str(row['id']),
                content=row['content'],
                relevance_score=row.get('similarity', 0.5),
                from_query=query
            )
            for row in rows
        ]

        if ctx.deps.cache is not None:
            ctx.deps.cache[cache_key] = documents

        return documents

    except Exception as e:
        print(f"Erreur de recherche: {e}")
        return []

# Agent de raffinement de requêtes
refine_agent = Agent(
    model=model,
    result_type=List[RefinedQuery],
    system_prompt="""Analyse les documents fournis et génère des requêtes complémentaires ciblées.
Limite-toi à 5 requêtes maximum, vraiment pertinentes et non redondantes."""
)

async def init_db_pool() -> Pool:
    """Initialise la connexion à la base de données"""
    return await create_pool(
        st.secrets['SUPABASE_PG_URL'],
        min_size=2,
        max_size=5,
        statement_cache_size=0
    )

async def process_query(question: str, pool: Pool, relevance_threshold: float = 0.65, max_queries: int = 3):
    """Workflow RAG simplifié avec requêtes complémentaires"""
    # Initialisation
    start_time = time.time()
    all_documents = []
    processed_queries = set()
    deps = RetrievalDeps(conn=pool, cache={})

    # Étape 1: Recherche initiale
    print(f"Recherche initiale: {question}")
    try:
        initial_result = await retrieval_agent.run(question, deps=deps)
        initial_docs = [
            doc for doc in initial_result.data.documents
            if doc.relevance_score is None or doc.relevance_score >= relevance_threshold
        ]
        all_documents.extend(initial_docs)
        processed_queries.add(question)
        print(f"Documents initiaux: {len(initial_docs)}")
    except Exception as e:
        print(f"Erreur lors de la recherche initiale: {e}")
        return {"error": str(e), "documents": []}

    # Étape 2: Analyse et génération de requêtes complémentaires
    if initial_docs:
        try:
            # Préparation des données pour l'analyse
            analysis_prompt = {
                "original_query": question,
                "documents": [
                    {
                        "id": doc.id,
                        "content": doc.content[:1000] + ("..." if len(doc.content) > 1000 else ""),
                        "score": doc.relevance_score
                    }
                    for doc in initial_docs
                ]
            }

            # Génération des requêtes complémentaires
            refined_queries = await refine_agent.run(json.dumps(analysis_prompt))
            queries_to_process = [q.text for q in refined_queries.data][:max_queries]
            print(f"Requêtes complémentaires générées: {len(queries_to_process)}")

            # Étape 3: Exécution des requêtes complémentaires
            for query in queries_to_process:
                if query in processed_queries:
                    continue

                try:
                    print(f"Exécution de la requête complémentaire: {query}")
                    retrieval_result = await retrieval_agent.run(query, deps=deps)
                    relevant_docs = [
                        doc for doc in retrieval_result.data.documents
                        if doc.relevance_score is None or doc.relevance_score >= relevance_threshold
                    ]
                    all_documents.extend(relevant_docs)
                    processed_queries.add(query)
                    print(f"Documents trouvés: {len(relevant_docs)}")
                except Exception as e:
                    print(f"Erreur lors de la recherche pour '{query}': {e}")

        except Exception as e:
            print(f"Erreur lors de l'analyse des résultats initiaux: {e}")

    # Étape 4: Déduplication des documents
    unique_docs = {}
    for doc in all_documents:
        if doc.id not in unique_docs:
            unique_docs[doc.id] = doc

    result = {
        "original_query": question,
        "total_queries": len(processed_queries),
        "processed_queries": list(processed_queries),
        "documents": list(unique_docs.values()),
        "total_documents": len(unique_docs),
        "execution_time": time.time() - start_time
    }

    return result

def execute_rag_workflow(question: str):
    """Fonction synchrone pour exécuter le workflow depuis Streamlit"""
    async def run():
        pool = await init_db_pool()
        try:
            return await process_query(question, pool)
        finally:
            try:
                # Ajouter un timeout à la fermeture du pool pour éviter les blocages
                await asyncio.wait_for(pool.close(), timeout=5.0)
            except asyncio.TimeoutError:
                print("Timeout lors de la fermeture du pool, des connexions peuvent rester ouvertes")

    return asyncio.run(run())
