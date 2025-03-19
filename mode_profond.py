"""
Framework d'agent de RAG (Retrieval Augmented Generation) avec raffinement adaptatif de requêtes
basé sur les premiers résultats et présentation détaillée des informations.
"""

import os
import time
import json
import asyncio
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from functools import partial

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.mistral import MistralModel
from asyncpg import Pool, create_pool
import streamlit as st
import nest_asyncio

from utils import vectorize_query
from clients import client

# Active le support des boucles asyncio imbriquées
nest_asyncio.apply()

#
# Modèles de données
#

class RefinedQuery(BaseModel):
    """Requête raffinée générée à partir des premiers résultats"""
    text: str
    reason: str = Field(description="Justification de cette requête supplémentaire")
    expected_information: str = Field(description="Informations attendues de cette requête")

class QueryRefinementResult(BaseModel):
    """Résultat de l'analyse des premiers documents et du raffinement de requêtes"""
    original_query: str
    refined_queries: List[RefinedQuery]
    analysis_summary: str = Field(description="Résumé de l'analyse des documents initiaux")
    information_gaps: List[str] = Field(description="Lacunes d'information identifiées")

class Document(BaseModel):
    """Document récupéré de la base de données"""
    id: str
    content: str
    relevance_score: Optional[float] = None
    from_query: Optional[str] = None  # Trace la requête source

class SearchResult(BaseModel):
    """Résultat de la recherche documentaire"""
    documents: List[Document]
    is_sufficient: bool = Field(description="Indique si les résultats sont suffisants")
    feedback: Optional[str] = Field(None, description="Raison si insuffisant")

class DetailedSection(BaseModel):
    """Section détaillée de la réponse finale"""
    title: str = Field(description="Titre de la section")
    content: str = Field(description="Contenu détaillé de la section")
    sources: List[str] = Field(description="Sources utilisées dans cette section")

class EnhancedAnswer(BaseModel):
    """Réponse finale enrichie avec sections détaillées et mise en forme soignée"""
    summary: str = Field(description="Résumé concis de la réponse")
    sections: List[DetailedSection] = Field(description="Sections détaillées de la réponse")
    key_insights: List[str] = Field(description="Points clés à retenir")
    sources: List[str] = Field(description="Toutes les sources utilisées")
    confidence: float = Field(ge=0.0, le=1.0, description="Niveau de confiance dans la réponse")
    limitations: Optional[str] = Field(None, description="Limitations éventuelles de la réponse")

@dataclass
class RetrievalDeps:
    """Dépendances pour l'agent de récupération"""
    conn: Pool
    max_attempts: int = 2
    cache: Dict[str, Any] = None

@dataclass
class ProcessingState:
    """État global du traitement d'une requête"""
    original_query: str
    refined_queries: List[str] = field(default_factory=list)
    initial_documents: List[Document] = field(default_factory=list)
    all_documents: List[Document] = field(default_factory=list)
    processed_queries: Set[str] = field(default_factory=set)
    information_gaps: List[str] = field(default_factory=list)

# Cache pour les embeddings
embedding_cache = {}

#
# Configuration des modèles LLM
#

API_KEY = st.secrets['MISTRAL_API_KEY']
refinement_model = MistralModel(model_name='mistral-small-latest', api_key=API_KEY)
retrieval_model = MistralModel(model_name='mistral-small-latest', api_key=API_KEY)
presentation_model = MistralModel(model_name='mistral-large-latest', api_key=API_KEY)

#
# Fonctions utilitaires
#

async def get_embedding(query: str, max_retries=3) -> List[float]:
    """Génère un embedding vectoriel avec gestion du cache et des erreurs"""
    cache_key = query.lower().strip()

    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    retry_count = 0
    base_delay = 1

    while retry_count < max_retries:
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, partial(vectorize_query, query), client)
            embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            if "rate limit" in str(e).lower() and retry_count < max_retries - 1:
                retry_count += 1
                print(f"Rate limit: attente {base_delay}s ({retry_count}/{max_retries})...")
                await asyncio.sleep(base_delay)
                base_delay *= 2
            else:
                print(f"Erreur d'embedding: {e}")
                return [random.uniform(-1, 1) for _ in range(1024)]


    return [random.uniform(-1, 1) for _ in range(1024)]

#
# Agent de récupération
#

retrieval_agent = Agent(
    model=retrieval_model,
    deps_type=RetrievalDeps,
    result_type=SearchResult,
    system_prompt="""Tu es un agent de récupération d'informations pertinentes.
Évalue si les résultats sont suffisants pour répondre à la question.
Sois critique et rigoureux.""",
    retries=3
)

@retrieval_agent.tool
async def search_docs(ctx: RunContext[RetrievalDeps], query: str, limit: int = 3) -> List[Document]:
    """Recherche des documents pertinents via embedding vectoriel"""
    cache_key = f"{query}:{limit}"

    if ctx.deps.cache and cache_key in ctx.deps.cache:
        return ctx.deps.cache[cache_key]

    start_time = time.time()
    try:
        query_embedding = await get_embedding(query)
        pg_vector = json.dumps(query_embedding)

        try:
            rows = await asyncio.wait_for(
                ctx.deps.conn.fetch(
                    f"""
                    SELECT id, id_doc, content, metadata, similarity
                    FROM match_{st.secrets['SUPABASE_TABLE']}($1, $2, $3, $4::jsonb, NULL)
                    """,
                    pg_vector, limit, 0.5, '{}'
                ),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            print("Timeout de la requête DB")
            rows = []

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

        execution_time = time.time() - start_time
        print(f"Recherche '{query}' exécutée en {execution_time:.2f}s - {len(documents)} documents trouvés")

        return documents

    except Exception as e:
        print(f"Erreur de recherche: {e}")
        return []

@retrieval_agent.result_validator
async def validate_search_results(ctx: RunContext[RetrievalDeps], result: SearchResult) -> SearchResult:
    """Valide la qualité des résultats de recherche"""
    attempt = getattr(ctx, 'attempt', 1)

    if not result.documents:
        result.is_sufficient = False
        result.feedback = "Aucun document trouvé"

        if attempt >= ctx.deps.max_attempts:
            result.is_sufficient = True
            result.feedback += " après plusieurs tentatives"
            return result

        setattr(ctx, 'attempt', attempt + 1)
        raise ModelRetry(f"Tentative {attempt}/{ctx.deps.max_attempts}: Aucun document trouvé.")

    if any(doc.relevance_score and doc.relevance_score > 0.8 for doc in result.documents):
        result.is_sufficient = True
        return result

    if not result.is_sufficient and attempt >= ctx.deps.max_attempts:
        result.is_sufficient = True
        result.feedback += f" (après {ctx.deps.max_attempts} tentatives)"
        return result

    if not result.is_sufficient:
        setattr(ctx, 'attempt', attempt + 1)
        feedback = result.feedback or "Résultats insuffisants"
        raise ModelRetry(f"Tentative {attempt}/{ctx.deps.max_attempts}: {feedback}")

    return result

#
# Agent de raffinement de requêtes
#

query_refinement_agent = Agent(
    model=refinement_model,
    result_type=QueryRefinementResult,
    system_prompt="""Tu es un expert en analyse documentaire et génération ciblée de requêtes complémentaires.

Ton objectif est d'analyser en profondeur les documents initiaux fournis et de:
1. Déterminer si les documents contiennent déjà suffisamment d'informations
2. Identifier les lacunes d'information vraiment importantes et critiques
3. Ne proposer des requêtes complémentaires QUE si absolument nécessaire

IMPORTANT:
- Privilégie toujours une analyse approfondie des documents existants avant de proposer des sous-requêtes
- Limite-toi aux sous-requêtes vraiment nécessaires (maximum 3)
- Chaque sous-requête doit apporter une valeur significative et combler une lacune critique
- Les sous-requêtes doivent être précises, spécifiques et non redondantes

Ton objectif n'est PAS de générer un maximum de sous-requêtes, mais de déterminer quelles informations manquantes sont réellement nécessaires pour répondre à la question originale.""",
    retries=2
)
#
# Agent de présentation amélioré
#

presentation_agent = Agent(
    model=presentation_model,
    result_type=EnhancedAnswer,
    system_prompt="""Tu es un agent de présentation spécialisé dans la structuration et le formatage d'informations détaillées. Tu recevras du texte mais également des tableaux.

Ton objectif est de créer une réponse riche et complète qui:
1. Présente les informations de manière claire et bien structurée
2. Ne néglige aucun détail pertinent trouvé dans les documents
3. Organise le contenu en sections thématiques avec titres explicites
4. Fournit un résumé concis mais complet en introduction
5. Met en évidence les points clés à retenir
6. Cite rigoureusement toutes les sources utilisées
7. Indique les éventuelles limitations de la réponse

Concernant les tableaux spécifiquement:
- Tu dois les restructurer pour optimiser leur lisibilité et leur compréhension
- Assure-toi que les en-têtes sont clairs et pertinents
- Utilise le formatage approprié (alignement, espacement) pour améliorer la lisibilité
- Ajoute des notes explicatives si certaines données du tableau nécessitent des clarifications
- Intègre harmonieusement les tableaux dans la structure globale de ta réponse
- N'hésite pas à transformer un tableau en liste à puces ou en paragraphes si cela améliore la compréhension

Ta priorité est d'offrir une présentation soignée qui valorise la qualité des informations trouvées tout en maximisant la couverture des détails pertinents. Le traitement optimal des tableaux est essentiel pour atteindre cet objectif.""",
    retries=2
)

def preprocess_documents(documents: List[Document], max_chars: int = 1500) -> List[Document]:
    """Tronque les documents pour limiter les tokens tout en préservant plus de contenu"""
    return [
        Document(
            id=doc.id,
            content=doc.content[:max_chars] + ("..." if len(doc.content) > max_chars else ""),
            relevance_score=doc.relevance_score,
            from_query=doc.from_query
        )
        for doc in documents
    ]

def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """Élimine les documents en double basés sur leur ID"""
    seen_ids = set()
    unique_docs = []

    for doc in documents:
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            unique_docs.append(doc)

    return unique_docs

def categorize_documents(documents: List[Document]) -> Dict[str, List[Document]]:
    """Organise les documents par requête source pour faciliter la structuration"""
    categories = {}

    for doc in documents:
        query = doc.from_query or "Requête principale"
        if query not in categories:
            categories[query] = []
        categories[query].append(doc)

    return categories

async def init_db_pool() -> Pool:
    """Initialise la connexion à la base de données"""
    return await create_pool(
        st.secrets['SUPABASE_PG_URL'],
        min_size=5,
        max_size=15,
        statement_cache_size=0,
        timeout=120.0,
        command_timeout=10.0
    )

def get_streamlit_deps(pool):
    """Crée des dépendances spécifiques pour l'environnement Streamlit avec plus de tentatives"""
    return RetrievalDeps(
        conn=pool,
        max_attempts=5,
        cache={}
    )

async def process_query(question: str, pool: Pool, relevance_threshold: float = 0.65) -> EnhancedAnswer:
    """Traite une requête utilisateur avec raffinement adaptatif et présentation enrichie"""
    state = ProcessingState(original_query=question)
    cache = {}
    deps = RetrievalDeps(conn=pool, cache=cache, max_attempts=2)

    try:
        # Étape 1: Recherche initiale avec la requête originale
        print(f"Recherche initiale: {question}")
        retrieval_start = time.time()

        try:
            initial_retrieval = await asyncio.wait_for(
                retrieval_agent.run(question, deps=deps),
                timeout=45.0
            )

            # Filtrer les documents par pertinence
            initial_documents = [
                doc for doc in initial_retrieval.data.documents
                if doc.relevance_score is None or doc.relevance_score >= relevance_threshold
            ]

            print(f"Documents filtrés: {len(initial_retrieval.data.documents)} → {len(initial_documents)} (seuil: {relevance_threshold})")

            state.initial_documents = initial_documents
            state.all_documents = initial_documents.copy()
            state.processed_queries.add(question)

            retrieval_time = time.time() - retrieval_start
            print(f"Recherche initiale terminée en {retrieval_time:.2f}s - {len(initial_documents)} documents pertinents")

            if not initial_documents:
                print("Aucun document pertinent trouvé lors de la recherche initiale")
                return EnhancedAnswer(
                    summary="Aucune information suffisamment pertinente n'a été trouvée pour répondre à cette question.",
                    sections=[],
                    key_insights=["Aucune donnée pertinente disponible"],
                    sources=[],
                    confidence=0.0,
                    limitations="Absence de documents pertinents dans la base de connaissances."
                )

        except Exception as e:
            print(f"Erreur lors de la recherche initiale: {e}")
            return EnhancedAnswer(
                summary=f"Une erreur s'est produite lors de la recherche initiale: {str(e)}",
                sections=[],
                key_insights=["Erreur technique lors de la recherche"],
                sources=[],
                confidence=0.0,
                limitations="Erreur technique lors de la recherche initiale"
            )

        # Étape 2: Analyse approfondie des documents initiaux SANS génération immédiate de sous-requêtes
        print("Analyse approfondie des documents initiaux...")
        analysis_start = time.time()

        try:
            # Préparation des documents initiaux pour l'analyse
            processed_initial_docs = preprocess_documents(initial_documents)

            # Modifier le prompt pour privilégier l'analyse approfondie avant de générer des sous-requêtes
            analysis_prompt = {
                "original_query": question,
                "documents_initiaux": [doc.model_dump() for doc in processed_initial_docs],
                "nombre_documents": len(processed_initial_docs),
                "directive": "Analyse en profondeur les documents initiaux. Identifie les informations essentielles manquantes AVANT de proposer des sous-requêtes."
            }

            # Utiliser l'agent de raffinement uniquement pour l'analyse approfondie
            refinement_result = await asyncio.wait_for(
                query_refinement_agent.run(json.dumps(analysis_prompt)),
                timeout=60.0
            )

            # Extraction des lacunes d'information et génération ciblée de sous-requêtes
            state.information_gaps = refinement_result.data.information_gaps

            # Ne garder que les requêtes raffinées vraiment nécessaires (max 3)
            important_queries = refinement_result.data.refined_queries[:3] if len(refinement_result.data.refined_queries) > 3 else refinement_result.data.refined_queries
            state.refined_queries = [rq.text for rq in important_queries]

            analysis_time = time.time() - analysis_start
            print(f"Analyse approfondie terminée en {analysis_time:.2f}s")
            print(f"Requêtes complémentaires générées: {len(state.refined_queries)}")

            # Afficher les requêtes complémentaires si elles existent
            for i, rq in enumerate(important_queries, 1):
                print(f"  {i}. {rq.text}")
                print(f"     ↳ Raison: {rq.reason}")
                print(f"     ↳ Information attendue: {rq.expected_information}")

        except Exception as e:
            print(f"Erreur lors de l'analyse approfondie: {e}")
            # En cas d'échec de l'analyse, on continue sans sous-requêtes
            state.refined_queries = []
            state.information_gaps = []

        # Étape 3: Exécution SÉLECTIVE des requêtes complémentaires (uniquement après analyse)
        if state.refined_queries:
            print("\nExécution des requêtes complémentaires...")

            for query in state.refined_queries:
                if query in state.processed_queries:
                    continue

                try:
                    retrieval_result = await asyncio.wait_for(
                        retrieval_agent.run(query, deps=deps),
                        timeout=30.0
                    )

                    # Filtrer les documents par pertinence
                    relevant_docs = [
                        doc for doc in retrieval_result.data.documents
                        if doc.relevance_score is None or doc.relevance_score >= relevance_threshold
                    ]

                    print(f"Requête '{query}': {len(retrieval_result.data.documents)} → {len(relevant_docs)} documents pertinents")

                    if relevant_docs:
                        state.all_documents.extend(relevant_docs)

                    state.processed_queries.add(query)

                except Exception as e:
                    print(f"Erreur lors de la recherche pour '{query}': {e}")

        # Étape 4: Déduplication et préparation des documents
        unique_documents = deduplicate_documents(state.all_documents)
        print(f"\nDocuments récupérés: {len(state.all_documents)} → {len(unique_documents)} après déduplication")

        if not unique_documents:
            return EnhancedAnswer(
                summary="Malgré plusieurs tentatives, aucune information pertinente n'a été trouvée.",
                sections=[],
                key_insights=["Aucune donnée disponible malgré l'analyse approfondie"],
                sources=[],
                confidence=0.0,
                limitations="Absence de documents pertinents dans la base de connaissances."
            )

        # Étape 5: Présentation enrichie
        processed_docs = preprocess_documents(unique_documents, max_chars=1500)
        categorized_docs = categorize_documents(processed_docs)

        # Construction d'un prompt enrichi pour l'agent de présentation
        prompt = {
            "question_originale": question,
            "requetes_executees": list(state.processed_queries),
            "lacunes_identifiees": state.information_gaps,
            "documents_par_categorie": {
                category: [doc.model_dump() for doc in docs]
                for category, docs in categorized_docs.items()
            },
            "nombre_total_documents": len(processed_docs),
            "instructions_particulieres": "Privilégier la précision et la pertinence des informations tout en présentant de manière structurée et concise."
        }

        presentation_start = time.time()
        final_result = await asyncio.wait_for(
            presentation_agent.run(json.dumps(prompt)),
            timeout=60.0
        )

        print(f"Présentation enrichie terminée en {time.time() - presentation_start:.2f}s")
        return final_result.data


    except asyncio.TimeoutError:
        print("Timeout global de l'opération")
        return EnhancedAnswer(
            summary="Le traitement a pris trop de temps et n'a pas pu être complété.",
            sections=[],
            key_insights=["Opération interrompue pour cause de délai excessif"],
            sources=[],
            confidence=0.0,
            limitations="Timeout de l'opération"
        )
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        return EnhancedAnswer(
            summary="Une erreur s'est produite lors du traitement de la requête.",
            sections=[],
            key_insights=["Erreur technique rencontrée"],
            sources=[],
            confidence=0.0,
            limitations=f"Erreur technique: {str(e)}"
        )


def execute_mode_profond(question: str) -> EnhancedAnswer:
    """
    Fonction synchrone qui permet d'appeler facilement le workflow d'agent
    depuis une application Streamlit.

    Args:
        question (str): La question de l'utilisateur

    Returns:
        EnhancedAnswer: La réponse structurée
    """
    async def run_query():
        # Initialise la connexion à la base de données
        pool = await init_db_pool()
        try:
            return await process_query(question, pool)
        finally:
            # Gestion plus robuste de la fermeture du pool
            try:
                # Utilisation de asyncio.shield pour éviter que le timeout n'annule l'opération de fermeture
                close_task = asyncio.shield(pool.close())
                await asyncio.wait_for(close_task, timeout=90.0)
            except asyncio.TimeoutError:
                print("Attention: Timeout lors de la fermeture du pool, mais l'opération continue en arrière-plan")
            except Exception as e:
                print(f"Problème de fermeture du pool: {e}")

    # Exécute la fonction asynchrone et retourne le résultat
    return asyncio.run(run_query())
