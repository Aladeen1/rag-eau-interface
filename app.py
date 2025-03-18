from typing import List, Dict, Any
import time
import nest_asyncio
import asyncio

from mistralai import Mistral, UserMessage
import psycopg2
import streamlit as st

from utils import retrieve_context, vectorize_query, format_chunks_with_bullets, system_prompt
from mode_profond import execute_mode_profond

# Active le support des boucles asyncio imbriquées (important pour Streamlit)
nest_asyncio.apply()

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

#Add "mode" to the session state
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'concis'

# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Configuration bouton + état suivant
option_names = ['concis', 'profond']

mode_button = st.button(f"Mode {st.session_state['mode']}")
next_mode = 'profond' if st.session_state['mode'] == 'concis' else 'concis'

# Logique quand le bouton est cliqué
if mode_button:
    st.session_state['mode'] = next_mode
    st.rerun()

# User input
if prompt := st.chat_input("Besoin de renseignement ?"):
    if st.session_state['mode'] == 'concis':

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

    # Mode profond (nouveau)
    elif st.session_state['mode'] == 'profond':
        with st.chat_message("assistant", avatar="images/logoRageau.jpg"):
            with st.spinner("Analyse approfondie en cours..."):
                # Appel au workflow d'agent
                resultat = execute_mode_profond(prompt)

                # Affichage du résultat
                st.markdown(resultat.summary)

                # Affichage des sections détaillées
                for section in resultat.sections:
                    with st.expander(section.title):
                        st.markdown(section.content)

                # Stockage de la réponse complète pour l'historique
                full_response = resultat.summary + "\n\n" + "\n".join([f"### {s.title}\n{s.content}" for s in resultat.sections])

        # Mise à jour de l'historique
        st.session_state.messages.append({"role": "assistant", "content": full_response})
