from typing import List, Dict, Any
import time
import nest_asyncio
import asyncio

from mistralai import Mistral, UserMessage
import psycopg2
import streamlit as st

from s3_client import upload_to_minio
from utils import retrieve_context, vectorize_query, format_chunks_with_bullets, system_prompt
from mode_profond import execute_mode_profond
from clients import client

# Active le support des boucles asyncio imbriquées (important pour Streamlit)
nest_asyncio.apply()

db_connection_string = st.secrets['SUPABASE_PG_URL']
supabase_document_table = st.secrets['SUPABASE_TABLE']
conn = psycopg2.connect(db_connection_string)

st.title("Le Ragueauteur")

st.text("Le Ragueauteur vous permet de poser n'importe quelle question en lien avec le système de gestion des données des eaux souterraines.")

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
        start_time = time.time()  # Pour calculer le temps total

        # Ajouter le message de l'utilisateur à l'historique et l'afficher
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="images/logoRageau.jpg"):
            with st.spinner("Analyse approfondie en cours..."):
                # Appel au workflow d'agent
                resultat = execute_mode_profond(prompt)

                # Bannière de résumé
                st.markdown("---")
                st.markdown("## 📌 RÉSUMÉ")
                st.markdown(resultat.summary)
                st.markdown("---")

                # Sections détaillées
                for i, section in enumerate(resultat.sections, 1):
                    with st.expander(f"{i}. {section.title}"):
                        st.markdown(section.content)
                        st.caption(f"Sources: {', '.join(section.sources)}")

                # Points clés
                if resultat.key_insights:
                    st.markdown("## 🔑 POINTS CLÉS")
                    for i, insight in enumerate(resultat.key_insights, 1):
                        st.markdown(f"**{i}.** {insight}")

                # Métadonnées
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if resultat.sources:
                        st.markdown(f"**📚 Sources utilisées:** {len(resultat.sources)}")
                        with st.expander("Voir les sources"):
                            for src in resultat.sources:
                                st.markdown(f"- {src}")

                with col2:
                    st.markdown(f"**🎯 Niveau de confiance:** {resultat.confidence * 100:.1f}%")

                    # Afficher le temps total
                    elapsed_time = time.time() - start_time
                    st.markdown(f"**⏱️ Temps total:** {elapsed_time:.2f}s")

                # Limitations si présentes
                if resultat.limitations:
                    st.warning(f"**⚠️ Limitations:** {resultat.limitations}")

            # Création de la réponse pour l'historique sans utiliser de f-string complexe
            full_response = "## 📌 RÉSUMÉ\n" + resultat.summary + "\n\n"

            # Ajouter les sections
            full_response += "## SECTIONS DÉTAILLÉES\n"
            for section in resultat.sections:
                full_response += f"### {section.title}\n{section.content}\n\n"

            # Ajouter les points clés
            full_response += "## 🔑 POINTS CLÉS\n"
            for insight in resultat.key_insights:
                full_response += f"- {insight}\n"

            # Ajouter les métadonnées
            full_response += f"\n## MÉTADONNÉES\n"
            full_response += f"- 📚 Sources: {len(resultat.sources)}\n"
            full_response += f"- 🎯 Confiance: {resultat.confidence * 100:.1f}%\n"

            # Ajouter les limitations si présentes
            if resultat.limitations:
                full_response += f"\n⚠️ **Limitations:** {resultat.limitations}"

        # Mise à jour de l'historique
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# Interface Streamlit pour drag and drop
uploaded_file = st.file_uploader("Glissez et déposez votre fichier", type=["pdf"])

    # Si un fichier est téléchargé
if uploaded_file is not None:
    st.write(f"Fichier téléchargé : {uploaded_file.name}")
    # Affiche un bouton pour valider l'upload
    if st.button("Valider l'upload sur MinIO"):
        # Appel de la fonction pour uploader le fichier sur MinIO
        file_name = upload_to_minio(uploaded_file)

