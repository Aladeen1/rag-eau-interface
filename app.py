from typing import List, Dict, Any
import time

from mistralai import Mistral, UserMessage
import psycopg2
import streamlit as st

from utils import (retrieve_context,
                   vectorize_query,
                   format_chunks_with_bullets,
                   system_prompt,
                   parse_mistral_response)

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
    # Dans votre section de traitement de la réponse
    with st.chat_message("assistant", avatar="images/logoRageau.jpg"):
        stream_response = client.chat.stream(
            model=st.session_state["mistral_model"],
            messages=messages,
        )
        full_response = ""
        placeholder = st.empty()
        for chunk in stream_response:
            content = chunk.data.choices[0].delta.content
            if content:
                full_response += content
                placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                time.sleep(0.05)

        # Approche simplifiée: vérifier simplement si des marqueurs de structure sont présents
        has_structure = "RESUME:" in full_response and "DEVELOPPEMENT:" in full_response

        if has_structure:
            # Effacer le placeholder
            placeholder.empty()

            # Parser la réponse
            parsed_response = parse_mistral_response(full_response)

            # Afficher formaté
            st.markdown("## :pushpin: Résumé")
            st.markdown(f"\n{parsed_response.resume}\n")

            st.markdown("---")

            st.markdown("## :books: Développement")
            st.markdown(parsed_response.developpement)

            if parsed_response.references:
                st.markdown("## :bookmark: Références")
                for i, ref in enumerate(parsed_response.references, 1):
                    st.markdown(f"**[{i}]** {ref}")

            with st.expander("ℹ️ Informations complémentaires"):
                st.markdown("Les réponses fournies par le modèle sont basées sur une partie de la documentation d'ADES.")
        else:
            # Laisser la réponse telle quelle
            placeholder.markdown(full_response, unsafe_allow_html=True)

    # Ajouter à l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})
