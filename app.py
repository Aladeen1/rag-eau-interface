from typing import List, Dict, Any

from mistralai import Mistral, UserMessage
import psycopg2
import streamlit as st

from utils import retrieve_context, vectorize_query

db_connection_string = st.secrets['SUPABASE_PG_URL']
conn = psycopg2.connect(db_connection_string)

st.title("Mistral Chatbot")

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
if prompt := st.chat_input("Ask me something!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    input_vector = vectorize_query(prompt, client)
    context = retrieve_context(conn, input_vector)

    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    # Call Mistral API
    with st.chat_message("assistant"):
        response = client.chat.complete(
            model=st.session_state["mistral_model"],
            messages=messages
        )
        assistant_reply = response.choices[0].message.content
        st.markdown(assistant_reply)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
