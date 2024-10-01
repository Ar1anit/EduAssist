import os
import streamlit as st
from openai import OpenAI
import openai
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

client = OpenAI()

def load_api_key():
    # Lade die Umgebungsvariablen aus der .env-Datei
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt. Bitte füge ihn zur .env-Datei hinzu.")
    OpenAI.api_key = openai_api_key
    openai.api_key = openai_api_key  # Füge dies hinzu, um openai direkt zu verwenden

def trim_conversation_history(conversation_history, max_tokens=1500):
    # Kürzt die Konversationshistorie, um das Token-Limit einzuhalten
    total_tokens = 0
    trimmed_history = []
    # Durchlaufe die Historie rückwärts
    for message in reversed(conversation_history):
        message_tokens = len(message['content'].split())
        total_tokens += message_tokens
        if total_tokens > max_tokens:
            break
        trimmed_history.insert(0, message)
    return trimmed_history

def answer_question(question, df, index, conversation_history):
    # Erstelle Embedding für die Frage
    response = client.embeddings.create(
        input=question,
        model='text-embedding-ada-002'
    )
    question_embedding = np.array(response.data[0].embedding).astype('float32')

    # Suche im FAISS-Index
    k = 5  # Anzahl der Ergebnisse
    distances, indices = index.search(np.array([question_embedding]), k)

    # Hole die entsprechenden Dokumente
    contexts = df.iloc[indices[0]]['content'].tolist()

    # Beschränke die Länge jedes Kontextes
    max_context_length = 1500  # Maximale Anzahl an Zeichen pro Kontext
    truncated_contexts = []
    for context in contexts:
        if len(context) > max_context_length:
            context = context[:max_context_length] + '...'
        truncated_contexts.append(context)

    # Erstelle den Prompt
    prompt = f"Beantworte die folgende Frage basierend auf den bereitgestellten Informationen.\n\n"
    for i, context in enumerate(truncated_contexts):
        prompt += f"Information {i+1}:\n{context}\n\n"
    prompt += f"Frage:\n{question}\nAntwort:"

    # Füge die aktuelle Benutzerfrage zur Konversationshistorie hinzu
    conversation_history.append({'role': 'user', 'content': prompt})

    # Kürze die Konversationshistorie, wenn nötig
    conversation_history_trimmed = trim_conversation_history(conversation_history)

    # Generiere die Antwort mit der ChatCompletion API mit Streaming
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=conversation_history_trimmed,
        max_tokens=150,
        temperature=0,
        n=1,
        stop=None,
        stream=True  # Streaming aktivieren
    )

    # Initialisiere eine leere Antwort
    answer = ''

    # Gib den Stream zurück
    return response

def main():
    # Lade den API-Schlüssel
    load_api_key()

    # Lade die Daten und den Index
    df = pd.read_pickle('../data/embeddings/embeddings.pkl')
    index = faiss.read_index('../data/embeddings/faiss_index.idx')

    st.title("Universitäts-Chatbot")

    # Initialisiere die Konversationshistorie im Session State
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = [
            {'role': 'system', 'content': 'Du bist ein hilfreicher Assistent.'}
        ]

    # Anzeige der Konversation
    for i in range(1, len(st.session_state.conversation_history), 2):
        user_msg = st.session_state.conversation_history[i]
        bot_msg = st.session_state.conversation_history[i+1] if i+1 < len(st.session_state.conversation_history) else None

        if user_msg['role'] == 'user':
            with st.chat_message("user"):
                # Extrahiere die Frage aus dem Prompt
                question = user_msg['content'].split('Frage:\n')[-1].split('\nAntwort:')[0]
                st.write(question)
        if bot_msg and bot_msg['role'] == 'assistant':
            with st.chat_message("assistant"):
                st.write(bot_msg['content'])

    # Eingabefeld am Ende
    if user_input := st.chat_input("Schreibe eine Nachricht"):
        # Zeige die Nachricht des Benutzers sofort an
        with st.chat_message("user"):
            st.write(user_input)
        # Platzhalter für die Antwort des Bots
        bot_message_placeholder = st.chat_message("assistant")
        with bot_message_placeholder:
            # Platzhalter für den gestreamten Text
            streamed_answer = st.empty()
            full_response = ''
            # Generiere die Antwort
            response = answer_question(user_input, df, index, st.session_state.conversation_history)
            try:
                for chunk in response:
                    chunk_message = chunk['choices'][0]['delta']
                    if 'content' in chunk_message:
                        content = chunk_message['content']
                        full_response += content
                        streamed_answer.markdown(full_response)
                # Füge die vollständige Antwort zur Konversationshistorie hinzu
                st.session_state.conversation_history.append({'role': 'assistant', 'content': full_response})
            except Exception as e:
                st.error(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == '__main__':
    main()
