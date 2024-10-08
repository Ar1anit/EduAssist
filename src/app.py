import os
import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def load_api_key():
    # Lade die Umgebungsvariablen aus der .env-Datei
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt. Bitte füge ihn zur .env-Datei hinzu.")
    OpenAI.api_key = openai_api_key

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

def semantic_search(query_embedding, document_embeddings, top_k=5):
    # Berechne Cosinus-Ähnlichkeit zwischen Query und allen Dokumenten
    similarities = cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)[0]
    
    # Sortiere die Indizes basierend auf der Ähnlichkeit (absteigend)
    top_indices = similarities.argsort()[::-1][:top_k]
    
    # Gebe die Top-K Indizes und ihre Ähnlichkeitswerte zurück
    return [(idx, similarities[idx]) for idx in top_indices]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def answer_question(question, df, index, conversation_history):
    try:
        # Erstelle Embedding für die Frage
        response = client.embeddings.create(
            input=question,
            model='text-embedding-ada-002'
        )
        question_embedding = np.array(response.data[0].embedding).astype('float32')

        # Suche im FAISS-Index
        k = 10  # Anzahl der Ergebnisse
        distances, indices = index.search(np.array([question_embedding]), k)

        # Wähle nur Dokumente aus, die über einem Ähnlichkeitsschwellenwert liegen
        threshold = 0.7  # Anpassen nach Bedarf
        relevant_indices = [i for i, d in zip(indices[0], distances[0]) if d < threshold]
        contexts = df.iloc[relevant_indices]['content'].tolist()

        # Beschränke die Länge jedes Kontextes
        max_context_length = 1500  # Maximale Anzahl an Zeichen pro Kontext
        truncated_contexts = []
        for context in contexts:
            if len(context) > max_context_length:
                context = context[:max_context_length] + '...'
            truncated_contexts.append(context)

        # Erstelle den Prompt
        prompt = f"""
            Basierend auf den folgenden Informationen, beantworte die Frage so präzise und hilfreich wie möglich. 
            Wenn die Informationen nicht ausreichen, sage ehrlich, dass du die Frage nicht beantworten kannst.
            Beziehe dich in deiner Antwort auf die relevantesten Teile der bereitgestellten Informationen.

            Informationen:
            {' '.join(truncated_contexts)}

            Frage: {question}
            Antwort:
        """
        for i, context in enumerate(truncated_contexts):
            prompt += f"Information {i+1}:\n{context}\n\n"
        prompt += f"Frage:\n{question}\nAntwort:"

        # Füge die aktuelle Benutzerfrage zur Konversationshistorie hinzu
        conversation_history.append({'role': 'user', 'content': prompt})

        # Kürze die Konversationshistorie, wenn nötig
        conversation_history_trimmed = trim_conversation_history(conversation_history)

        # Generiere die Antwort mit der ChatCompletion API (gestreamt)
        stream = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=conversation_history_trimmed,
            max_tokens=150,
            temperature=0.3,
            top_p=0.9,
            n=1,
            stop=None,
            stream=True
        )

        # Verarbeite den Stream und gib die Antwort stückweise zurück
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except client.APIError as e:
        yield f"OpenAI API error: {str(e)}"
    except client.APIConnectionError as e:
        yield f"Failed to connect to OpenAI API: {str(e)}"
    except client.RateLimitError as e:
        yield f"OpenAI API request exceeded rate limit: {str(e)}"
    except Exception as e:
        yield f"An unexpected error occurred: {str(e)}"

def suggest_followup_questions(answer):
    prompt = f"Basierend auf dieser Antwort, schlage 3 relevante Followup-Fragen vor:\n\n{answer}\n\nFollowup-Fragen:"
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content.split('\n')

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
        
        # Generiere und zeige die Antwort des Bots an
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response_chunk in answer_question(user_input, df, index, st.session_state.conversation_history):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        followup_questions = suggest_followup_questions(full_response)
        st.write("Mögliche Followup-Fragen:")
        for question in followup_questions:
            st.write(question)
        
        # Aktualisiere die Konversationshistorie nur, wenn keine Fehlermeldung zurückgegeben wurde
        if not any(error_message in full_response for error_message in ["API error", "Failed to connect", "rate limit", "unexpected error"]):
            st.session_state.conversation_history.append({'role': 'user', 'content': user_input})
            st.session_state.conversation_history.append({'role': 'assistant', 'content': full_response})
        else:
            st.error("Ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut.")

if __name__ == '__main__':
    main()