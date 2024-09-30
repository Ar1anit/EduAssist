import os
from openai import OpenAI
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

client = OpenAI()

conversation_history = [
    {'role': 'system', 'content': """
       Du bist ein hilfreicher Assistent, der für die Studenten und Mitarbeiter der Universität Duisburg-Essen. Beantworte die Frage anhand der dir gegebenen Informationen. Es werden speziell Fragen über Vorlesungen, 
        Inforamtionen zur Universität und allgemeine Fragen über das Studium erwartet. Bitte sei stets freundlich und profesionell.
     """}
]

def load_api_key():
    # Lade die Umgebungsvariablen aus der .env-Datei
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt. Bitte füge ihn zur .env-Datei hinzu.")
    OpenAI.api_key = openai_api_key

def trim_conversation_history(max_tokens=1500):
    total_tokens = 0
    trimmed_history = []
    # Wir gehen die Historie rückwärts durch
    for message in reversed(conversation_history):
        message_tokens = len(message['content'].split())
        total_tokens += message_tokens
        if total_tokens > max_tokens:
            break
        trimmed_history.insert(0, message)
    return trimmed_history

def answer_question(question, df, index):
    global conversation_history  # Damit wir auf die globale Variable zugreifen können
    
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
    
    # Generiere die Antwort mit der ChatCompletion API
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=conversation_history,
        max_tokens=150,
        temperature=0,
        n=1,
        stop=None
    )
    
    # Hole die Antwort und füge sie zur Konversationshistorie hinzu
    answer = completion.choices[0].message.content
    conversation_history.append({'role': 'assistant', 'content': answer})
    
    return answer

if __name__ == '__main__':
    # Lade den API-Schlüssel
    load_api_key()
    
    # Lade die Daten und den Index
    df = pd.read_pickle('data/embeddings/embeddings.pkl')
    index = faiss.read_index('data/embeddings/faiss_index.idx')
    
    # Starte den Chatbot
    print("Willkommen zum Universitäts-Chatbot! Stelle deine Frage oder tippe 'exit', um zu beenden.")
    while True:
        question = input("\nDeine Frage: ")
        if question.lower() == 'exit':
            print("Auf Wiedersehen!")
            break
        try:
            answer = answer_question(question, df, index)
            print(f"\nAntwort: {answer}")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")
