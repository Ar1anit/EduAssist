import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

client = OpenAI()

def load_api_key():
    # Lade die Umgebungsvariablen aus der .env-Datei
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt. Bitte füge ihn zur .env-Datei hinzu.")
    OpenAI.api_key = openai_api_key

def count_tokens(text, model_name='text-embedding-3-small'):
    # Berechne die Anzahl der Tokens im Text
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def chunk_text(text, max_tokens=8000, model_name='text-embedding-3-small'):
    # Teile den Text in kleinere Abschnitte, basierend auf der Token-Anzahl
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return [encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

def create_embeddings():
    # Lade den API-Schlüssel
    load_api_key()
    
    # Lade die verarbeiteten Daten
    df = pd.read_csv('data/processed/processed_data.csv')
    
    # Erstelle Embeddings
    embeddings = []
    for text in df['content']:
        # Teile den Text in kleinere Abschnitte, wenn er zu lang ist
        chunks = chunk_text(text, max_tokens=8000)
        text_embeddings = []
        for chunk in chunks:
            # Erstelle Embedding für jeden Abschnitt
            response = client.embeddings.create(
                input=chunk,
                model='text-embedding-3-small'
            )
            embedding = response.data[0].embedding
            text_embeddings.append(embedding)
        
        # Durchschnittliches Embedding pro Text
        # Dies kombiniert die Embeddings der Abschnitte zu einem Gesamtembedding
        embedding = [sum(x) / len(x) for x in zip(*text_embeddings)]
        embeddings.append(embedding)
    
    # Füge Embeddings zum DataFrame hinzu
    df['embedding'] = embeddings
    
    # Speichere die Embeddings
    df.to_pickle('data/embeddings/embeddings.pkl')

if __name__ == '__main__':
    create_embeddings()
