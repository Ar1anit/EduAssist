import faiss
import numpy as np
import pandas as pd

def build_vector_store():
    # Lade die Embeddings
    df = pd.read_pickle('data/embeddings/embeddings.pkl')
    embeddings = np.array(df['embedding'].tolist()).astype('float32')
    
    # Erstelle den FAISS-Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Speichere den Index
    faiss.write_index(index, 'data/embeddings/faiss_index.idx')
    
    print("FAISS-Index wurde erfolgreich erstellt und gespeichert.")

if __name__ == '__main__':
    build_vector_store()
