import os
import pickle
import numpy as np
from typing import List, Dict, Any
from .config import VECTOR_STORE_PATH, EMBEDDING_MODEL
import json

class VectorStore:
    def __init__(self, store_path: str = VECTOR_STORE_PATH, model_name: str = EMBEDDING_MODEL):
        self.store_path = store_path
        self.index_path = os.path.join(store_path, "faiss_index.bin")
        self.metadata_path = os.path.join(store_path, "metadata.pkl")
        self.model = None
        self.index = None
        self.metadata = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Try to import optional dependencies
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.model = SentenceTransformer(model_name)
            self.faiss = faiss
        except ImportError:
            print("WARNING: sentence-transformers or faiss not installed. RAG features will not work.")
            return
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()
        else:
            self.initialize()

    def initialize(self):
        """Initialize a new FAISS index."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        self.index = self.faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def load(self):
        """Load existing index and metadata."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        self.index = self.faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def save(self):
        """Save index and metadata to disk."""
        if self.index is not None:
            self.faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

    def add_schema_entries(self, schema: Dict[str, Any]):
        """Add schema entries to the vector store."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        
        # Clear existing data
        self.initialize()
        
        # Process schema into text chunks
        entries = []
        for table_name, table_info in schema.items():
            # Create table-level entry
            table_desc = f"Table {table_name}: {table_info.get('description', 'No description')}"
            entries.append(table_desc)
            
            # Create column-level entries
            for column in table_info.get('columns', []):
                col_desc = f"Column {table_name}.{column['name']}: {column.get('description', 'No description')} (Type: {column['type']})"
                entries.append(col_desc)
        
        # Generate embeddings and add to index
        embeddings = self.model.encode(entries)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = entries
        self.save()

    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for relevant schema entries."""
        if not self.model or not self.index:
            raise ImportError("Vector store not properly initialized")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search the index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            min(k, len(self.metadata))
        )
        
        # Return relevant entries
        return [self.metadata[i] for i in indices[0]]

def get_relevant_schema_from_rag(question: str, db_schema: Dict[str, Any]) -> str:
    """
    Retrieve relevant schema information using RAG.
    If vector store is empty, it will be populated with the current schema.
    """
    try:
        # Initialize vector store
        vector_store = VectorStore()
        
        # If vector store is empty, populate it
        if len(vector_store.metadata) == 0:
            print("Populating vector store with schema information...")
            vector_store.add_schema_entries(db_schema)
        
        # Search for relevant schema entries
        relevant_entries = vector_store.search(question)
        
        if not relevant_entries:
            return f"DATABASE SCHEMA:\n{json.dumps(db_schema, indent=2)}"
        
        # Format the relevant entries
        schema_context = "RELEVANT SCHEMA INFORMATION:\n"
        schema_context += "\n".join(relevant_entries)
        schema_context += "\n\nFULL SCHEMA:\n"
        schema_context += json.dumps(db_schema, indent=2)
        
        return schema_context
        
    except Exception as e:
        print(f"Error in RAG schema retrieval: {e}")
        return f"DATABASE SCHEMA:\n{json.dumps(db_schema, indent=2)}" 