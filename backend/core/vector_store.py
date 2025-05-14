import os
import pickle
import numpy as np
from typing import List, Dict, Any, Union, Optional
from .config import VECTOR_STORE_PATH, EMBEDDING_MODEL
import json
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

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

    def add_schema_entries(self, schema: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Add schema entries to the vector store.
        Handles various schema formats including the nested structure with datasets.
        """
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        
        # Clear existing data
        self.initialize()
        
        # Process schema into text chunks
        entries = []
        
        # Handle schema as the nested structure with datasets key
        if isinstance(schema, dict) and "datasets" in schema:
            for dataset_name, dataset_info in schema["datasets"].items():
                # Create dataset-level entry
                dataset_desc = f"Dataset {dataset_name}: {dataset_info.get('description', 'No description')}"
                if "type" in dataset_info:
                    dataset_desc += f" (Type: {dataset_info['type']})"
                entries.append(dataset_desc)
                
                # Create column-level entries
                if "columns" in dataset_info:
                    for col_name, col_info in dataset_info["columns"].items():
                        # Handle both dictionary and string descriptions
                        if isinstance(col_info, dict):
                            col_type = col_info.get("type", "Unknown")
                            col_desc = col_info.get("description", "No description")
                            entries.append(f"Column {dataset_name}.{col_name}: {col_desc} (Type: {col_type})")
                        else:
                            # If column info is just a string or other simple value
                            entries.append(f"Column {dataset_name}.{col_name}: {col_info}")
                
                # Add other dataset-specific information
                for key, value in dataset_info.items():
                    if key not in ["columns", "description", "type"]:
                        if isinstance(value, (dict, list)):
                            entry = f"Dataset {dataset_name} {key}: {json.dumps(value)}"
                        else:
                            entry = f"Dataset {dataset_name} {key}: {value}"
                        entries.append(entry)
                        
        # Handle schema as a dictionary mapping table names to table info
        elif isinstance(schema, dict):
            for table_name, table_info in schema.items():
                if isinstance(table_info, dict):
                    # Create table-level entry
                    table_desc = f"Table {table_name}: {table_info.get('description', 'No description')}"
                    entries.append(table_desc)
                    
                    # Create column-level entries
                    columns = table_info.get('columns', [])
                    if isinstance(columns, dict):
                        for col_name, col_info in columns.items():
                            if isinstance(col_info, dict):
                                col_desc = f"Column {table_name}.{col_name}: {col_info.get('description', 'No description')}"
                                if "type" in col_info:
                                    col_desc += f" (Type: {col_info['type']})"
                                entries.append(col_desc)
                            else:
                                entries.append(f"Column {table_name}.{col_name}: {col_info}")
                    elif isinstance(columns, list):
                        for column in columns:
                            col_desc = f"Column {table_name}.{column.get('name', 'Unknown')}: {column.get('description', 'No description')}"
                            if "type" in column:
                                col_desc += f" (Type: {column['type']})"
                            entries.append(col_desc)
        
        # Handle schema as a list of table dictionaries
        elif isinstance(schema, list):
            for table in schema:
                if isinstance(table, dict):
                    table_name = table.get('name', 'Unknown')
                    # Create table-level entry
                    table_desc = f"Table {table_name}: {table.get('description', 'No description')}"
                    entries.append(table_desc)
                    
                    # Create column-level entries
                    for column in table.get('columns', []):
                        if isinstance(column, dict):
                            col_desc = f"Column {table_name}.{column.get('name', 'Unknown')}: {column.get('description', 'No description')}"
                            if "type" in column:
                                col_desc += f" (Type: {column['type']})"
                            entries.append(col_desc)
        
        # Handle any key_metrics or other top-level elements in the schema
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key not in ["datasets", "description"] and isinstance(value, (list, dict)):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and "name" in item:
                                entries.append(f"{key} - {item['name']}: {item.get('description', item.get('calculation', 'No description'))}")
                    elif isinstance(value, dict):
                        for item_name, item_info in value.items():
                            if isinstance(item_info, dict):
                                entries.append(f"{key} - {item_name}: {item_info.get('description', 'No description')}")
        
        if not entries:
            print("Warning: No schema entries were created - schema format might be incorrect")
            return
            
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

class ConversationStore:
    def __init__(self, store_path: str = "data/vector_store"):
        self.store_path = store_path
        self.index_path = os.path.join(store_path, "conversation_index.faiss")
        self.data_path = os.path.join(store_path, "conversation_data.pkl")
        self.model = None
        self.index = None
        self.conversations = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Create directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)
        
        # Initialize model and index
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            if os.path.exists(self.index_path) and os.path.exists(self.data_path):
                self.load()
            else:
                self.initialize()
        except ImportError:
            print("WARNING: sentence-transformers not installed. Conversation history features will not work.")

    def initialize(self):
        """Initialize a new FAISS index for conversations."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.conversations = []

    def load(self):
        """Load existing conversation index and data."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        self.index = faiss.read_index(self.index_path)
        with open(self.data_path, 'rb') as f:
            self.conversations = pickle.load(f)

    def save(self):
        """Save conversation index and data to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.data_path, 'wb') as f:
                pickle.dump(self.conversations, f)

    def add_conversation(self, conversation_data: Dict[str, Any]):
        """Add a conversation to the store."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        
        # Add timestamp
        conversation_data['timestamp'] = datetime.now().isoformat()
        
        # Create text representation for embedding
        text_to_embed = f"Question: {conversation_data['question']}\n"
        text_to_embed += f"Response: {conversation_data['response']}\n"
        if conversation_data.get('results'):
            text_to_embed += f"Results: {json.dumps(conversation_data['results'][:5])}\n"
        
        # Generate embedding
        embedding = self.model.encode([text_to_embed])[0]
        
        # Add to index and data
        self.index.add(np.array([embedding]))
        self.conversations.append(conversation_data)
        
        # Save updated index and data
        self.save()

    def search_conversations(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant conversations."""
        if not self.model or not self.index or not self.conversations:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search for similar conversations
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        # Return relevant conversations
        return [self.conversations[i] for i in indices[0] if i < len(self.conversations)]

    def clear(self):
        """Clear all stored conversations and the vector index."""
        self.initialize()
        self.save()

def get_relevant_schema_from_rag(question: str, db_schema: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Retrieve relevant schema information using RAG.
    If vector store is empty, it will be populated with the current schema.
    """
    print('Vector store is being used.')
    try:
        # Initialize vector store
        vector_store = VectorStore()
        print('Vector store initialised')
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

def store_conversation(conversation_data: Dict[str, Any]):
    """
    Store a conversation in the vector store.
    """
    try:
        conversation_store = ConversationStore()
        conversation_store.add_conversation(conversation_data)
    except Exception as e:
        print(f"Error storing conversation: {e}")

def get_relevant_conversations(question: str, k: int = 1) -> List[Dict[str, Any]]:
    """
    Retrieve relevant conversations based on the question.
    """
    try:
        conversation_store = ConversationStore()
        return conversation_store.search_conversations(question, k)
    except Exception as e:
        print(f"Error retrieving conversations: {e}")
        return []

def clear_conversation_history():
    """
    Clear all stored conversations and the vector index.
    """
    try:
        conversation_store = ConversationStore()
        conversation_store.clear()
    except Exception as e:
        print(f"Error clearing conversation history: {e}")
        raise