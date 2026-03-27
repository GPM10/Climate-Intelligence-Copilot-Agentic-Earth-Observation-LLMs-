"""
Utility modules for logging, configuration, and common functions.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv


class Config:
    """Load and manage configuration from YAML and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        
        # Load from YAML
        if config_path:
            self._load_yaml(config_path)
        else:
            # Try default location
            default_path = Path(__file__).parent.parent.parent / 'config' / 'settings.yaml'
            if default_path.exists():
                self._load_yaml(str(default_path))
        
        # Load from environment
        load_dotenv()
        self._load_env()
    
    def _load_yaml(self, path: str):
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {path}")
    
    def _load_env(self):
        """Load configuration from environment variables."""
        env_config = {
            'llm': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'provider': os.getenv('LLM_PROVIDER', 'openai')
            },
            'gee': {
                'project_id': os.getenv('GEE_PROJECT_ID'),
                'service_account_path': os.getenv('GEE_SERVICE_ACCOUNT_PATH')
            },
            'database': {
                'url': os.getenv('DATABASE_URL', 'sqlite:///./data/climate_copilot.db')
            }
        }
        
        # Merge with YAML config
        for key, value in env_config.items():
            if key not in self.config:
                self.config[key] = {}
            self.config[key].update({k: v for k, v in value.items() if v})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self.config.copy()


class Logger:
    """Configure and get logger instances."""
    
    _initialized = False
    
    @staticmethod
    def setup(log_level: str = 'INFO', log_dir: str = './logs'):
        """Setup logging configuration."""
        
        if Logger._initialized:
            return
        
        # Create log directory
        Path(log_dir).mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/climate_copilot.log'),
                logging.StreamHandler()
            ]
        )
        
        Logger._initialized = True
    
    @staticmethod
    def get(name: str) -> logging.Logger:
        """Get logger instance."""
        return logging.getLogger(name)


class VectorStoreManager:
    """Manage FAISS vector database for embeddings and similarity search."""
    
    def __init__(self, index_path: str = './data/faiss_index', 
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.index = None
        self.metadata = []
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
    
    def add_documents(self, documents: list, embeddings: list) -> None:
        """
        Add documents and embeddings to vector store.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
        """
        
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings count mismatch")
        
        # Create index if not exists
        if self.index is None:
            embedding_dim = len(embeddings[0])
            self.index = self.faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings
        import numpy as np
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata.extend(documents)
    
    def search(self, query_embedding: list, top_k: int = 5) -> list:
        """
        Search for top-k similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
        
        Returns:
            List of (distance, document) tuples
        """
        
        if self.index is None:
            return []
        
        import numpy as np
        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, top_k)
        
        results = [
            (float(dist), self.metadata[idx])
            for dist, idx in zip(distances[0], indices[0])
        ]
        
        return results
    
    def save(self) -> None:
        """Save index to disk."""
        if self.index:
            self.faiss.write_index(self.index, self.index_path)
    
    def load(self) -> None:
        """Load index from disk."""
        try:
            self.index = self.faiss.read_index(self.index_path)
        except Exception as e:
            print(f"Could not load index: {e}")


__all__ = ['Config', 'Logger', 'VectorStoreManager']
