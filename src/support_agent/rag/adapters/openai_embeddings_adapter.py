"""
OpenAI Embeddings Adapter - Concrete implementation using OpenAI API
"""

import logging
import time
import os
import threading
from typing import List, Dict, Any, Optional
from openai import OpenAI

from ..ports import EmbeddingsPort, EmbeddingError
from ..models import EmbeddingSignature, EmbeddingProvider, OPENAI_TEXT_EMBEDDING_3_SMALL, OPENAI_TEXT_EMBEDDING_3_LARGE

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

logger = logging.getLogger(__name__)

class OpenAIEmbeddingsAdapter(EmbeddingsPort):
    """OpenAI embeddings adapter with singleton client"""
    
    _client: Optional[OpenAI] = None
    _client_config: Optional[Dict[str, Any]] = None
    _client_lock = threading.Lock()
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """Initialize OpenAI embeddings adapter"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key or self.api_key.startswith('YOUR_'):
            raise EmbeddingError("Valid OPENAI_API_KEY required for OpenAI embeddings")
        
        self.signature = EmbeddingSignature(
            provider=EmbeddingProvider.OPENAI,
            model_name=self.model_name,
            dimensions=self._get_dimensions_for_model(model_name),
            normalize=False
        )
        
        # Initialize client lazily
        self._ensure_client()
        
        logger.info(f"OpenAI Embeddings Adapter initialized: {self.signature.provider}:{self.signature.model_name}:{self.signature.dimensions}d")
    
    def _ensure_client(self):
        """Ensure OpenAI client is initialized (singleton pattern with thread safety)"""
        current_config = {
            'api_key': self.api_key,
            'model_name': self.model_name
        }
        
        # Use lock to prevent race conditions
        with OpenAIEmbeddingsAdapter._client_lock:
            # Reuse client if config hasn't changed
            if (self._client is None or 
                self._client_config != current_config):
                
                self._client = OpenAI(api_key=self.api_key)
                self._client_config = current_config
                logger.debug("OpenAI client (re)initialized")
    
    @staticmethod
    def _get_dimensions_for_model(model_name: str) -> int:
        """Get dimensions for OpenAI model"""
        dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "text-embedding-davinci-003": 12288
        }
        return dimensions_map.get(model_name, 1536)  # Default to 1536
    
    def get_signature(self) -> EmbeddingSignature:
        """Get the embedding signature for this adapter"""
        return self.signature
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Ensure client is available
        self._ensure_client()
        
        try:
            # Process in batches to respect rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Create embeddings
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting delay
                if i + batch_size < len(texts):
                    time.sleep(0.1)
            
            self.logger.info(f"Generated {len(all_embeddings)} embeddings using {self.model_name}")
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings with OpenAI: {e}")
            raise EmbeddingError(f"OpenAI embedding generation failed: {e}")
    
    def health_check(self) -> bool:
        """Check if OpenAI embeddings service is healthy"""
        try:
            # Test with a simple embedding
            test_embedding = self.embed(["health check test"])
            return len(test_embedding) > 0 and len(test_embedding[0]) > 0
        except Exception as e:
            self.logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        return self._model_info.copy()
