"""
Local Embeddings Adapter - Fallback implementation using TF-IDF
Optimized for local deployment without external dependencies
"""

import re
import math
import asyncio
import logging
from typing import List, Dict, Any, Optional
from collections import Counter
import hashlib

from ..ports import EmbeddingsPort, EmbeddingError
from ..models import (
    EmbeddingSignature, 
    LOCAL_TFIDF_384, LOCAL_MINILM_384
)

logger = logging.getLogger(__name__)

class LocalEmbeddingsAdapter(EmbeddingsPort):
    """
    Local embeddings adapter using optimized TF-IDF
    Fallback option when OpenAI is not available
    """
    
    def __init__(self, vector_size: int = 384):
        """Initialize local embeddings adapter"""
        self.vector_size = vector_size
        self.logger = logging.getLogger(f"{__name__}.LocalEmbeddingsAdapter")
        
        # Create signature based on vector size
        if vector_size == 384:
            self._signature = LOCAL_TFIDF_384
        else:
            self._signature = EmbeddingSignature(
                provider="local",
                model_name="tfidf",
                dimensions=vector_size,
                normalize=True
            )
        
        # TF-IDF components
        self.vocab = {}
        self.idf_cache = {}
        self.vocab_built = False
        
        # Embedding cache
        self._embedding_cache = {}
        self._cache_size = 1000
        
        # Compiled regex for performance
        self._word_pattern = re.compile(r'\b\w+\b')
        
        self.logger.info(f"Local Embeddings Adapter initialized: {self._signature}")
    
    def get_signature(self) -> EmbeddingSignature:
        """Get the embedding signature for this adapter"""
        return self._signature
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using TF-IDF.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Filter valid texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return []
            
            # Build vocabulary if not done
            if not self.vocab_built:
                self._build_vocab(valid_texts)
            
            # Generate embeddings
            embeddings = []
            for text in valid_texts:
                embedding = await self._get_single_embedding(text)
                embeddings.append(embedding)
            
            self.logger.info(f"Generated {len(embeddings)} local embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate local embeddings: {e}")
            raise EmbeddingError(f"Local embedding generation failed: {e}")
    
    def _build_vocab(self, texts: List[str]) -> None:
        """Build TF-IDF vocabulary from texts"""
        self.logger.info("Building TF-IDF vocabulary...")
        
        # Collect all words
        all_words = []
        for text in texts:
            words = self._tokenize(text.lower())
            all_words.extend(words)
        
        # Calculate document frequency
        total_docs = len(texts)
        doc_freq = Counter(all_words)
        
        # Filter frequent words (performance optimization)
        min_freq = max(2, total_docs // 100)  # At least 2% frequency
        frequent_words = {word: freq for word, freq in doc_freq.items() if freq >= min_freq}
        
        # Calculate IDF scores
        for word, freq in frequent_words.items():
            self.idf_cache[word] = math.log(total_docs / freq)
        
        self.vocab = frequent_words
        self.vocab_built = True
        
        self.logger.info(f"Built TF-IDF vocab with {len(self.vocab)} terms (filtered from {len(doc_freq)})")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return self._word_pattern.findall(text.lower())
    
    async def _get_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        # Check cache first
        text_hash = self._get_text_hash(text)
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Generate TF-IDF embedding
        words = self._tokenize(text.lower())
        embedding = [0.0] * self.vector_size
        
        # Calculate TF-IDF for each word
        for i, word in enumerate(words):
            if word in self.idf_cache and i < self.vector_size:
                tf = words.count(word) / len(words)
                embedding[i] = tf * self.idf_cache[word]
        
        # Normalize embedding
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        # Cache result
        self._cache_embedding(text_hash, embedding)
        
        return embedding
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _cache_embedding(self, text_hash: str, embedding: List[float]) -> None:
        """Cache embedding with size limit"""
        if len(self._embedding_cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[text_hash] = embedding
    
    async def health_check(self) -> bool:
        """Check if local embeddings service is healthy"""
        try:
            # Test with a simple embedding
            test_embedding = await self.embed(["health check test"])
            return len(test_embedding) > 0 and len(test_embedding[0]) > 0
        except Exception as e:
            self.logger.error(f"Local embeddings health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        return {
            'provider': 'local',
            'model_name': 'tfidf',
            'dimensions': self.vector_size,
            'vocab_size': len(self.vocab),
            'cache_size': len(self._embedding_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self._embedding_cache.clear()
        self.logger.info("Local embeddings cache cleared")
    
    def reset_vocab(self) -> None:
        """Reset vocabulary (useful for testing)"""
        self.vocab.clear()
        self.idf_cache.clear()
        self.vocab_built = False
        self.clear_cache()
        self.logger.info("Local embeddings vocabulary reset")
