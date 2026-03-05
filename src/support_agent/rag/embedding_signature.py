"""
Embedding Signature - Single Source of Truth for Embedding Configuration
Define a unique signature for embedding adapters to prevent dimension mismatches
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from enum import Enum

class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"

@dataclass
class EmbeddingSignature:
    """Unique signature for embedding configuration"""
    provider: EmbeddingProvider
    model_name: str
    dimensions: int
    normalize: bool = False
    extra: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to dictionary"""
        result = asdict(self)
        result['provider'] = self.provider.value
        return result
    
    def stable_hash(self) -> str:
        """Generate stable hash for this signature"""
        # Create deterministic JSON
        signature_dict = self.to_dict()
        
        # Sort keys for deterministic hash
        sorted_dict = json.dumps(signature_dict, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(sorted_dict.encode('utf-8')).hexdigest()[:16]
    
    def is_compatible_with(self, other: 'EmbeddingSignature') -> bool:
        """Check if this signature is compatible with another"""
        return (
            self.provider == other.provider and
            self.model_name == other.model_name and
            self.dimensions == other.dimensions and
            self.normalize == other.normalize
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingSignature':
        """Create signature from dictionary"""
        provider_str = data.get('provider', 'local')
        provider = EmbeddingProvider(provider_str) if isinstance(provider_str, str) else provider_str
        
        return cls(
            provider=provider,
            model_name=data.get('model_name', 'unknown'),
            dimensions=data.get('dimensions', 0),
            normalize=data.get('normalize', False),
            extra=data.get('extra')
        )
    
    def __str__(self) -> str:
        """String representation of signature"""
        return f"{self.provider.value}:{self.model_name}:{self.dimensions}d"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"EmbeddingSignature({self.__str__()})"

# Common signature presets
OPENAI_TEXT_EMBEDDING_3_SMALL = EmbeddingSignature(
    provider=EmbeddingProvider.OPENAI,
    model_name="text-embedding-3-small",
    dimensions=1536,
    normalize=False
)

OPENAI_TEXT_EMBEDDING_3_LARGE = EmbeddingSignature(
    provider=EmbeddingProvider.OPENAI,
    model_name="text-embedding-3-large",
    dimensions=3072,
    normalize=False
)

LOCAL_TFIDF_384 = EmbeddingSignature(
    provider=EmbeddingProvider.LOCAL,
    model_name="tfidf",
    dimensions=384,
    normalize=True
)

LOCAL_MINILM_384 = EmbeddingSignature(
    provider=EmbeddingProvider.LOCAL,
    model_name="minilm",
    dimensions=384,
    normalize=False
)
