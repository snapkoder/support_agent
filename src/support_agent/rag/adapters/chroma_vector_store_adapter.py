"""
ChromaDB Vector Store Adapter - Persistent local vector storage
Preferred implementation for local-first RAG
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path

from ..ports import VectorStorePort, DocumentChunk, RetrievedChunk, VectorStoreError

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

logger = logging.getLogger(__name__)

class ChromaVectorStoreAdapter(VectorStorePort):
    """
    ChromaDB adapter for persistent local vector storage
    """
    
    def __init__(self, persist_dir: str = "./var/rag_index/chroma", collection_name: str = "jota_kb"):
        """Initialize ChromaDB vector store adapter"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")
        
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.logger = logging.getLogger(f"{__name__}.ChromaVectorStoreAdapter")
        
        # Ensure persist directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.logger.info(f"Connected to existing ChromaDB collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Jota Knowledge Base"}
            )
            self.logger.info(f"Created new ChromaDB collection: {collection_name}")
        
        self.logger.info(f"ChromaDB Vector Store initialized at: {persist_dir}")
    
    async def upsert(self, chunks: List[DocumentChunk]) -> None:
        """
        Insert or update document chunks in ChromaDB.
        
        Args:
            chunks: List of chunks to store
        """
        if not chunks:
            return
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in chunks:
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                
                # Prepare metadata
                metadata = {
                    'chunk_id': chunk.chunk_id,
                    'source_file': chunk.source_file,
                    'section_title': chunk.section_title,
                    'domain': chunk.domain,
                    'agent_type': chunk.agent_type,
                    'language': chunk.language,
                    'version': chunk.version,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'breadcrumb': chunk.breadcrumb,
                    'content_type': chunk.content_type,
                    'priority': chunk.priority,
                }
                # Only add last_updated if it's not None (ChromaDB doesn't accept None in metadata)
                if chunk.last_updated:
                    metadata['last_updated'] = chunk.last_updated.isoformat()
                metadatas.append(metadata)
                
                # Add embedding if available
                if chunk.embedding:
                    embeddings.append(chunk.embedding)
                else:
                    # ChromaDB will generate embeddings if not provided
                    embeddings.append(None)
            
            # Use embeddings for ChromaDB if available
            embeddings_for_chroma = embeddings if any(e is not None for e in embeddings) else None
            
            # Upsert to ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_for_chroma
            )
            
            self.logger.info(f"Upserted {len(chunks)} chunks to ChromaDB")
            
        except Exception as e:
            self.logger.error(f"Failed to upsert chunks to ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB upsert failed: {e}")
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Search for similar chunks in ChromaDB.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of retrieved chunks with scores
        """
        try:
            # Prepare where clause for filters
            where_conditions = []
            if filters:
                for key, value in filters.items():
                    if key in ['domain', 'agent_type', 'language', 'content_type', 'source_file']:
                        where_conditions.append({key: {"$eq": value}})
            
            where_clause = {"$and": where_conditions} if where_conditions else None
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Convert to RetrievedChunk objects
            retrieved_chunks = []
            
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    document = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    # Create DocumentChunk
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=document,
                        source_file=metadata.get('source_file', ''),
                        section_title=metadata.get('section_title', ''),
                        domain=metadata.get('domain', ''),
                        agent_type=metadata.get('agent_type', ''),
                        language=metadata.get('language', 'pt-BR'),
                        version=metadata.get('version', '1.0'),
                        chunk_index=metadata.get('chunk_index', 0),
                        total_chunks=metadata.get('total_chunks', 0),
                        breadcrumb=metadata.get('breadcrumb', ''),
                        content_type=metadata.get('content_type', 'text'),
                        priority=metadata.get('priority', 1),
                        last_updated=metadata.get('last_updated')
                    )
                    
                    # Create RetrievedChunk
                    retrieved_chunk = RetrievedChunk(
                        chunk=chunk,
                        score=similarity_score,
                        retrieval_method='chroma_similarity'
                    )
                    
                    retrieved_chunks.append(retrieved_chunk)
            
            self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks from ChromaDB")
            return retrieved_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to search ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB search failed: {e}")
    
    async def persist(self) -> None:
        """Persist ChromaDB to disk (automatic with PersistentClient)"""
        try:
            # ChromaDB PersistentClient automatically persists
            # This is a no-op but kept for interface compatibility
            self.logger.debug("ChromaDB persistence (automatic with PersistentClient)")
        except Exception as e:
            self.logger.error(f"Failed to persist ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB persist failed: {e}")
    
    async def load(self) -> None:
        """Load ChromaDB from disk (automatic with PersistentClient)"""
        try:
            # ChromaDB PersistentClient automatically loads
            # This is a no-op but kept for interface compatibility
            self.logger.debug("ChromaDB load (automatic with PersistentClient)")
        except Exception as e:
            self.logger.error(f"Failed to load ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB load failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if ChromaDB is healthy"""
        try:
            # Try to get collection count
            count = self.collection.count()
            self.logger.debug(f"ChromaDB health check: {count} documents in collection")
            return True
        except Exception as e:
            self.logger.error(f"ChromaDB health check failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze domains
            sample_results = self.collection.get(
                limit=100,
                include=['metadatas']
            )
            
            domains = set()
            agent_types = set()
            source_files = set()
            
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    domains.add(metadata.get('domain', 'unknown'))
                    agent_types.add(metadata.get('agent_type', 'unknown'))
                    source_files.add(metadata.get('source_file', 'unknown'))
            
            return {
                'total_documents': count,
                'domains': list(domains),
                'agent_types': list(agent_types),
                'source_files': list(source_files),
                'persist_dir': self.persist_dir,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get ChromaDB stats: {e}")
            return {'error': str(e)}
    
    async def clear(self) -> None:
        """Clear all documents from ChromaDB"""
        try:
            # Delete all documents
            self.collection.delete(where={})
            self.logger.info("Cleared all documents from ChromaDB")
        except Exception as e:
            self.logger.error(f"Failed to clear ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB clear failed: {e}")
