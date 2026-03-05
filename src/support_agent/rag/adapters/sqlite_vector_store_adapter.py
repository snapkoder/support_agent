"""
SQLite Vector Store Adapter - Alternative persistent storage using SQLite + cosine similarity
Fallback option when ChromaDB is not available
"""

import os
import sqlite3
import json
import math
import logging
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path

from ..ports import VectorStorePort, DocumentChunk, RetrievedChunk, VectorStoreError

logger = logging.getLogger(__name__)

class SQLiteVectorStoreAdapter(VectorStorePort):
    """
    SQLite adapter for persistent local vector storage
    Uses SQLite for metadata and cosine similarity for search
    """
    
    def __init__(self, db_path: str = "./var/rag_index/vectors.sqlite"):
        """Initialize SQLite vector store adapter"""
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.SQLiteVectorStoreAdapter")
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"SQLite Vector Store initialized at: {db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create chunks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source_file TEXT,
                        section_title TEXT,
                        domain TEXT,
                        agent_type TEXT,
                        language TEXT DEFAULT 'pt-BR',
                        version TEXT DEFAULT '1.0',
                        chunk_index INTEGER,
                        total_chunks INTEGER,
                        breadcrumb TEXT,
                        content_type TEXT DEFAULT 'text',
                        priority INTEGER DEFAULT 1,
                        last_updated TEXT,
                        embedding BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON chunks(domain)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_type ON chunks(agent_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_file ON chunks(source_file)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite database: {e}")
            raise VectorStoreError(f"SQLite initialization failed: {e}")
    
    async def upsert(self, chunks: List[DocumentChunk]) -> None:
        """
        Insert or update document chunks in SQLite.
        
        Args:
            chunks: List of chunks to store
        """
        if not chunks:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for chunk in chunks:
                    # Serialize embedding
                    embedding_blob = json.dumps(chunk.embedding) if chunk.embedding else None
                    
                    # Upsert chunk
                    cursor.execute('''
                        INSERT OR REPLACE INTO chunks 
                        (chunk_id, content, source_file, section_title, domain, agent_type,
                         language, version, chunk_index, total_chunks, breadcrumb, content_type,
                         priority, last_updated, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        chunk.chunk_id,
                        chunk.content,
                        chunk.source_file,
                        chunk.section_title,
                        chunk.domain,
                        chunk.agent_type,
                        chunk.language,
                        chunk.version,
                        chunk.chunk_index,
                        chunk.total_chunks,
                        chunk.breadcrumb,
                        chunk.content_type,
                        chunk.priority,
                        chunk.last_updated.isoformat() if chunk.last_updated else None,
                        embedding_blob
                    ))
                
                conn.commit()
                
            self.logger.info(f"Upserted {len(chunks)} chunks to SQLite")
            
        except Exception as e:
            self.logger.error(f"Failed to upsert chunks to SQLite: {e}")
            raise VectorStoreError(f"SQLite upsert failed: {e}")
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of retrieved chunks with scores
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build WHERE clause for filters
                where_conditions = []
                params = []
                
                if filters:
                    for key, value in filters.items():
                        if key in ['domain', 'agent_type', 'language', 'content_type', 'source_file']:
                            where_conditions.append(f"{key} = ?")
                            params.append(value)
                
                where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                
                # Get all chunks with filters
                cursor.execute(f'''
                    SELECT chunk_id, content, source_file, section_title, domain, agent_type,
                           language, version, chunk_index, total_chunks, breadcrumb, content_type,
                           priority, last_updated, embedding
                    FROM chunks {where_clause}
                ''', params)
                
                rows = cursor.fetchall()
                
                # Calculate cosine similarity for each chunk
                similarities = []
                for row in rows:
                    chunk_data = list(row)
                    embedding_blob = chunk_data[-1]
                    
                    if embedding_blob:
                        try:
                            chunk_embedding = json.loads(embedding_blob)
                            similarity = self._cosine_similarity(query_vector, chunk_embedding)
                            similarities.append((similarity, chunk_data))
                        except Exception as e:
                            self.logger.warning(f"Failed to parse embedding for chunk {chunk_data[0]}: {e}")
                            continue
                
                # Sort by similarity (descending) and take top_k
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_similarities = similarities[:top_k]
                
                # Convert to RetrievedChunk objects
                retrieved_chunks = []
                for similarity, chunk_data in top_similarities:
                    chunk = DocumentChunk(
                        chunk_id=chunk_data[0],
                        content=chunk_data[1],
                        source_file=chunk_data[2] or '',
                        section_title=chunk_data[3] or '',
                        domain=chunk_data[4] or '',
                        agent_type=chunk_data[5] or '',
                        language=chunk_data[6] or 'pt-BR',
                        version=chunk_data[7] or '1.0',
                        chunk_index=chunk_data[8] or 0,
                        total_chunks=chunk_data[9] or 0,
                        breadcrumb=chunk_data[10] or '',
                        content_type=chunk_data[11] or 'text',
                        priority=chunk_data[12] or 1,
                        last_updated=chunk_data[13]
                    )
                    
                    retrieved_chunk = RetrievedChunk(
                        chunk=chunk,
                        score=similarity,
                        retrieval_method='sqlite_cosine'
                    )
                    
                    retrieved_chunks.append(retrieved_chunk)
                
                self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks from SQLite")
                return retrieved_chunks
                
        except Exception as e:
            self.logger.error(f"Failed to search SQLite: {e}")
            raise VectorStoreError(f"SQLite search failed: {e}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def persist(self) -> None:
        """Persist SQLite database (automatic)"""
        try:
            # SQLite automatically persists
            self.logger.debug("SQLite persistence (automatic)")
        except Exception as e:
            self.logger.error(f"Failed to persist SQLite: {e}")
            raise VectorStoreError(f"SQLite persist failed: {e}")
    
    async def load(self) -> None:
        """Load SQLite database (automatic)"""
        try:
            # SQLite automatically loads
            self.logger.debug("SQLite load (automatic)")
        except Exception as e:
            self.logger.error(f"Failed to load SQLite: {e}")
            raise VectorStoreError(f"SQLite load failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if SQLite vector store is healthy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                count = cursor.fetchone()[0]
                self.logger.debug(f"SQLite health check: {count} documents in database")
                return True
        except Exception as e:
            self.logger.error(f"SQLite health check failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get SQLite statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total documents
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_docs = cursor.fetchone()[0]
                
                # Domains
                cursor.execute("SELECT DISTINCT domain FROM chunks")
                domains = [row[0] for row in cursor.fetchall() if row[0]]
                
                # Agent types
                cursor.execute("SELECT DISTINCT agent_type FROM chunks")
                agent_types = [row[0] for row in cursor.fetchall() if row[0]]
                
                # Source files
                cursor.execute("SELECT DISTINCT source_file FROM chunks")
                source_files = [row[0] for row in cursor.fetchall() if row[0]]
                
                return {
                    'total_documents': total_docs,
                    'domains': domains,
                    'agent_types': agent_types,
                    'source_files': source_files,
                    'db_path': self.db_path,
                    'db_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get SQLite stats: {e}")
            return {'error': str(e)}
    
    async def clear(self) -> None:
        """Clear all documents from SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chunks")
                conn.commit()
                
            self.logger.info("Cleared all documents from SQLite")
        except Exception as e:
            self.logger.error(f"Failed to clear SQLite: {e}")
            raise VectorStoreError(f"SQLite clear failed: {e}")
