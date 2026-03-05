"""
RAG Indexer - Builds and maintains the RAG index
Handles document loading, chunking, and index persistence
"""

import os
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import time
from datetime import datetime

from .ports import DocumentChunk, KnowledgeBasePort, VectorStorePort, EmbeddingsPort
from .models import IndexMetadata, RAGConfig, validate_chunk, calculate_corpus_hash, calculate_config_hash
from .rag_facade import RAGFacade

logger = logging.getLogger(__name__)

class RAGIndexer:
    """
    RAG Indexer - Builds and maintains the RAG index
    
    This indexer handles:
    - Loading documents from knowledge base
    - Chunking documents with metadata
    - Building embeddings
    - Persisting index with fingerprint
    """
    
    def __init__(
        self,
        knowledge_base_port: KnowledgeBasePort,
        vector_store_port: VectorStorePort,
        embeddings_port: EmbeddingsPort,
        config: Optional[RAGConfig] = None
    ):
        """Initialize RAG indexer with required ports"""
        self.knowledge_base_port = knowledge_base_port
        self.vector_store_port = vector_store_port
        self.embeddings_port = embeddings_port
        self.config = config or RAGConfig.from_env()
        
        self.logger = logging.getLogger(f"{__name__}.RAGIndexer")
        
        # Ensure index directory exists
        Path(self.config.index_dir).mkdir(parents=True, exist_ok=True)
    
    async def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build the RAG index from knowledge base documents.
        
        Args:
            force_rebuild: Force rebuild even if index is up-to-date
            
        Returns:
            Dictionary with build statistics
        """
        start_time = time.time()
        
        try:
            # 1. List all documents in knowledge base
            document_files = await self.knowledge_base_port.list_documents()
            
            if not document_files:
                self.logger.warning("No documents found in knowledge base")
                return {
                    'success': False,
                    'error': 'No documents found',
                    'chunks_created': 0,
                    'build_time_ms': 0
                }
            
            # 2. Calculate corpus hash
            corpus_hash = calculate_corpus_hash(document_files)
            config_hash = calculate_config_hash(self.config)
            
            # 3. Check if rebuild is needed
            metadata_file = os.path.join(self.config.index_dir, 'index_metadata.json')
            existing_metadata = IndexMetadata.load_from_file(metadata_file)
            
            if not force_rebuild and existing_metadata:
                if (existing_metadata.corpus_hash == corpus_hash and 
                    existing_metadata.config_hash == config_hash):
                    self.logger.info("Index is up-to-date, skipping rebuild")
                    return {
                        'success': True,
                        'chunks_created': existing_metadata.total_chunks,
                        'documents_processed': existing_metadata.total_documents,
                        'build_time_ms': 0,
                        'rebuild_needed': False
                    }
            
            self.logger.info(f"Building RAG index from {len(document_files)} documents")
            
            # 4. Load documents
            documents = await self.knowledge_base_port.load_documents(document_files)
            
            # 5. Chunk documents
            chunks = await self._chunk_documents(documents, document_files)
            
            # 6. Generate embeddings
            chunks_with_embeddings = await self._generate_embeddings(chunks)
            
            # 7. Store in vector store
            await self.vector_store_port.upsert(chunks_with_embeddings)
            
            # 8. Persist vector store
            if self.config.persist_enabled:
                await self.vector_store_port.persist()
            
            # 9. Save metadata
            # Get embedding signature and store info
            embedding_signature = self.embeddings_port.get_signature().to_dict()
            
            # Determine store type and collection info
            store_type = "chroma" if "chroma" in self.vector_store_port.__class__.__name__.lower() else "sqlite"
            collection_name = getattr(self.vector_store_port, 'collection_name', None)
            index_path = getattr(self.vector_store_port, 'persist_dir', None)
            
            metadata = IndexMetadata(
                corpus_hash=corpus_hash,
                total_chunks=len(chunks_with_embeddings),
                total_documents=len(document_files),
                created_at=existing_metadata.created_at if existing_metadata else datetime.now(),
                updated_at=datetime.now(),
                version="1.0",
                config_hash=config_hash,
                embedding_signature=embedding_signature,
                adapter_name=self.embeddings_port.__class__.__name__,
                store_type=store_type,
                collection_name=collection_name,
                index_path=index_path
            )
            
            metadata.save_to_file(metadata_file)
            
            # Invalidate metadata cache to ensure fresh data is used
            RAGFacade._invalidate_metadata_cache()
            
            build_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"RAG index built successfully: {len(chunks_with_embeddings)} chunks in {build_time:.1f}ms")
            
            return {
                'success': True,
                'chunks_created': len(chunks_with_embeddings),
                'documents_processed': len(document_files),
                'build_time_ms': build_time,
                'rebuild_needed': True,
                'corpus_hash': corpus_hash,
                'config_hash': config_hash
            }
            
        except Exception as e:
            build_time = (time.time() - start_time) * 1000
            self.logger.error(f"Failed to build RAG index: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_created': 0,
                'build_time_ms': build_time
            }
    
    async def _chunk_documents(self, documents: List[str], file_paths: List[str]) -> List[DocumentChunk]:
        """Chunk documents with metadata"""
        chunks = []
        
        for doc_idx, (document, file_path) in enumerate(zip(documents, file_paths)):
            try:
                # Extract metadata from file
                metadata = await self.knowledge_base_port.get_document_metadata(file_path)
                
                # Determine domain and agent type from file path or metadata
                domain = self._extract_domain(file_path, metadata)
                agent_type = domain  # Map domain to agent_type
                
                # Split document into sections
                sections = self._split_into_sections(document)
                
                for section_idx, (section_title, section_content) in enumerate(sections):
                    if not section_content.strip():
                        continue
                    
                    # Chunk the section
                    section_chunks = self._chunk_text(
                        section_content,
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap
                    )
                    
                    for chunk_idx, chunk_content in enumerate(section_chunks):
                        # Create chunk with metadata
                        chunk = DocumentChunk(
                            chunk_id=f"chunk_{doc_idx}_{section_idx}_{chunk_idx}",
                            content=chunk_content.strip(),
                            source_file=os.path.basename(file_path),
                            section_title=section_title,
                            domain=domain,
                            agent_type=agent_type,
                            chunk_index=chunk_idx,
                            total_chunks=len(section_chunks),
                            breadcrumb=self._generate_breadcrumb(metadata, section_title)
                        )
                        
                        if validate_chunk(chunk):
                            chunks.append(chunk)
                        else:
                            self.logger.warning(f"Invalid chunk skipped: {chunk.chunk_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to chunk document {file_path}: {e}")
                continue
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for chunks"""
        if not chunks:
            return chunks
        
        try:
            # Extract text content
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings in batch
            embeddings = await self.embeddings_port.embed(texts)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            self.logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _extract_domain(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Extract domain from file path or metadata"""
        file_name = os.path.basename(file_path).lower()
        
        # Check file name for domain hints
        if 'open_finance' in file_name or 'openfinance' in file_name:
            return 'open_finance'
        elif 'golpe' in file_name or 'med' in file_name:
            return 'golpe_med'
        elif 'criacao' in file_name or 'conta' in file_name:
            return 'criacao_conta'
        else:
            # Default to atendimento_geral
            return 'atendimento_geral'
    
    def _split_into_sections(self, document: str) -> List[Tuple[str, str]]:
        """Split document into sections based on headers"""
        sections = []
        
        # Pattern for markdown headers (# ## ###)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = document.split('\n')
        current_section = ""
        current_title = "Introdução"
        
        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                # Save previous section
                if current_section.strip():
                    sections.append((current_title, current_section.strip()))
                
                # Start new section
                current_title = match.group(2).strip()
                current_section = ""
            else:
                current_section += line + '\n'
        
        # Save last section
        if current_section.strip():
            sections.append((current_title, current_section.strip()))
        
        return sections
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk text with overlap"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Don't create tiny chunks at the end
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence or paragraph boundary
            chunk_text = text[start:end]
            
            # Look for sentence endings near the chunk boundary
            sentence_endings = ['.', '!', '?', '\n']
            best_break = -1
            
            for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 100), -1):
                if chunk_text[i] in sentence_endings:
                    best_break = i + 1
                    break
            
            if best_break > 0:
                chunks.append(text[start:start + best_break])
                start = start + best_break - chunk_overlap
            else:
                chunks.append(chunk_text)
                start = end - chunk_overlap
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _generate_breadcrumb(self, metadata: Dict[str, Any], section_title: str) -> str:
        """Generate breadcrumb for chunk"""
        breadcrumb_parts = ["Jota"]
        
        # Add domain if available
        if 'domain' in metadata:
            breadcrumb_parts.append(metadata['domain'].replace('_', ' ').title())
        
        # Add section title
        if section_title and section_title != "Introdução":
            breadcrumb_parts.append(section_title)
        
        return " > ".join(breadcrumb_parts)
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            # Get vector store stats
            vector_stats = await self.vector_store_port.get_stats()
            
            # Get metadata
            metadata_file = os.path.join(self.config.index_dir, 'index_metadata.json')
            metadata = IndexMetadata.load_from_file(metadata_file)
            
            stats = {
                'vector_store_stats': vector_stats,
                'index_metadata': metadata.to_dict() if metadata else None,
                'config': {
                    'chunk_size': self.config.chunk_size,
                    'chunk_overlap': self.config.chunk_overlap,
                    'default_top_k': self.config.default_top_k
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {'error': str(e)}
    
    async def load_index(self) -> bool:
        """Load existing index from disk"""
        try:
            if not self.config.persist_enabled:
                self.logger.info("Persistence disabled, skipping index load")
                return True
            
            # Load vector store
            await self.vector_store_port.load()
            
            # Check metadata
            metadata_file = os.path.join(self.config.index_dir, 'index_metadata.json')
            metadata = IndexMetadata.load_from_file(metadata_file)
            
            if metadata:
                self.logger.info(f"Loaded index: {metadata.total_chunks} chunks from {metadata.total_documents} documents")
                return True
            else:
                self.logger.warning("No index metadata found, need to build index")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
