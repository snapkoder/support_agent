"""
External Knowledge Base Stub Adapter
Simula integração com base de conhecimento externa sem depender de APIs reais
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..ports import ExternalKnowledgePort, DocumentChunk

logger = logging.getLogger(__name__)

class ExternalKBStubAdapter(ExternalKnowledgePort):
    """Stub adapter for external knowledge base integration"""
    
    def __init__(self, external_kb_path: Optional[str] = None):
        """Initialize external KB stub adapter
        
        Args:
            external_kb_path: Path to external KB file (optional)
        """
        self.external_kb_path = external_kb_path or os.getenv('EXTERNAL_KB_PATH', './assets/knowledge_base/external_stub.md')
        self.logger = logging.getLogger(f"{__name__}.ExternalKBStubAdapter")
        self._external_chunks: List[DocumentChunk] = []
        self._loaded = False
        
        # Load external KB if enabled
        if os.getenv('EXTERNAL_KB_ENABLED', 'false').lower() == 'true':
            self._load_external_kb()
        else:
            self.logger.info("External KB disabled - stub adapter will return empty results")
    
    def _load_external_kb(self) -> None:
        """Load external knowledge base from file"""
        try:
            if not os.path.exists(self.external_kb_path):
                self.logger.warning(f"External KB file not found: {self.external_kb_path}")
                return
            
            self.logger.info(f"Loading external KB from: {self.external_kb_path}")
            
            with open(self.external_kb_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create chunks from external KB
            self._external_chunks = self._create_chunks_from_content(content)
            self._loaded = True
            
            self.logger.info(f"Loaded {len(self._external_chunks)} external KB chunks")
            
        except Exception as e:
            self.logger.error(f"Failed to load external KB: {e}")
            self._external_chunks = []
            self._loaded = False
    
    def _create_chunks_from_content(self, content: str) -> List[DocumentChunk]:
        """Create DocumentChunk objects from content"""
        chunks = []
        
        # Split content by sections
        sections = content.split('\n## ')
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Extract section title and content
            lines = section.split('\n')
            title = lines[0].strip() if lines else f"Section {i+1}"
            section_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            if not section_content.strip():
                continue
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=f"external_{i+1}",
                content=section_content.strip(),
                source_file="external_stub.md",
                section_title=title,
                domain="external",
                agent_type="general",
                language="pt-BR",
                content_type="text",
                priority=2,  # Lower priority than local KB
                metadata={
                    "source": "external_stub",
                    "external": True,
                    "loaded_at": datetime.now().isoformat(),
                    "chunk_index": i
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def query_external_knowledge(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        Query external knowledge base for relevant chunks.
        
        Args:
            query: Search query
            filters: Metadata filters
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks from external source
        """
        try:
            # Load external KB if not loaded
            if not self._loaded:
                self._load_external_kb()
            
            # If still no chunks, return empty
            if not self._external_chunks:
                self.logger.info("No external KB chunks available")
                return []
            
            # Simple keyword-based matching (stub implementation)
            query_lower = query.lower()
            scored_chunks = []
            
            for chunk in self._external_chunks:
                # Simple scoring based on keyword overlap
                content_lower = chunk.content.lower()
                title_lower = chunk.section_title.lower()
                
                # Calculate simple score
                score = 0.0
                query_words = query_lower.split()
                
                # Check title matches (higher weight)
                for word in query_words:
                    if word in title_lower:
                        score += 2.0
                
                # Check content matches
                for word in query_words:
                    if word in content_lower:
                        score += 1.0
                
                # Add chunk if it has any score
                if score > 0:
                    scored_chunks.append((chunk, score))
            
            # Sort by score and return top_k
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            result_chunks = [chunk for chunk, score in scored_chunks[:top_k]]
            
            self.logger.info(f"External KB query returned {len(result_chunks)} chunks")
            return result_chunks
            
        except Exception as e:
            self.logger.error(f"External KB query failed: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if external knowledge source is healthy"""
        try:
            if not self._loaded:
                self._load_external_kb()
            
            # Consider healthy if we can load or if disabled
            is_enabled = os.getenv('EXTERNAL_KB_ENABLED', 'false').lower() == 'true'
            
            if not is_enabled:
                return True  # Disabled is considered healthy
            
            return self._loaded and len(self._external_chunks) > 0
            
        except Exception as e:
            self.logger.error(f"External KB health check failed: {e}")
            return False
