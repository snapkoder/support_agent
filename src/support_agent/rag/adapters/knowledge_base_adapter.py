"""
Knowledge Base Adapter - Loads documents from local knowledge base
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..ports import KnowledgeBasePort, KnowledgeBaseError

logger = logging.getLogger(__name__)

class KnowledgeBaseAdapter(KnowledgeBasePort):
    """
    Knowledge base adapter for loading local documents
    """
    
    def __init__(self, kb_dir: str = "./assets/knowledge_base"):
        """Initialize knowledge base adapter"""
        self.kb_dir = Path(kb_dir)
        self.logger = logging.getLogger(f"{__name__}.KnowledgeBaseAdapter")
        
        # Ensure KB directory exists
        if not self.kb_dir.exists():
            self.logger.warning(f"Knowledge base directory not found: {kb_dir}")
            self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Knowledge Base Adapter initialized at: {kb_dir}")
    
    async def list_documents(self) -> List[str]:
        """
        List all documents in knowledge base.
        
        Returns:
            List of document file paths
        """
        try:
            documents = []
            
            # Supported file extensions
            supported_extensions = ['.md', '.txt', '.json']
            
            # Recursively find documents
            for file_path in self.kb_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    documents.append(str(file_path))
            
            self.logger.info(f"Found {len(documents)} documents in knowledge base")
            return sorted(documents)
            
        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            raise KnowledgeBaseError(f"Failed to list documents: {e}")
    
    async def load_documents(self, file_paths: List[str]) -> List[str]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of document contents
        """
        if not file_paths:
            return []
        
        try:
            documents = []
            
            for file_path in file_paths:
                try:
                    path = Path(file_path)
                    
                    if not path.exists():
                        self.logger.warning(f"Document not found: {file_path}")
                        continue
                    
                    # Read file based on extension
                    if path.suffix.lower() == '.json':
                        content = self._load_json_file(path)
                    else:
                        content = self._load_text_file(path)
                    
                    if content:
                        documents.append(content)
                        self.logger.debug(f"Loaded document: {path.name}")
                    else:
                        self.logger.warning(f"Empty document: {path.name}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to load document {file_path}: {e}")
                    continue
            
            self.logger.info(f"Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load documents: {e}")
            raise KnowledgeBaseError(f"Failed to load documents: {e}")
    
    def _load_text_file(self, file_path: Path) -> Optional[str]:
        """Load text file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read text file {file_path}: {e}")
            return None
    
    def _load_json_file(self, file_path: Path) -> Optional[str]:
        """Load JSON file content and convert to text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text
            if isinstance(data, list):
                return '\n'.join(str(item) for item in data)
            elif isinstance(data, dict):
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                return str(data)
                
        except Exception as e:
            self.logger.error(f"Failed to read JSON file {file_path}: {e}")
            return None
    
    async def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Document metadata
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {}
            
            # Basic file metadata
            stat = path.stat()
            
            metadata = {
                'file_name': path.name,
                'file_path': str(path),
                'file_size': stat.st_size,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'extension': path.suffix.lower()
            }
            
            # Try to extract domain from file path
            domain = self._extract_domain_from_path(path)
            if domain:
                metadata['domain'] = domain
            
            # Try to extract title from content (for markdown files)
            if path.suffix.lower() == '.md':
                title = self._extract_title_from_markdown(path)
                if title:
                    metadata['title'] = title
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {file_path}: {e}")
            return {}
    
    def _extract_domain_from_path(self, file_path: Path) -> Optional[str]:
        """Extract domain from file path"""
        path_str = str(file_path).lower()
        
        if 'open_finance' in path_str or 'openfinance' in path_str:
            return 'open_finance'
        elif 'golpe' in path_str or 'med' in path_str:
            return 'golpe_med'
        elif 'criacao' in path_str or 'conta' in path_str:
            return 'criacao_conta'
        else:
            return 'atendimento_geral'
    
    def _extract_title_from_markdown(self, file_path: Path) -> Optional[str]:
        """Extract title from markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to find title
                for line in f.readlines()[:10]:
                    line = line.strip()
                    if line.startswith('# '):
                        return line[2:].strip()
                    elif line.startswith('## '):
                        return line[3:].strip()
        except Exception as e:
            self.logger.debug(f"Failed to extract title from {file_path}: {e}")
        
        return None
    
    async def health_check(self) -> bool:
        """Check if knowledge base is healthy"""
        try:
            # Check if directory exists and is accessible
            if not self.kb_dir.exists():
                return False
            
            # Try to list documents
            documents = await self.list_documents()
            return len(documents) >= 0  # At least no errors
            
        except Exception as e:
            self.logger.error(f"Knowledge base health check failed: {e}")
            return False
