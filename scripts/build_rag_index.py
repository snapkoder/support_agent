#!/usr/bin/env python3
"""
Build RAG Index Script
Builds and maintains the RAG index from knowledge base documents
"""

import os
import sys
import asyncio
import logging
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from support_agent.rag.ports import EmbeddingsPort, VectorStorePort, KnowledgeBasePort
from support_agent.rag.adapters import (
    OpenAIEmbeddingsAdapter, LocalEmbeddingsAdapter,
    ChromaVectorStoreAdapter, SQLiteVectorStoreAdapter,
    KnowledgeBaseAdapter
)
from support_agent.rag.indexer import RAGIndexer
from support_agent.rag.models import RAGConfig
from support_agent.rag.rag_facade import RAGFacade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_adapters(config: RAGConfig):
    """Create adapters based on configuration and availability"""
    
    # 1. Knowledge Base Adapter (always local)
    kb_adapter = KnowledgeBaseAdapter(
        kb_dir=os.getenv('RAG_KB_DIR', './assets/knowledge_base')
    )
    
    # 2. Embeddings Adapter - Use RAG Facade for single source of truth
    embeddings_adapter = RAGFacade.build_embeddings_adapter_from_config()
    
    # 3. Vector Store Adapter (ChromaDB preferred, SQLite fallback)
    vector_store_adapter = None
    try:
        vector_store_adapter = ChromaVectorStoreAdapter(
            persist_dir=os.path.join(config.index_dir, 'chroma')
        )
        logger.info("Using ChromaDB vector store adapter")
    except Exception as e:
        logger.warning(f"Failed to create ChromaDB adapter: {e}")
    
    if vector_store_adapter is None:
        vector_store_adapter = SQLiteVectorStoreAdapter(
            db_path=os.path.join(config.index_dir, 'vectors.sqlite')
        )
        logger.info("Using SQLite vector store adapter")
    
    return kb_adapter, embeddings_adapter, vector_store_adapter

async def main():
    """Main build function"""
    logger.info("Starting RAG index build...")
    
    try:
        # Load configuration
        config = RAGConfig.from_env()
        
        # Create adapters
        kb_adapter, embeddings_adapter, vector_store_adapter = await create_adapters(config)
        
        # Create indexer
        indexer = RAGIndexer(
            knowledge_base_port=kb_adapter,
            vector_store_port=vector_store_adapter,
            embeddings_port=embeddings_adapter,
            config=config
        )
        
        # Health check
        logger.info("Performing health checks...")
        health_status = {
            'knowledge_base': await kb_adapter.health_check(),
            'embeddings': await embeddings_adapter.health_check(),
            'vector_store': await vector_store_adapter.health_check()
        }
        
        logger.info(f"Health status: {health_status}")
        
        if not all(health_status.values()):
            logger.error("Some components failed health check")
            sys.exit(1)
        
        # Build index
        logger.info("Building RAG index...")
        start_time = time.time()
        
        result = await indexer.build_index(force_rebuild='--force' in sys.argv)
        
        build_time = time.time() - start_time
        
        if result['success']:
            logger.info(f"✅ RAG index built successfully!")
            logger.info(f"📊 Statistics:")
            logger.info(f"   - Chunks created: {result['chunks_created']}")
            logger.info(f"   - Documents processed: {result['documents_processed']}")
            logger.info(f"   - Build time: {build_time:.2f}s")
            logger.info(f"   - Rebuild needed: {result.get('rebuild_needed', False)}")
            
            # Get index stats
            stats = await indexer.get_index_stats()
            logger.info(f"📈 Index stats: {json.dumps(stats, indent=2)}")
            
        else:
            logger.error(f"❌ Failed to build RAG index: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Save build report
        report = {
            'build_timestamp': time.time(),
            'build_time_seconds': build_time,
            'result': result,
            'health_status': health_status,
            'config': {
                'chunk_size': config.chunk_size,
                'chunk_overlap': config.chunk_overlap,
                'index_dir': config.index_dir
            }
        }
        
        report_file = os.path.join(config.index_dir, 'build_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Build report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
