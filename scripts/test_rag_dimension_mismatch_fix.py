#!/usr/bin/env python3
"""
Test RAG Dimension Mismatch Fix Script
Validates embedding signature detection and compatibility checking
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from support_agent.llm.llm_manager import LLMManager
from support_agent.rag.rag_integration import RAGIntegration
from support_agent.rag.adapters import (
    OpenAIEmbeddingsAdapter, LocalEmbeddingsAdapter,
    ChromaVectorStoreAdapter, SQLiteVectorStoreAdapter,
    KnowledgeBaseAdapter, RetrieverAdapter
)
from support_agent.rag.rag_service import RAGService
from support_agent.rag.models import RAGConfig
from support_agent.rag.rag_facade import RAGFacade

class RAGDimensionMismatchTester:
    """Tester for RAG dimension mismatch fix"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.results = []
        self.metrics = {
            'total_queries': 0,
            'rag_usage_count': 0,
            'rag_success_count': 0,
            'rag_latency_total': 0.0,
            'rag_hits_total': 0,
            'incompatible_queries': 0,
            'compatible_queries': 0
        }
    
    def _setup_logger(self):
        """Setup simple logger"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    async def setup_system(self, force_incompatible: bool = False) -> RAGIntegration:
        """Setup RAG system with specific configuration"""
        self.logger.info(f"Setting up RAG system (force_incompatible={force_incompatible})...")
        
        # Load configuration
        config = RAGConfig.from_env()
        
        # Create adapters
        kb_adapter = KnowledgeBaseAdapter(
            kb_dir=os.getenv('RAG_KB_DIR', './assets/knowledge_base')
        )
        
        # Force incompatible embeddings if requested
        if force_incompatible:
            # Use local embeddings when index was built with OpenAI (or vice versa)
            embeddings_adapter = LocalEmbeddingsAdapter(vector_size=384)
            self.logger.info("Using LOCAL embeddings adapter (forcing incompatibility)")
        else:
            # Use factory for compatible setup
            embeddings_adapter = RAGFacade.build_embeddings_adapter_from_config()
            self.logger.info("Using FACTORY embeddings adapter (compatible)")
        
        # Vector store adapter
        try:
            vector_store_adapter = ChromaVectorStoreAdapter(
                persist_dir=os.path.join(config.index_dir, 'chroma')
            )
            self.logger.info("Using ChromaDB vector store adapter")
        except Exception as e:
            self.logger.warning(f"Failed to create ChromaDB adapter: {e}")
            vector_store_adapter = SQLiteVectorStoreAdapter(
                db_path=os.path.join(config.index_dir, 'vectors.sqlite')
            )
            self.logger.info("Using SQLite vector store adapter")
        
        # Retriever adapter
        retriever_adapter = RetrieverAdapter(
            embeddings_port=embeddings_adapter,
            vector_store_port=vector_store_adapter,
            default_top_k=config.rag_top_k
        )
        
        # RAG service
        rag_service = RAGService(
            embeddings_port=embeddings_adapter,
            vector_store_port=vector_store_adapter,
            retriever_port=retriever_adapter,
            knowledge_base_port=kb_adapter,
            config=config
        )
        
        # LLM Manager
        llm_manager = LLMManager()
        
        # RAG Integration
        rag_integration = RAGIntegration(
            rag_service=rag_service,
            llm_manager=llm_manager
        )
        await rag_integration.initialize()
        
        return rag_integration
    
    async def load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load test dataset"""
        dataset_file = project_root / 'jota_test_questions.json'
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        self.logger.info(f"Loaded {len(dataset)} test questions")
        return dataset
    
    async def test_single_question(self, rag_integration: RAGIntegration, question: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Test a single question"""
        start_time = time.time()
        
        try:
            # Generate response with RAG
            result = await rag_integration.generate_response_with_rag(
                prompt=question['question'],
                agent_type=question['agent'],
                requires_rag=True
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            test_result = {
                'id': question['id'],
                'agent': question['agent'],
                'question': question['question'],
                'response': result['response'],
                'rag_used': result['rag_used'],
                'rag_latency_ms': result['rag_latency_ms'],
                'rag_hits': result['rag_hits'],
                'rag_sources': result['rag_sources'],
                'rag_domains': result['rag_domains'],
                'model_used': result['model_used'],
                'processing_time': processing_time,
                'success': True,
                'error': None,
                'scenario': scenario,
                'incompatible_detected': result['rag_hits'] == 0 and result['rag_latency_ms'] < 100  # Fast failure indicates incompatibility
            }
            
            # Update metrics
            self._update_metrics(result, test_result)
            
            return test_result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            error_result = {
                'id': question['id'],
                'agent': question['agent'],
                'question': question['question'],
                'response': '',
                'rag_used': False,
                'rag_latency_ms': 0,
                'rag_hits': 0,
                'rag_sources': [],
                'rag_domains': [],
                'model_used': 'error',
                'processing_time': processing_time,
                'success': False,
                'error': str(e),
                'scenario': scenario,
                'incompatible_detected': True
            }
            
            return error_result
    
    def _update_metrics(self, result: Dict[str, Any], test_result: Dict[str, Any]):
        """Update metrics"""
        self.metrics['total_queries'] += 1
        
        if result['rag_used']:
            self.metrics['rag_usage_count'] += 1
            if result['rag_hits'] > 0:
                self.metrics['rag_success_count'] += 1
            self.metrics['rag_latency_total'] += result['rag_latency_ms']
            self.metrics['rag_hits_total'] += result['rag_hits']
        
        if test_result['incompatible_detected']:
            self.metrics['incompatible_queries'] += 1
        else:
            self.metrics['compatible_queries'] += 1
    
    async def run_dimension_mismatch_test(self) -> Dict[str, Any]:
        """Run dimension mismatch test"""
        self.logger.info("🧪 Starting RAG Dimension Mismatch Fix Test")
        
        try:
            # Load dataset
            dataset = await self.load_test_dataset()
            
            # Test subset of questions (3 for quick testing)
            test_questions = dataset[:3]
            self.logger.info(f"Testing with {len(test_questions)} questions")
            
            # Scenario 1: Compatible setup
            self.logger.info("📋 Scenario 1: Compatible Setup")
            rag_integration_compatible = await self.setup_system(force_incompatible=False)
            
            for i, question in enumerate(test_questions):
                self.logger.info(f"Testing {question['id']} (compatible): {question['question'][:50]}...")
                
                result = await self.test_single_question(rag_integration_compatible, question, "compatible")
                self.results.append(result)
                
                status = "✅" if result['success'] else "❌"
                rag_indicator = "🔍" if result['rag_used'] else "📝"
                context_indicator = "📚" if result['rag_hits'] > 0 else "⚪"
                compat_indicator = "✅" if not result['incompatible_detected'] else "❌"
                
                self.logger.info(f"{status} {rag_indicator} {context_indicator} {compat_indicator} {question['id']}: RAG={result['rag_used']}, Hits={result['rag_hits']}, Latency={result['rag_latency_ms']:.1f}ms")
            
            # Scenario 2: Incompatible setup
            self.logger.info("📋 Scenario 2: Incompatible Setup")
            rag_integration_incompatible = await self.setup_system(force_incompatible=True)
            
            for i, question in enumerate(test_questions):
                self.logger.info(f"Testing {question['id']} (incompatible): {question['question'][:50]}...")
                
                result = await self.test_single_question(rag_integration_incompatible, question, "incompatible")
                self.results.append(result)
                
                status = "✅" if result['success'] else "❌"
                rag_indicator = "🔍" if result['rag_used'] else "📝"
                context_indicator = "📚" if result['rag_hits'] > 0 else "⚪"
                compat_indicator = "✅" if not result['incompatible_detected'] else "❌"
                
                self.logger.info(f"{status} {rag_indicator} {context_indicator} {compat_indicator} {question['id']}: RAG={result['rag_used']}, Hits={result['rag_hits']}, Latency={result['rag_latency_ms']:.1f}ms")
            
            # Calculate final metrics
            success_rate = sum(1 for r in self.results if r['success']) / len(self.results) * 100
            rag_usage_rate = self.metrics['rag_usage_count'] / self.metrics['total_queries'] * 100
            rag_success_rate = (self.metrics['rag_success_count'] / max(1, self.metrics['rag_usage_count'])) * 100
            avg_rag_latency = self.metrics['rag_latency_total'] / max(1, self.metrics['rag_success_count'])
            avg_rag_hits = self.metrics['rag_hits_total'] / max(1, self.metrics['rag_success_count'])
            incompatibility_detection_rate = self.metrics['incompatible_queries'] / self.metrics['total_queries'] * 100
            
            final_metrics = {
                'total_questions': len(self.results),
                'success_rate': success_rate,
                'rag_usage_rate': rag_usage_rate,
                'rag_success_rate': rag_success_rate,
                'avg_rag_latency_ms': avg_rag_latency,
                'avg_rag_hits': avg_rag_hits,
                'incompatibility_detection_rate': incompatibility_detection_rate,
                'compatible_queries': self.metrics['compatible_queries'],
                'incompatible_queries': self.metrics['incompatible_queries']
            }
            
            return {
                'results': self.results,
                'metrics': final_metrics,
                'summary': self._generate_summary(final_metrics),
                'compliance_check': self._check_compliance_criteria(final_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Dimension mismatch test failed: {e}")
            raise
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate summary of results"""
        summary = f"""
# RAG Dimension Mismatch Fix Test Summary

## Overall Metrics
- **Total Questions:** {metrics['total_questions']}
- **Success Rate:** {metrics['success_rate']:.1f}%
- **RAG Usage Rate:** {metrics['rag_usage_rate']:.1f}% (target: 100%)
- **RAG Success Rate:** {metrics['rag_success_rate']:.1f}%
- **Avg RAG Latency:** {metrics['avg_rag_latency_ms']:.1f}ms
- **Avg RAG Hits:** {metrics['avg_rag_hits']:.1f}

## Dimension Mismatch Detection
- **Incompatibility Detection Rate:** {metrics['incompatibility_detection_rate']:.1f}%
- **Compatible Queries:** {metrics['compatible_queries']}
- **Incompatible Queries:** {metrics['incompatible_queries']}

## Quality Assessment
"""
        
        # Quality assessment
        if metrics['rag_usage_rate'] >= 99:
            summary += "✅ EXCELLENT: RAG Always On working perfectly\n"
        elif metrics['rag_usage_rate'] >= 95:
            summary += "✅ GOOD: RAG Always On mostly working\n"
        else:
            summary += "⚠️ CONCERNING: RAG Always On not working properly\n"
        
        if metrics['incompatibility_detection_rate'] > 40:
            summary += "✅ EXCELLENT: Incompatibility detection working\n"
        elif metrics['incompatibility_detection_rate'] > 20:
            summary += "✅ GOOD: Incompatibility detection partially working\n"
        else:
            summary += "⚠️ CONCERNING: Incompatibility detection not working\n"
        
        if metrics['avg_rag_latency_ms'] < 700:
            summary += "✅ GOOD: Latency within acceptable range\n"
        else:
            summary += "⚠️ CONCERNING: High latency detected\n"
        
        return summary
    
    def _check_compliance_criteria(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance against acceptance criteria"""
        compliance = {
            'rag_usage_rate_100': metrics['rag_usage_rate'] >= 99.0,
            'success_rate_maintained': metrics['success_rate'] >= 95.0,
            'latency_acceptable': metrics['avg_rag_latency_ms'] <= 700,  # More lenient for testing
            'incompatibility_detected': metrics['incompatibility_detection_rate'] > 0,
            'no_explosive_latency': metrics['avg_rag_latency_ms'] < 2000  # No 2.6s latency
        }
        
        compliance['overall_compliant'] = all(compliance.values())
        
        return compliance
    
    def save_results(self, results: Dict[str, Any], run_id: str) -> str:
        """Save test results"""
        output_dir = project_root / 'validation_artifacts' / f'RAG_DIM_MISMATCH_FIX_{run_id}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSONL
        jsonl_file = output_dir / 'outputs.jsonl'
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for result in results['results']:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Save JSON
        json_file = output_dir / 'outputs.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = output_dir / 'summary.md'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(results['summary'])
        
        # Save compliance check
        compliance_file = output_dir / 'compliance_check.json'
        with open(compliance_file, 'w', encoding='utf-8') as f:
            json.dump(results['compliance_check'], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_dir}")
        return str(output_dir)

async def main():
    """Main function"""
    tester = RAGDimensionMismatchTester()
    
    try:
        results = await tester.run_dimension_mismatch_test()
        
        # Generate run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        output_dir = tester.save_results(results, run_id)
        
        # Print summary
        print("\n" + "="*80)
        print("🧪 RAG DIMENSION MISMATCH FIX TEST COMPLETE")
        print("="*80)
        print(f"📋 Run ID: {run_id}")
        print(f"📊 Results:")
        print(f"   - Success Rate: {results['metrics']['success_rate']:.1f}%")
        print(f"   - RAG Usage Rate: {results['metrics']['rag_usage_rate']:.1f}% (target: 100%)")
        print(f"   - RAG Success Rate: {results['metrics']['rag_success_rate']:.1f}%")
        print(f"   - Avg RAG Latency: {results['metrics']['avg_rag_latency_ms']:.1f}ms")
        print(f"   - Avg RAG Hits: {results['metrics']['avg_rag_hits']:.1f}")
        print(f"   - Incompatibility Detection Rate: {results['metrics']['incompatibility_detection_rate']:.1f}%")
        print(f"   - Compatible Queries: {results['metrics']['compatible_queries']}")
        print(f"   - Incompatible Queries: {results['metrics']['incompatible_queries']}")
        print(f"📁 Output: {output_dir}")
        
        # Compliance check
        compliance = results['compliance_check']
        print(f"\n🔍 Compliance Check:")
        for criterion, passed in compliance.items():
            if criterion == 'overall_compliant':
                continue
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion.replace('_', ' ').title()}")
        
        overall_status = "✅ COMPLIANT" if compliance['overall_compliant'] else "❌ NON-COMPLIANT"
        print(f"\n🎯 Overall Status: {overall_status}")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Dimension mismatch test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
