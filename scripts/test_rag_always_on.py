#!/usr/bin/env python3
"""
Test RAG Always On Script
Valida que RAG está sempre ativo e coleta métricas
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

class RAGAlwaysOnTester:
    """Tester for RAG Always On functionality"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.results = []
        self.metrics = {
            'total_queries': 0,
            'rag_usage_count': 0,
            'rag_success_count': 0,
            'rag_latency_total': 0.0,
            'rag_hits_total': 0,
            'empty_context_count': 0,
            'filtered_queries_count': 0
        }
    
    def _setup_logger(self):
        """Setup simple logger"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    async def setup_system(self) -> RAGIntegration:
        """Setup RAG system with Always On enabled"""
        self.logger.info("Setting up RAG Always On system...")
        
        # Load configuration
        config = RAGConfig.from_env()
        
        # Ensure RAG Always On is enabled
        config.rag_always_on = True
        config.rag_top_k = 6  # Controlled top_k
        
        # Create adapters
        kb_adapter = KnowledgeBaseAdapter(
            kb_dir=os.getenv('RAG_KB_DIR', './assets/knowledge_base')
        )
        
        # Embeddings adapter - Use OpenAI to match ChromaDB dimensions
        if os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('YOUR_'):
            try:
                embeddings_adapter = OpenAIEmbeddingsAdapter()
                self.logger.info("Using OpenAI embeddings adapter for RAG Always On")
            except Exception as e:
                self.logger.warning(f"Failed to create OpenAI embeddings adapter: {e}")
                embeddings_adapter = LocalEmbeddingsAdapter(vector_size=384)
        else:
            embeddings_adapter = LocalEmbeddingsAdapter(vector_size=384)
            self.logger.info("Using local TF-IDF embeddings adapter for RAG Always On")
        
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
    
    async def test_single_question(self, rag_integration: RAGIntegration, question: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single question with RAG Always On"""
        start_time = time.time()
        
        try:
            # Generate response with RAG Always On
            result = await rag_integration.generate_response_with_rag(
                prompt=question['question'],
                agent_type=question['agent'],
                requires_rag=True  # Always true with RAG Always On
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Analyze RAG Always On compliance
            compliance_analysis = self._analyze_compliance(
                question=question,
                response=result['response'],
                rag_result=result
            )
            
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
                'compliance_analysis': compliance_analysis
            }
            
            # Update metrics
            self._update_metrics(result, compliance_analysis)
            
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
                'compliance_analysis': {}
            }
            
            return error_result
    
    def _analyze_compliance(self, question: Dict[str, Any], response: str, rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze RAG Always On compliance"""
        analysis = {
            'rag_always_on_compliant': rag_result['rag_used'],
            'has_context': rag_result['rag_hits'] > 0,
            'has_sources': len(rag_result['rag_sources']) > 0,
            'has_domains': len(rag_result['rag_domains']) > 0,
            'response_length': len(response),
            'uses_kb_facts': self._uses_kb_facts(response),
            'agent_type_filter_applied': rag_result['rag_domains'] and question['agent'] in rag_result['rag_domains']
        }
        
        return analysis
    
    def _uses_kb_facts(self, response: str) -> bool:
        """Check if response uses KB facts"""
        kb_facts = [
            "(11) 4004-8006",
            "WhatsApp",
            "gratuito",
            "Pix",
            "Open Finance",
            "MED",
            "7 dias"
        ]
        
        return any(fact in response for fact in kb_facts)
    
    def _update_metrics(self, result: Dict[str, Any], compliance_analysis: Dict[str, Any]):
        """Update RAG Always On metrics"""
        self.metrics['total_queries'] += 1
        
        if result['rag_used']:
            self.metrics['rag_usage_count'] += 1
            if result['rag_hits'] > 0:
                self.metrics['rag_success_count'] += 1
            self.metrics['rag_latency_total'] += result['rag_latency_ms']
            self.metrics['rag_hits_total'] += result['rag_hits']
        
        if not compliance_analysis['has_context']:
            self.metrics['empty_context_count'] += 1
        
        if compliance_analysis['agent_type_filter_applied']:
            self.metrics['filtered_queries_count'] += 1
    
    async def run_compliance_test(self) -> Dict[str, Any]:
        """Run RAG Always On compliance test"""
        self.logger.info("🧪 Starting RAG Always On Compliance Test")
        
        try:
            # Setup system
            rag_integration = await self.setup_system()
            
            # Load dataset
            dataset = await self.load_test_dataset()
            
            # Test subset of questions (5 for quick testing)
            test_questions = dataset[:5]
            self.logger.info(f"Testing with {len(test_questions)} questions (subset)")
            
            for i, question in enumerate(test_questions):
                self.logger.info(f"Testing {question['id']}: {question['question'][:50]}...")
                
                result = await self.test_single_question(rag_integration, question)
                self.results.append(result)
                
                status = "✅" if result['success'] else "❌"
                rag_indicator = "🔍" if result['rag_used'] else "📝"
                context_indicator = "📚" if result['rag_hits'] > 0 else "⚪"
                
                self.logger.info(f"{status} {rag_indicator} {context_indicator} {question['id']}: RAG={result['rag_used']}, Hits={result['rag_hits']}, Latency={result['rag_latency_ms']:.1f}ms")
            
            # Calculate final metrics
            success_rate = sum(1 for r in self.results if r['success']) / len(self.results) * 100
            rag_usage_rate = self.metrics['rag_usage_count'] / self.metrics['total_queries'] * 100
            rag_success_rate = (self.metrics['rag_success_count'] / max(1, self.metrics['rag_usage_count'])) * 100
            avg_rag_latency = self.metrics['rag_latency_total'] / max(1, self.metrics['rag_success_count'])
            avg_rag_hits = self.metrics['rag_hits_total'] / max(1, self.metrics['rag_success_count'])
            empty_context_rate = self.metrics['empty_context_count'] / self.metrics['total_queries'] * 100
            filter_applied_rate = self.metrics['filtered_queries_count'] / self.metrics['total_queries'] * 100
            
            final_metrics = {
                'total_questions': self.metrics['total_queries'],
                'success_rate': success_rate,
                'rag_usage_rate': rag_usage_rate,
                'rag_success_rate': rag_success_rate,
                'avg_rag_latency_ms': avg_rag_latency,
                'avg_rag_hits': avg_rag_hits,
                'empty_context_rate': empty_context_rate,
                'filter_applied_rate': filter_applied_rate
            }
            
            return {
                'results': self.results,
                'metrics': final_metrics,
                'summary': self._generate_summary(final_metrics),
                'compliance_check': self._check_compliance_criteria(final_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"RAG Always On compliance test failed: {e}")
            raise
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate summary of results"""
        summary = f"""
# RAG Always On Compliance Test Summary

## Overall Metrics
- **Total Questions:** {metrics['total_questions']}
- **Success Rate:** {metrics['success_rate']:.1f}%
- **RAG Usage Rate:** {metrics['rag_usage_rate']:.1f}% (target: 100%)
- **RAG Success Rate:** {metrics['rag_success_rate']:.1f}%
- **Avg RAG Latency:** {metrics['avg_rag_latency_ms']:.1f}ms
- **Avg RAG Hits:** {metrics['avg_rag_hits']:.1f}

## RAG Always On Specific
- **Empty Context Rate:** {metrics['empty_context_rate']:.1f}% (lower is better)
- **Filter Applied Rate:** {metrics['filter_applied_rate']:.1f}% (higher is better)

## Quality Assessment
"""
        
        # Quality assessment
        if metrics['rag_usage_rate'] >= 99:
            summary += "✅ EXCELLENT: RAG Always On working perfectly\n"
        elif metrics['rag_usage_rate'] >= 95:
            summary += "✅ GOOD: RAG Always On mostly working\n"
        else:
            summary += "⚠️ CONCERNING: RAG Always On not working properly\n"
        
        if metrics['empty_context_rate'] < 10:
            summary += "✅ GOOD: Low empty context rate\n"
        else:
            summary += "⚠️ CONCERNING: High empty context rate\n"
        
        if metrics['filter_applied_rate'] > 50:
            summary += "✅ GOOD: Intelligent filtering working\n"
        else:
            summary += "⚠️ CONCERNING: Low filtering rate\n"
        
        return summary
    
    def _check_compliance_criteria(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance against acceptance criteria"""
        compliance = {
            'rag_usage_rate_100': metrics['rag_usage_rate'] >= 99.0,
            'success_rate_maintained': metrics['success_rate'] >= 95.0,
            'latency_acceptable': metrics['avg_rag_latency_ms'] <= 500,  # 25% increase from ~400ms
            'filtering_working': metrics['filter_applied_rate'] > 30.0,
            'empty_context_acceptable': metrics['empty_context_rate'] < 20.0
        }
        
        compliance['overall_compliant'] = all(compliance.values())
        
        return compliance
    
    def save_results(self, results: Dict[str, Any], run_id: str) -> str:
        """Save test results"""
        output_dir = project_root / 'validation_artifacts' / f'RAG_ALWAYS_ON_{run_id}'
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
        
        # Add breadcrumb examples
        self._add_breadcrumb_examples(output_dir, results['results'])
        
        self.logger.info(f"Results saved to: {output_dir}")
        return str(output_dir)
    
    def _add_breadcrumb_examples(self, output_dir: Path, results: List[Dict[str, Any]]):
        """Add breadcrumb examples"""
        examples_file = output_dir / 'breadcrumb_examples.md'
        
        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write("# RAG Always On - Breadcrumb Examples\n\n")
            
            # Get examples with breadcrumbs
            breadcrumb_examples = [r for r in results if r['rag_hits'] > 0][:10]
            
            for i, result in enumerate(breadcrumb_examples):
                f.write(f"## Example {i+1}: {result['id']}\n\n")
                f.write(f"**Agent:** {result['agent']}\n")
                f.write(f"**Question:** {result['question']}\n")
                f.write(f"**RAG Hits:** {result['rag_hits']}\n")
                f.write(f"**Sources:** {result['rag_sources']}\n")
                f.write(f"**Domains:** {result['rag_domains']}\n")
                f.write(f"**Response Preview:** {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}\n")
                f.write(f"**Compliance:** {'✅' if result['rag_used'] else '❌'} RAG Always On\n")
                f.write("\n---\n\n")

async def main():
    """Main function"""
    tester = RAGAlwaysOnTester()
    
    try:
        results = await tester.run_compliance_test()
        
        # Generate run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        output_dir = tester.save_results(results, run_id)
        
        # Print summary
        print("\n" + "="*80)
        print("🧪 RAG ALWAYS ON COMPLIANCE TEST COMPLETE")
        print("="*80)
        print(f"📋 Run ID: {run_id}")
        print(f"📊 Results:")
        print(f"   - Success Rate: {results['metrics']['success_rate']:.1f}%")
        print(f"   - RAG Usage Rate: {results['metrics']['rag_usage_rate']:.1f}% (target: 100%)")
        print(f"   - RAG Success Rate: {results['metrics']['rag_success_rate']:.1f}%")
        print(f"   - Avg RAG Latency: {results['metrics']['avg_rag_latency_ms']:.1f}ms")
        print(f"   - Avg RAG Hits: {results['metrics']['avg_rag_hits']:.1f}")
        print(f"   - Empty Context Rate: {results['metrics']['empty_context_rate']:.1f}%")
        print(f"   - Filter Applied Rate: {results['metrics']['filter_applied_rate']:.1f}%")
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
        print(f"\n❌ RAG Always On compliance test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
