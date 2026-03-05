#!/usr/bin/env python3
"""
Test Prompt Restructure Quality Script
Testa os prompts reestruturados com o dataset existente
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

class PromptQualityTester:
    """Tester for prompt restructure quality"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.results = []
        self.metrics = {
            'total_questions': 0,
            'rag_usage_count': 0,
            'rag_success_count': 0,
            'avg_response_length': 0,
            'template_like_count': 0,
            'context_usage_count': 0,
            'clarification_count': 0
        }
    
    def _setup_logger(self):
        """Setup simple logger"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    async def setup_system(self) -> RAGIntegration:
        """Setup RAG system with restructured KB"""
        self.logger.info("Setting up system with restructured KB...")
        
        # Load configuration
        config = RAGConfig.from_env()
        
        # Update KB path to restructured version
        kb_adapter = KnowledgeBaseAdapter(
            kb_dir=os.getenv('RAG_KB_DIR', './assets/knowledge_base')
        )
        
        # Embeddings adapter
        if os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('YOUR_'):
            try:
                embeddings_adapter = OpenAIEmbeddingsAdapter()
                self.logger.info("Using OpenAI embeddings adapter")
            except Exception as e:
                self.logger.warning(f"Failed to create OpenAI embeddings adapter: {e}")
                embeddings_adapter = LocalEmbeddingsAdapter(vector_size=384)
        else:
            embeddings_adapter = LocalEmbeddingsAdapter(vector_size=384)
            self.logger.info("Using local TF-IDF embeddings adapter")
        
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
            default_top_k=config.default_top_k
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
        """Test a single question"""
        start_time = time.time()
        
        try:
            # Generate response
            result = await rag_integration.generate_response_with_rag(
                prompt=question['question'],
                agent_type=question['agent'],
                requires_rag=question.get('requires_rag', False)
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Analyze response quality
            quality_analysis = self._analyze_response_quality(
                question=question,
                response=result['response'],
                rag_used=result['rag_used'],
                rag_sources=result['rag_sources']
            )
            
            test_result = {
                'id': question['id'],
                'agent': question['agent'],
                'question': question['question'],
                'expected_answer': question.get('expected_answer', ''),
                'response': result['response'],
                'rag_used': result['rag_used'],
                'rag_latency_ms': result['rag_latency_ms'],
                'rag_hits': result['rag_hits'],
                'rag_sources': result['rag_sources'],
                'model_used': result['model_used'],
                'processing_time': processing_time,
                'requires_rag': question.get('requires_rag', False),
                'success': True,
                'error': None,
                'quality_analysis': quality_analysis
            }
            
            # Update metrics
            self._update_metrics(result, quality_analysis)
            
            return test_result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            error_result = {
                'id': question['id'],
                'agent': question['agent'],
                'question': question['question'],
                'expected_answer': question.get('expected_answer', ''),
                'response': '',
                'rag_used': False,
                'rag_latency_ms': 0,
                'rag_hits': 0,
                'rag_sources': [],
                'model_used': 'error',
                'processing_time': processing_time,
                'requires_rag': question.get('requires_rag', False),
                'success': False,
                'error': str(e),
                'quality_analysis': {}
            }
            
            return error_result
    
    def _analyze_response_quality(self, question: Dict[str, Any], response: str, rag_used: bool, rag_sources: List[str]) -> Dict[str, Any]:
        """Analyze response quality"""
        analysis = {
            'response_length': len(response),
            'is_template_like': self._is_template_like(response),
            'uses_context': self._uses_context(response, rag_sources),
            'asks_clarification': self._asks_clarification(response),
            'mentions_jota_facts': self._mentions_jota_facts(response),
            'varies_from_expected': self._varies_from_expected(response, question.get('expected_answer', '')),
            'professional_tone': self._has_professional_tone(response)
        }
        
        return analysis
    
    def _is_template_like(self, response: str) -> bool:
        """Check if response is too template-like"""
        template_phrases = [
            "Olá! Esse é o canal oficial de atendimento do Jota.",
            "Para abrir sua conta digital, é só entrar em contato",
            "Posso ajudar com mais alguma coisa?"
        ]
        
        template_count = sum(1 for phrase in template_phrases if phrase.lower() in response.lower())
        return template_count >= 2  # If 2+ template phrases, it's too template-like
    
    def _uses_context(self, response: str, rag_sources: List[str]) -> bool:
        """Check if response uses RAG context"""
        if not rag_sources:
            return False
        
        # Check if response contains specific facts that would come from KB
        context_indicators = [
            "(11) 4004-8006",
            "100% do CDI",
            "Open Finance",
            "MED",
            "7 dias úteis"
        ]
        
        return any(indicator in response for indicator in context_indicators)
    
    def _asks_clarification(self, response: str) -> bool:
        """Check if response asks for clarification"""
        clarification_phrases = [
            "posso ajudar",
            "pode me dizer",
            "preciso de",
            "qual o",
            "poderia informar"
        ]
        
        return any(phrase.lower() in response.lower() for phrase in clarification_phrases)
    
    def _mentions_jota_facts(self, response: str) -> bool:
        """Check if response mentions specific Jota facts"""
        jota_facts = [
            "WhatsApp",
            "gratuito",
            "Pix",
            "boleto",
            "conta digital"
        ]
        
        return any(fact.lower() in response.lower() for fact in jota_facts)
    
    def _varies_from_expected(self, response: str, expected: str) -> bool:
        """Check if response varies from expected (good thing)"""
        if not expected:
            return True
        
        # Simple similarity check
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        
        # If less than 70% similarity, it's varied (good)
        if len(expected_words) == 0:
            return True
        
        similarity = len(response_words & expected_words) / len(expected_words)
        return similarity < 0.7
    
    def _has_professional_tone(self, response: str) -> bool:
        """Check if response has professional tone"""
        professional_indicators = [
            "poderia",
            "gostaria",
            "por favor",
            "agradeço",
            "atenciosamente"
        ]
        
        casual_indicators = [
            "oi",
            "blz",
            "td bem",
            "vlw"
        ]
        
        professional_count = sum(1 for ind in professional_indicators if ind in response.lower())
        casual_count = sum(1 for ind in casual_indicators if ind in response.lower())
        
        return professional_count > casual_count
    
    def _update_metrics(self, result: Dict[str, Any], quality_analysis: Dict[str, Any]):
        """Update quality metrics"""
        self.metrics['total_questions'] += 1
        
        if result['rag_used']:
            self.metrics['rag_usage_count'] += 1
            if result['rag_hits'] > 0:
                self.metrics['rag_success_count'] += 1
        
        # Update average response length
        current_avg = self.metrics['avg_response_length']
        new_length = quality_analysis['response_length']
        count = self.metrics['total_questions']
        self.metrics['avg_response_length'] = ((current_avg * (count - 1)) + new_length) / count
        
        # Update quality indicators
        if quality_analysis['is_template_like']:
            self.metrics['template_like_count'] += 1
        
        if quality_analysis['uses_context']:
            self.metrics['context_usage_count'] += 1
        
        if quality_analysis['asks_clarification']:
            self.metrics['clarification_count'] += 1
    
    async def run_quality_test(self) -> Dict[str, Any]:
        """Run complete quality test"""
        self.logger.info("🧪 Starting Prompt Restructure Quality Test")
        
        try:
            # Setup system
            rag_integration = await self.setup_system()
            
            # Load dataset
            dataset = await self.load_test_dataset()
            
            # Test all questions
            for i, question in enumerate(dataset):
                self.logger.info(f"Testing {question['id']}: {question['question'][:50]}...")
                
                result = await self.test_single_question(rag_integration, question)
                self.results.append(result)
                
                status = "✅" if result['success'] else "❌"
                rag_indicator = "🔍" if result['rag_used'] else "📝"
                
                self.logger.info(f"{status} {rag_indicator} {question['id']}: Length={len(result['response'])}, RAG={result['rag_used']}")
            
            # Calculate final metrics
            success_rate = sum(1 for r in self.results if r['success']) / len(self.results) * 100
            rag_usage_rate = self.metrics['rag_usage_count'] / self.metrics['total_questions'] * 100
            template_rate = self.metrics['template_like_count'] / self.metrics['total_questions'] * 100
            context_usage_rate = self.metrics['context_usage_count'] / self.metrics['total_questions'] * 100
            
            final_metrics = {
                'total_questions': self.metrics['total_questions'],
                'success_rate': success_rate,
                'rag_usage_rate': rag_usage_rate,
                'rag_success_rate': (self.metrics['rag_success_count'] / max(1, self.metrics['rag_usage_count'])) * 100,
                'avg_response_length': self.metrics['avg_response_length'],
                'template_like_rate': template_rate,
                'context_usage_rate': context_usage_rate,
                'clarification_rate': (self.metrics['clarification_count'] / self.metrics['total_questions']) * 100
            }
            
            return {
                'results': self.results,
                'metrics': final_metrics,
                'summary': self._generate_summary(final_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Quality test failed: {e}")
            raise
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate summary of results"""
        summary = f"""
# Prompt Restructure Quality Test Summary

## Overall Metrics
- **Total Questions:** {metrics['total_questions']}
- **Success Rate:** {metrics['success_rate']:.1f}%
- **RAG Usage Rate:** {metrics['rag_usage_rate']:.1f}%
- **RAG Success Rate:** {metrics['rag_success_rate']:.1f}%

## Response Quality
- **Avg Response Length:** {metrics['avg_response_length']:.1f} chars
- **Template-like Rate:** {metrics['template_like_rate']:.1f}% (lower is better)
- **Context Usage Rate:** {metrics['context_usage_rate']:.1f}% (higher is better)
- **Clarification Rate:** {metrics['clarification_rate']:.1f}%

## Quality Assessment
"""
        
        # Quality assessment
        if metrics['template_like_rate'] < 20:
            summary += "✅ Good: Low template-like responses\n"
        else:
            summary += "⚠️ Concerning: High template-like responses\n"
        
        if metrics['context_usage_rate'] > 30:
            summary += "✅ Good: High context usage\n"
        else:
            summary += "⚠️ Concerning: Low context usage\n"
        
        if metrics['success_rate'] > 80:
            summary += "✅ Good: High success rate\n"
        else:
            summary += "⚠️ Concerning: Low success rate\n"
        
        return summary
    
    def save_results(self, results: Dict[str, Any], run_id: str) -> str:
        """Save test results"""
        output_dir = project_root / 'validation_artifacts' / f'PR_PROMPT_RESTRUCTURE_{run_id}'
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
        
        # Add sample responses
        self._add_sample_responses(output_dir, results['results'])
        
        self.logger.info(f"Results saved to: {output_dir}")
        return str(output_dir)
    
    def _add_sample_responses(self, output_dir: Path, results: List[Dict[str, Any]]):
        """Add sample responses to analysis"""
        samples_file = output_dir / 'sample_responses.md'
        
        with open(samples_file, 'w', encoding='utf-8') as f:
            f.write("# Sample Responses Analysis\n\n")
            
            # Get 10 diverse samples
            samples = results[:10]
            
            for i, result in enumerate(samples):
                f.write(f"## Sample {i+1}: {result['id']}\n\n")
                f.write(f"**Agent:** {result['agent']}\n")
                f.write(f"**Question:** {result['question']}\n")
                f.write(f"**RAG Used:** {result['rag_used']}\n")
                f.write(f"**Response Length:** {len(result['response'])}\n")
                f.write(f"**Response:** {result['response']}\n")
                
                if result.get('quality_analysis'):
                    qa = result['quality_analysis']
                    f.write(f"**Quality Analysis:**\n")
                    f.write(f"- Template-like: {qa['is_template_like']}\n")
                    f.write(f"- Uses context: {qa['uses_context']}\n")
                    f.write(f"- Asks clarification: {qa['asks_clarification']}\n")
                    f.write(f"- Varies from expected: {qa['varies_from_expected']}\n")
                
                f.write("\n---\n\n")

async def main():
    """Main function"""
    tester = PromptQualityTester()
    
    try:
        results = await tester.run_quality_test()
        
        # Generate run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        output_dir = tester.save_results(results, run_id)
        
        # Print summary
        print("\n" + "="*80)
        print("🧪 PROMPT RESTRUCTURE QUALITY TEST COMPLETE")
        print("="*80)
        print(f"📋 Run ID: {run_id}")
        print(f"📊 Results:")
        print(f"   - Success Rate: {results['metrics']['success_rate']:.1f}%")
        print(f"   - RAG Usage Rate: {results['metrics']['rag_usage_rate']:.1f}%")
        print(f"   - Template-like Rate: {results['metrics']['template_like_rate']:.1f}%")
        print(f"   - Context Usage Rate: {results['metrics']['context_usage_rate']:.1f}%")
        print(f"   - Avg Response Length: {results['metrics']['avg_response_length']:.1f} chars")
        print(f"📁 Output: {output_dir}")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Quality test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
