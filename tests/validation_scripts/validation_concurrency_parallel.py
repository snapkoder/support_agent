#!/usr/bin/env python3
"""
VALIDAÇÃO DE CONCORRÊNCIA PARALELA - Pipeline LLM
Executa 100 chamadas simultâneas para validar isolamento e consistência
"""

import os
import sys
import json
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

# Adicionar o projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from support_agent.llm.llm_manager import LLMManager

@dataclass
class ConcurrencyResult:
    """Resultado de uma requisição concorrente"""
    task_id: str
    expected_bucket: str
    expected_model: str
    model_selected: str
    model_sent_to_api: str
    model_used: str
    trace_id: str
    request_id: str
    timestamp_start: float
    timestamp_end: float
    status: str  # success/fail
    error: Optional[str] = None
    latency_ms: Optional[float] = None

class ParallelConcurrencyValidator:
    def __init__(self):
        self.setup_logging()
        self.results = []
        self.lock = asyncio.Lock()
        
    def setup_logging(self):
        """Configurar logging"""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Limpar handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler para capturar logs estruturados
        log_file = project_root / "validation_logs" / f"concurrency_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
    
    async def make_concurrent_request(self, task_id: str, bucket: str, model: str) -> ConcurrencyResult:
        """Faz uma requisição concorrente com contexto A/B explícito"""
        timestamp_start = time.time()
        
        result = ConcurrencyResult(
            task_id=task_id,
            expected_bucket=bucket,
            expected_model=model,
            model_selected="",
            model_sent_to_api="",
            model_used="",
            trace_id="",
            request_id="",
            timestamp_start=timestamp_start,
            timestamp_end=0,
            status="fail"
        )
        
        try:
            # Criar contexto A/B explícito
            context = {
                "ab_testing_active": True,
                "ab_model": model,
                "user_id": f"concurrent_user_{task_id}",
                "bucket": bucket
            }
            
            # Criar LLM Manager com contexto
            llm_manager = LLMManager()
            await llm_manager.initialize(context)
            
            # Fazer chamada
            prompt = f"Concurrency test {task_id} - bucket {bucket}. Respond just with 'OK'."
            response = await llm_manager.generate_response(prompt, context=context)
            
            # Capturar resultados
            timestamp_end = time.time()
            result.timestamp_end = timestamp_end
            result.latency_ms = (timestamp_end - timestamp_start) * 1000
            
            result.model_selected = getattr(response, 'model_selected', '')
            result.model_sent_to_api = getattr(response, 'model_sent_to_api', '')
            result.model_used = getattr(response, 'model_used', '')
            result.trace_id = getattr(response, 'trace_id', '')
            result.request_id = getattr(response, 'request_id', '')
            result.status = "success"
            
            # Validar consistência
            if result.model_selected != model or result.model_sent_to_api != model or result.model_used != model:
                result.status = "mismatch"
                result.error = f"Model mismatch: expected {model}, got selected={result.model_selected}, sent={result.model_sent_to_api}, used={result.model_used}"
            
        except Exception as e:
            timestamp_end = time.time()
            result.timestamp_end = timestamp_end
            result.latency_ms = (timestamp_end - timestamp_start) * 1000
            result.error = str(e)
            result.status = "fail"
        
        # Armazenar resultado com lock
        async with self.lock:
            self.results.append(result)
        
        return result
    
    async def run_concurrency_test(self, num_tasks_per_bucket: int = 50) -> Dict[str, Any]:
        """Executa teste de concorrência com N tasks por bucket"""
        self.logger.info(f"🚀 INICIANDO VALIDAÇÃO DE CONCORRÊNCIA PARALELA")
        self.logger.info(f"📊 Tasks por bucket: {num_tasks_per_bucket}")
        self.logger.info(f"📊 Total tasks: {num_tasks_per_bucket * 2}")
        
        # Configurar ambiente para A/B
        os.environ["LLM_MODEL_PRIMARY"] = "gpt-4o"  # Será sobrescrito pelo contexto
        
        # Criar tasks para Bucket A (gpt-4o)
        tasks_bucket_a = []
        for i in range(num_tasks_per_bucket):
            task_id = f"A_{i:03d}"
            task = self.make_concurrent_request(task_id, "A", "gpt-4o")
            tasks_bucket_a.append(task)
        
        # Criar tasks para Bucket B (gpt-5.1)
        tasks_bucket_b = []
        for i in range(num_tasks_per_bucket):
            task_id = f"B_{i:03d}"
            task = self.make_concurrent_request(task_id, "B", "gpt-5.1")
            tasks_bucket_b.append(task)
        
        # Combinar todas as tasks
        all_tasks = tasks_bucket_a + tasks_bucket_b
        
        # Executar em paralelo
        self.logger.info("⚡ Executando {len(all_tasks)} tasks simultâneas...")
        start_time = time.time()
        
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analisar resultados
        analysis = await self.analyze_results(results, total_time)
        
        # Salvar logs estruturados
        await self.save_structured_logs()
        
        return analysis
    
    async def analyze_results(self, results: List[Any], total_time: float) -> Dict[str, Any]:
        """Analisa os resultados do teste de concorrência"""
        self.logger.info("📊 Analisando resultados...")
        
        # Filtrar apenas resultados válidos
        valid_results = [r for r in results if isinstance(r, ConcurrencyResult)]
        
        # Estatísticas básicas
        total_calls = len(valid_results)
        success_count = sum(1 for r in valid_results if r.status == "success")
        fail_count = sum(1 for r in valid_results if r.status == "fail")
        mismatch_count = sum(1 for r in valid_results if r.status == "mismatch")
        
        # Análise por bucket
        bucket_a_results = [r for r in valid_results if r.expected_bucket == "A"]
        bucket_b_results = [r for r in valid_results if r.expected_bucket == "B"]
        
        # Validar unicidade de IDs
        trace_ids = [r.trace_id for r in valid_results if r.trace_id]
        request_ids = [r.request_id for r in valid_results if r.request_id]
        
        unique_trace_ids = len(trace_ids) == len(set(trace_ids))
        unique_request_ids = len(request_ids) == len(set(request_ids))
        
        # Validar consistência de modelos por bucket
        bucket_a_correct = all(r.model_sent_to_api == "gpt-4o" for r in bucket_a_results)
        bucket_b_correct = all(r.model_sent_to_api == "gpt-5.1" for r in bucket_b_results)
        
        # Calcular latências
        latencies_a = [r.latency_ms for r in bucket_a_results if r.latency_ms]
        latencies_b = [r.latency_ms for r in bucket_b_results if r.latency_ms]
        
        def calculate_percentiles(latencies):
            if not latencies:
                return {"p50": 0, "p95": 0, "mean": 0}
            return {
                "p50": statistics.median(latencies),
                "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                "mean": statistics.mean(latencies)
            }
        
        latency_stats = {
            "bucket_a": calculate_percentiles(latencies_a),
            "bucket_b": calculate_percentiles(latencies_b),
            "overall": calculate_percentiles(latencies_a + latencies_b)
        }
        
        # Determinar status final
        all_criteria_passed = (
            bucket_a_correct and 
            bucket_b_correct and 
            unique_trace_ids and 
            unique_request_ids and 
            mismatch_count == 0
        )
        
        analysis = {
            "test_type": "PARALLEL_CONCURRENCY_VALIDATION",
            "timestamp": datetime.now().isoformat(),
            "test_duration_seconds": total_time,
            "total_calls": total_calls,
            "success_count": success_count,
            "fail_count": fail_count,
            "mismatch_count": mismatch_count,
            "unique_trace_ids": unique_trace_ids,
            "unique_request_ids": unique_request_ids,
            "bucket_analysis": {
                "bucket_a": {
                    "expected_model": "gpt-4o",
                    "total_tasks": len(bucket_a_results),
                    "correct_models": bucket_a_correct,
                    "model_consistency": f"{sum(1 for r in bucket_a_results if r.model_sent_to_api == 'gpt-4o')}/{len(bucket_a_results)}"
                },
                "bucket_b": {
                    "expected_model": "gpt-5.1",
                    "total_tasks": len(bucket_b_results),
                    "correct_models": bucket_b_correct,
                    "model_consistency": f"{sum(1 for r in bucket_b_results if r.model_sent_to_api == 'gpt-5.1')}/{len(bucket_b_results)}"
                }
            },
            "latency_stats": latency_stats,
            "throughput": total_calls / total_time if total_time > 0 else 0,
            "status": "PASS" if all_criteria_passed else "FAIL",
            "criteria": {
                "bucket_a_correct": bucket_a_correct,
                "bucket_b_correct": bucket_b_correct,
                "no_mismatches": mismatch_count == 0,
                "unique_trace_ids": unique_trace_ids,
                "unique_request_ids": unique_request_ids
            }
        }
        
        return analysis
    
    async def save_structured_logs(self):
        """Salva logs estruturados em formato JSONL"""
        self.logger.info("💾 Salvando logs estruturados...")
        
        # Ler logs do arquivo e extrair eventos relevantes
        structured_logs = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                if '"event": "llm_request"' in line:
                    try:
                        # Extrair JSON da linha
                        json_start = line.find('{')
                        if json_start != -1:
                            json_str = line[json_start:]
                            log_data = json.loads(json_str)
                            structured_logs.append(log_data)
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            self.logger.error(f"Erro ao ler logs: {e}")
        
        # Salvar logs estruturados
        logs_file = self.log_file.with_suffix('.jsonl')
        with open(logs_file, 'w', encoding='utf-8') as f:
            for log in structured_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')
        
        self.logger.info(f"💾 {len(structured_logs)} logs estruturados salvos em {logs_file}")
        
        return structured_logs
    
    def print_summary(self, analysis: Dict[str, Any], structured_logs: List[Dict]):
        """Imprime resumo dos resultados"""
        print("\n" + "="*100)
        print("🚀 RELATÓRIO DE VALIDAÇÃO DE CONCORRÊNCIA PARALELA")
        print("="*100)
        
        status_icon = "✅" if analysis["status"] == "PASS" else "❌"
        print(f"{status_icon} Status: {analysis['status']}")
        print(f"📊 Total Calls: {analysis['total_calls']}")
        print(f"✅ Success: {analysis['success_count']}")
        print(f"❌ Fail: {analysis['fail_count']}")
        print(f"🔍 Mismatch: {analysis['mismatch_count']}")
        print(f"🆔 Unique Trace IDs: {analysis['unique_trace_ids']}")
        print(f"📋 Unique Request IDs: {analysis['unique_request_ids']}")
        print(f"⚡ Throughput: {analysis['throughput']:.2f} calls/sec")
        print(f"⏱️ Test Duration: {analysis['test_duration_seconds']:.2f}s")
        print()
        
        print("📊 Análise por Bucket:")
        for bucket_name, bucket_data in analysis["bucket_analysis"].items():
            print(f"  {bucket_name.upper()}:")
            print(f"    Expected Model: {bucket_data['expected_model']}")
            print(f"    Total Tasks: {bucket_data['total_tasks']}")
            print(f"    Correct Models: {bucket_data['correct_models']}")
            print(f"    Consistency: {bucket_data['model_consistency']}")
        print()
        
        print("⏱️ Latency (ms):")
        latency = analysis["latency_stats"]
        print(f"  Bucket A: P50={latency['bucket_a']['p50']:.1f}, P95={latency['bucket_a']['p95']:.1f}, Mean={latency['bucket_a']['mean']:.1f}")
        print(f"  Bucket B: P50={latency['bucket_b']['p50']:.1f}, P95={latency['bucket_b']['p95']:.1f}, Mean={latency['bucket_b']['mean']:.1f}")
        print(f"  Overall: P50={latency['overall']['p50']:.1f}, P95={latency['overall']['p95']:.1f}, Mean={latency['overall']['mean']:.1f}")
        print()
        
        print("🔍 Critérios de Sucesso:")
        criteria = analysis["criteria"]
        for criterion, passed in criteria.items():
            icon = "✅" if passed else "❌"
            print(f"  {icon} {criterion}: {passed}")
        print()
        
        print("📋 Logs de Exemplo (llm_request):")
        for i, log in enumerate(structured_logs[:5]):
            model = log.get("model_sent_to_api", "unknown")
            trace_id = log.get("trace_id", "unknown")
            print(f"  {i+1}. Bucket {'A' if model == 'gpt-4o' else 'B'} → {model} (trace: {trace_id})")
        
        print()
        print("🎯 Conclusão:")
        if analysis["status"] == "PASS":
            print("✅ VALIDAÇÃO DE CONCORRÊNCIA PASSOU")
            print("✅ Bucket A sempre usa gpt-4o")
            print("✅ Bucket B sempre usa gpt-5.1")
            print("✅ Nenhum vazamento entre buckets")
            print("✅ Todos os IDs são únicos")
            print("✅ Pipeline seguro para execução paralela")
        else:
            print("❌ VALIDAÇÃO DE CONCORRÊNCIA FALHOU")
            print("❌ Problemas detectados na execução paralela")
        
        print("="*100)

async def main():
    """Main execution"""
    validator = ParallelConcurrencyValidator()
    
    print("🚀 VALIDAÇÃO DE CONCORRÊNCIA PARALELA - PIPELINE LLM")
    print("🎯 Objetivo: 100 chamadas simultâneas (50 A + 50 B)")
    print("📋 Validar isolamento, consistência e unicidade de IDs")
    print()
    
    try:
        # Executar teste
        analysis = await validator.run_concurrency_test(num_tasks_per_bucket=50)
        
        # Salvar relatório
        reports_dir = Path("validation_reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"CONCURRENCY_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Ler logs estruturados para exibição
        logs_file = validator.log_file.with_suffix('.jsonl')
        structured_logs = []
        if logs_file.exists():
            with open(logs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        structured_logs.append(json.loads(line.strip()))
        
        # Imprimir resumo
        validator.print_summary(analysis, structured_logs)
        
        print(f"\n📁 Relatório completo: {report_file}")
        print(f"📁 Logs estruturados: {logs_file}")
        
        return 0 if analysis["status"] == "PASS" else 1
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO NA VALIDAÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
