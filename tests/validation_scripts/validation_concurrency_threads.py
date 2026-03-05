#!/usr/bin/env python3
"""
VALIDAÇÃO DE CONCORRÊNCIA MULTI-THREAD - Pipeline LLM
Usa ThreadPoolExecutor para simular runtime multi-thread real
"""

import os
import sys
import json
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adicionar o projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from support_agent.llm.llm_manager import LLMManager

@dataclass
class ThreadResult:
    """Resultado de uma requisição em thread"""
    thread_id: str
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
    status: str  # success/fail/mismatch
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    thread_name: str = ""

class MultiThreadConcurrencyValidator:
    def __init__(self):
        self.setup_logging()
        self.results = []
        self.lock = threading.Lock()
        self.thread_counter = 0
        
    def setup_logging(self):
        """Configurar logging thread-safe"""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Limpar handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler para capturar logs estruturados
        log_file = project_root / "validation_logs" / f"concurrency_threads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
    
    def make_thread_request(self, task_id: str, bucket: str, model: str) -> ThreadResult:
        """Faz uma requisição em thread separada com contexto A/B explícito"""
        thread_id = f"thread_{threading.get_ident()}"
        timestamp_start = time.time()
        
        result = ThreadResult(
            thread_id=thread_id,
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
            status="fail",
            thread_name=threading.current_thread().name
        )
        
        try:
            # Criar contexto A/B explícito
            context = {
                "ab_testing_active": True,
                "ab_model": model,
                "user_id": f"thread_user_{task_id}",
                "bucket": bucket
            }
            
            # Criar LLM Manager com contexto
            llm_manager = LLMManager()
            
            # 🚨 CORREÇÃO: Usar asyncio.run() em thread para evitar problemas
            try:
                # Inicializar e gerar resposta de forma síncrona
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Executar tarefas assíncronas
                loop.run_until_complete(llm_manager.initialize(context))
                response = loop.run_until_complete(llm_manager.generate_response("Thread test " + task_id, context=context))
                
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
                
            finally:
                # Garantir que o loop seja fechado
                try:
                    loop.close()
                except:
                    pass
                
        except Exception as e:
            timestamp_end = time.time()
            result.timestamp_end = timestamp_end
            result.latency_ms = (timestamp_end - timestamp_start) * 1000
            result.error = str(e)
            result.status = "fail"
        
        # Armazenar resultado com lock
        with self.lock:
            self.results.append(result)
        
        return result
    
    def run_thread_test(self, num_workers: int = 20, num_tasks_per_bucket: int = 100) -> Dict[str, Any]:
        """Executa teste de concorrência com ThreadPoolExecutor"""
        self.logger.info(f"🧵 INICIANDO VALIDAÇÃO DE CONCORRÊNCIA MULTI-THREAD")
        self.logger.info(f"👥 Workers: {num_workers}")
        self.logger.info(f"📊 Tasks por bucket: {num_tasks_per_bucket}")
        self.logger.info(f"📊 Total tasks: {num_tasks_per_bucket * 2}")
        
        # Configurar ambiente para A/B
        os.environ["LLM_MODEL_PRIMARY"] = "gpt-4o"  # Será sobrescrito pelo contexto
        
        # Criar tasks para Bucket A (gpt-4o)
        tasks_bucket_a = []
        for i in range(num_tasks_per_bucket):
            task_id = f"A_{i:03d}"
            task = (self.make_thread_request, (task_id, "A", "gpt-4o"))
            tasks_bucket_a.append(task)
        
        # Criar tasks para Bucket B (gpt-5.1)
        tasks_bucket_b = []
        for i in range(num_tasks_per_bucket):
            task_id = f"B_{i:03d}"
            task = (self.make_thread_request, (task_id, "B", "gpt-5.1"))
            tasks_bucket_b.append(task)
        
        # Combinar todas as tasks
        all_tasks = tasks_bucket_a + tasks_bucket_b
        
        # Executar com ThreadPoolExecutor
        self.logger.info(f"⚡ Executando {len(all_tasks)} tasks com {num_workers} workers...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submeter todas as tasks
            future_to_task = {executor.submit(task_func, *args): (task_func, args) for task_func, args in all_tasks}
            
            # Coletar resultados na ordem de conclusão
            results = []
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analisar resultados
        analysis = self.analyze_results(results, total_time, num_workers)
        
        # Salvar logs estruturados
        self.save_structured_logs()
        
        return analysis
    
    def analyze_results(self, results: List[ThreadResult], total_time: float, num_workers: int) -> Dict[str, Any]:
        """Analisa os resultados do teste multi-thread"""
        self.logger.info("📊 Analisando resultados multi-thread...")
        
        # Estatísticas básicas
        total_calls = len(results)
        success_count = sum(1 for r in results if r.status == "success")
        fail_count = sum(1 for r in results if r.status == "fail")
        mismatch_count = sum(1 for r in results if r.status == "mismatch")
        
        # Análise por bucket
        bucket_a_results = [r for r in results if r.expected_bucket == "A"]
        bucket_b_results = [r for r in results if r.expected_bucket == "B"]
        
        # Validar unicidade de IDs
        trace_ids = [r.trace_id for r in results if r.trace_id]
        request_ids = [r.request_id for r in results if r.request_id]
        
        unique_trace_ids = len(trace_ids) == len(set(trace_ids))
        unique_request_ids = len(request_ids) == len(set(request_ids))
        
        # Validar consistência de modelos por bucket
        bucket_a_correct = all(r.model_sent_to_api == "gpt-4o" for r in bucket_a_results)
        bucket_b_correct = all(r.model_sent_to_api == "gpt-5.1" for r in bucket_b_results)
        
        # Análise de threads
        thread_ids = [r.thread_id for r in results]
        unique_threads = len(set(thread_ids))
        thread_names = [r.thread_name for r in results]
        
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
            "test_type": "MULTI_THREAD_CONCURRENCY_VALIDATION",
            "timestamp": datetime.now().isoformat(),
            "test_duration_seconds": total_time,
            "num_workers": num_workers,
            "total_calls": total_calls,
            "success_count": success_count,
            "fail_count": fail_count,
            "mismatch_count": mismatch_count,
            "unique_trace_ids": unique_trace_ids,
            "unique_request_ids": unique_request_ids,
            "thread_analysis": {
                "unique_threads": unique_threads,
                "thread_names": list(set(thread_names)),
                "avg_tasks_per_thread": total_calls / unique_threads if unique_threads > 0 else 0
            },
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
                "unique_request_ids": unique_request_ids,
                "thread_safe": unique_threads >= num_workers * 0.8  # Pelo menos 80% dos workers usados
            }
        }
        
        return analysis
    
    def save_structured_logs(self):
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
        print("🧵 RELATÓRIO DE VALIDAÇÃO DE CONCORRÊNCIA MULTI-THREAD")
        print("="*100)
        
        status_icon = "✅" if analysis["status"] == "PASS" else "❌"
        print(f"{status_icon} Status: {analysis['status']}")
        print(f"👥 Workers: {analysis['num_workers']}")
        print(f"📊 Total Calls: {analysis['total_calls']}")
        print(f"✅ Success: {analysis['success_count']}")
        print(f"❌ Fail: {analysis['fail_count']}")
        print(f"🔍 Mismatch: {analysis['mismatch_count']}")
        print(f"🆔 Unique Trace IDs: {analysis['unique_trace_ids']}")
        print(f"📋 Unique Request IDs: {analysis['unique_request_ids']}")
        print(f"⚡ Throughput: {analysis['throughput']:.2f} calls/sec")
        print(f"⏱️ Test Duration: {analysis['test_duration_seconds']:.2f}s")
        print()
        
        print("👥 Análise de Threads:")
        thread_analysis = analysis["thread_analysis"]
        print(f"  Unique Threads: {thread_analysis['unique_threads']}")
        print(f"  Thread Names: {thread_analysis['thread_names']}")
        print(f"  Avg Tasks/Thread: {thread_analysis['avg_tasks_per_thread']:.1f}")
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
            print("✅ VALIDAÇÃO DE CONCORRÊNCIA MULTI-THREAD PASSOU")
            print("✅ Bucket A sempre usa gpt-4o")
            print("✅ Bucket B sempre usa gpt-5.1")
            print("✅ Nenhum vazamento entre buckets")
            print("✅ Todos os IDs são únicos")
            print("✅ Pipeline seguro para execução multi-thread")
            print("✅ ThreadPoolExecutor funciona corretamente")
        else:
            print("❌ VALIDAÇÃO DE CONCORRÊNCIA MULTI-THREAD FALHOU")
            print("❌ Problemas detectados na execução multi-thread")
        
        print("="*100)

def main():
    """Main execution"""
    validator = MultiThreadConcurrencyValidator()
    
    print("🧵 VALIDAÇÃO DE CONCORRÊNCIA MULTI-THREAD - PIPELINE LLM")
    print("🎯 Objetivo: 200 chamadas com ThreadPoolExecutor (20 workers)")
    print("📋 Validar isolamento, consistência e thread-safety")
    print()
    
    try:
        # Executar teste
        analysis = validator.run_thread_test(num_workers=20, num_tasks_per_bucket=100)
        
        # Salvar relatório
        reports_dir = Path("validation_reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"CONCURRENCY_THREADS_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    exit_code = main()
    sys.exit(exit_code)
