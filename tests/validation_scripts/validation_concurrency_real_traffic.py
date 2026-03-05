#!/usr/bin/env python3
"""
VALIDAÇÃO DE CONCORRÊNCIA - TRÁFEGO REAL WHATSAPP
Simula 300 chamadas paralelas com prompts reais de suporte
Valida isolamento completo entre requests
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics

# Adicionar o projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from support_agent.llm.llm_manager import LLMManager

@dataclass
class RealTrafficResult:
    """Resultado de uma requisição com tráfego real"""
    task_id: str
    request_tag: str
    expected_bucket: str
    expected_model: str
    prompt_type: str
    prompt_length: int
    model_selected: str
    model_sent_to_api: str
    model_used: str
    trace_id: str
    request_id: str
    response_content: str
    response_length: int
    timestamp_start: float
    timestamp_end: float
    status: str  # success/fail/mismatch/tag_leak
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    tag_leak_detected: bool = False
    leaked_tags: List[str] = None

class RealTrafficConcurrencyValidator:
    def __init__(self):
        self.setup_logging()
        self.results = []
        self.lock = asyncio.Lock()
        self.request_tags = set()
        
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
        log_file = project_root / "validation_logs" / f"real_traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
    
    def get_real_prompts(self) -> List[Dict[str, str]]:
        """Retorna 30 prompts reais de suporte WhatsApp"""
        return [
            # Curtos (1-10)
            {"type": "curto", "text": "quero 2ª via do boleto"},
            {"type": "curto", "text": "não consigo fazer pix"},
            {"type": "curto", "text": "meu cartão foi bloqueado"},
            {"type": "curto", "text": "esqueci minha senha"},
            {"type": "curto", "text": "onde fico minha agência"},
            {"type": "curto", "text": "como cancelo conta"},
            {"type": "curto", "text": "meu limite baixou"},
            {"type": "curto", "text": "quero empréstimo"},
            {"type": "curto", "text": "app não funciona"},
            {"type": "curto", "text": "saldo zerado"},
            
            # Médios (11-20)
            {"type": "medio", "text": "preciso fazer transferência para outra conta mas não sei como usar o app"},
            {"type": "medio", "text": "recebi uma cobrança indevida e preciso contestar o valor cobrado"},
            {"type": "medio", "text": "meu cartão de crédito chegou com a fatura errada e preciso corrigir"},
            {"type": "medio", "text": "quero abrir uma conta poupança para meu filho menor de idade"},
            {"type": "medio", "text": "não estou recebendo os alertas de transação no meu celular"},
            {"type": "medio", "text": "preciso atualizar meu endereço pois mudei de casa recentemente"},
            {"type": "medio", "text": "como faço para desbloquear meu aplicativo que está travado"},
            {"type": "medio", "text": "quero saber como funciona o investimento em CDB do banco"},
            {"type": "medio", "text": "meu cartão de débito não funciona na maquininha de alguns lugares"},
            {"type": "medio", "text": "preciso sacar dinheiro mas não tenho agência perto de mim"},
            
            # Longos (21-30)
            {"type": "longo", "text": "Olá, sou cliente há mais de 10 anos e estou com um problema sério. Recentemente fiz uma compra internacional e minha conta foi bloqueada sem aviso prévio. Já liguei várias vezes para o suporte mas não consigo falar com um humano. Preciso desbloquear urgentemente pois tenho contas para pagar e não consigo acessar meu dinheiro. Podem me ajudar com isso?"},
            {"type": "longo", "text": "Bom dia, estou tentando fazer um pix de alto valor para pagamento de imóvel, mas o sistema não permite. Já verifiquei meu limite diário e mensal, e deveria ser possível. O erro diz que há uma restrição de segurança, mas nunca tive problemas antes. Preciso liberar essa transação hoje pois o vendedor está aguardando. Como posso resolver?"},
            {"type": "longo", "text": "Prezados, estou extremamente preocupado. Apareceu uma transação na minha fatura que não reconheço. É um valor alto e nunca usei meu cartão nesse estabelecimento. Acho que meu cartão foi clonado. Já cancelei o cartão mas preciso entender como aconteceu e garantir que não terei prejuízo financeiro. Preciso abrir um protocolo de fraude."},
            {"type": "longo", "text": "Senhores, sou empresário e preciso de uma solução urgente. Minha conta PJ está com problemas para receber pagamentos dos clientes. Eles dizem que a conta está com restrições, mas não entendo o motivo. Minha empresa está regular, todos os documentos em dia. Preciso resolver isso pois meu fluxo de caixa está comprometido. Podem analisar meu caso com prioridade?"},
            {"type": "longo", "text": "Caro suporte, estou planejando minha aposentadoria e preciso de orientação. Tenho 45 anos, trabalho como autônomo e quero começar a investir. Gostaria de saber quais são as melhores opções de investimento no banco, quais as taxas, riscos, e como posso começar com um valor baixo. Também preciso entender sobre resgate e tributação. Podem me ajudar com esse planejamento?"},
            {"type": "longo", "text": "Olá equipe, estou com um problema complexo. Herdei uma conta bancária de meu falecido pai e preciso fazer a transferência dos valores para minha conta. Já apresentei todos os documentos necessários (certidão de óbito, inventário, alvará), mas o banco continua pedindo mais documentos. Já faz 3 meses que estou tentando resolver isso. Preciso de ajuda para finalizar esse processo e liberar o dinheiro para minha família. Podem acelerar esse caso?"},
            {"type": "longo", "text": "Prezados, sou estudante universitário e tenho uma conta universitária no banco. Recentemente comecei a receber mensagens de cobrança por serviços que não contratei. Quando verifico, vejo que estão sendo debitados valores da minha conta sem autorização. Já cancelei o cartão mas os débitos continuam. Acho que houve algum tipo de fraude ou erro sistêmico. Preciso urgentemente parar esses débitos e reaver o dinheiro perdido. Como proceder?"},
            {"type": "longo", "text": "Senhores, estou com uma situação delicada. Meu nome consta negativado em órgãos de proteção ao crédito, mas isso é um erro. Já paguei todas as minhas dívidas e tenho os comprovantes. No entanto, quando tento fazer um financiamento ou abrir crédito, sou negado. Preciso regularizar essa situação pois preciso de crédito para meu negócio. Como posso limpar meu nome e garantir que as informações corretas constem nos bureaus de crédito?"},
            {"type": "longo", "text": "Bom dia a todos. Sou cliente internacional e estou enfrentando dificuldades com transferências. Preciso enviar dinheiro para minha família no exterior, mas as taxas são muito altas e o processo é burocrático. Já pesquisei outras opções mas o banco me oferece as melhores condições. Gostaria de entender se há alguma forma de reduzir os custos, se há programas para clientes internacionais, e qual a documentação necessária. Também preciso saber sobre limites e prazos. Aguardo orientação."}
        ]
    
    async def make_real_traffic_request(self, task_id: str, bucket: str, model: str, prompt_data: Dict[str, str]) -> RealTrafficResult:
        """Faz uma requisição com tráfego real e validação de isolamento"""
        timestamp_start = time.time()
        
        # Gerar tag única para este request
        request_tag = f"REQ_{uuid.uuid4().hex[:8]}"
        
        result = RealTrafficResult(
            task_id=task_id,
            request_tag=request_tag,
            expected_bucket=bucket,
            expected_model=model,
            prompt_type=prompt_data["type"],
            prompt_length=len(prompt_data["text"]),
            model_selected="",
            model_sent_to_api="",
            model_used="",
            trace_id="",
            request_id="",
            response_content="",
            response_length=0,
            timestamp_start=timestamp_start,
            timestamp_end=0,
            status="fail",
            leaked_tags=[]
        )
        
        try:
            # Criar contexto A/B explícito
            context = {
                "ab_testing_active": True,
                "ab_model": model,
                "user_id": f"whatsapp_user_{task_id}",
                "bucket": bucket
            }
            
            # Criar LLM Manager com contexto
            llm_manager = LLMManager()
            await llm_manager.initialize(context)
            
            # Preparar prompt com tag única
            tagged_prompt = f"{prompt_data['text']}. [request_tag={request_tag}]"
            
            # Fazer chamada
            response = await llm_manager.generate_response(tagged_prompt, context=context)
            
            # Capturar resultados
            timestamp_end = time.time()
            result.timestamp_end = timestamp_end
            result.latency_ms = (timestamp_end - timestamp_start) * 1000
            
            result.model_selected = getattr(response, 'model_selected', '')
            result.model_sent_to_api = getattr(response, 'model_sent_to_api', '')
            result.model_used = getattr(response, 'model_used', '')
            result.trace_id = getattr(response, 'trace_id', '')
            result.request_id = getattr(response, 'request_id', '')
            result.response_content = getattr(response, 'content', '')
            result.response_length = len(result.response_content)
            
            # Validar consistência de modelo
            model_consistent = (result.model_selected == model and 
                              result.model_sent_to_api == model and 
                              result.model_used == model)
            
            # Validar isolamento de tags - detectar vazamento
            leaked_tags = []
            for other_tag in self.request_tags:
                if other_tag != request_tag and other_tag in result.response_content:
                    leaked_tags.append(other_tag)
            
            result.leaked_tags = leaked_tags
            result.tag_leak_detected = len(leaked_tags) > 0
            
            # Determinar status
            if not model_consistent:
                result.status = "mismatch"
                result.error = f"Model mismatch: expected {model}, got selected={result.model_selected}, sent={result.model_sent_to_api}, used={result.model_used}"
            elif result.tag_leak_detected:
                result.status = "tag_leak"
                result.error = f"Tag leak detected: {leaked_tags}"
            else:
                result.status = "success"
            
        except Exception as e:
            timestamp_end = time.time()
            result.timestamp_end = timestamp_end
            result.latency_ms = (timestamp_end - timestamp_start) * 1000
            result.error = str(e)
            result.status = "fail"
        
        # Armazenar resultado e tag com lock
        async with self.lock:
            self.results.append(result)
            self.request_tags.add(request_tag)
        
        return result
    
    async def run_real_traffic_test(self, num_calls: int = 300) -> Dict[str, Any]:
        """Executa teste de concorrência com tráfego real"""
        self.logger.info(f"📱 INICIANDO VALIDAÇÃO DE CONCORRÊNCIA - TRÁFEGO REAL WHATSAPP")
        self.logger.info(f"📊 Total calls: {num_calls}")
        
        # Configurar ambiente para A/B
        os.environ["LLM_MODEL_PRIMARY"] = "gpt-4o"  # Será sobrescrito pelo contexto
        
        # Obter prompts reais
        real_prompts = self.get_real_prompts()
        
        # Criar tasks com distribuição aleatória
        tasks = []
        for i in range(num_calls):
            # Alternar buckets A/B
            bucket = "A" if i % 2 == 0 else "B"
            model = "gpt-4o" if bucket == "A" else "gpt-5.1"
            
            # Selecionar prompt aleatório
            prompt_data = random.choice(real_prompts)
            
            task_id = f"TRAFFIC_{i:03d}"
            task = self.make_real_traffic_request(task_id, bucket, model, prompt_data)
            tasks.append(task)
        
        # Executar em paralelo
        self.logger.info(f"⚡ Executando {len(tasks)} chamadas simultâneas...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analisar resultados
        analysis = await self.analyze_results(results, total_time)
        
        # Salvar logs estruturados
        await self.save_structured_logs()
        
        return analysis
    
    async def analyze_results(self, results: List[Any], total_time: float) -> Dict[str, Any]:
        """Analisa os resultados do teste de tráfego real"""
        self.logger.info("📊 Analisando resultados de tráfego real...")
        
        # Filtrar apenas resultados válidos
        valid_results = [r for r in results if isinstance(r, RealTrafficResult)]
        
        # Estatísticas básicas
        total_calls = len(valid_results)
        success_count = sum(1 for r in valid_results if r.status == "success")
        fail_count = sum(1 for r in valid_results if r.status == "fail")
        mismatch_count = sum(1 for r in valid_results if r.status == "mismatch")
        tag_leak_count = sum(1 for r in valid_results if r.status == "tag_leak")
        
        # Análise por bucket
        bucket_a_results = [r for r in valid_results if r.expected_bucket == "A"]
        bucket_b_results = [r for r in valid_results if r.expected_bucket == "B"]
        
        # Análise por tipo de prompt
        prompt_types = defaultdict(list)
        for r in valid_results:
            prompt_types[r.prompt_type].append(r)
        
        # Validar unicidade de IDs
        trace_ids = [r.trace_id for r in valid_results if r.trace_id]
        request_ids = [r.request_id for r in valid_results if r.request_id]
        request_tags = [r.request_tag for r in valid_results if r.request_tag]
        
        unique_trace_ids = len(trace_ids) == len(set(trace_ids))
        unique_request_ids = len(request_ids) == len(set(request_ids))
        unique_request_tags = len(request_tags) == len(set(request_tags))
        
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
        
        # Análise de vazamento de tags
        total_leaked_tags = sum(len(r.leaked_tags) for r in valid_results)
        requests_with_leaks = [r for r in valid_results if r.tag_leak_detected]
        
        # Análise de tamanho de prompts
        prompt_lengths = [r.prompt_length for r in valid_results]
        response_lengths = [r.response_length for r in valid_results]
        
        # Determinar status final
        all_criteria_passed = (
            bucket_a_correct and 
            bucket_b_correct and 
            unique_trace_ids and 
            unique_request_ids and 
            unique_request_tags and
            mismatch_count == 0 and
            tag_leak_count == 0
        )
        
        analysis = {
            "test_type": "REAL_TRAFFIC_CONCURRENCY_VALIDATION",
            "timestamp": datetime.now().isoformat(),
            "test_duration_seconds": total_time,
            "total_calls": total_calls,
            "success_count": success_count,
            "fail_count": fail_count,
            "mismatch_count": mismatch_count,
            "tag_leak_count": tag_leak_count,
            "unique_trace_ids": unique_trace_ids,
            "unique_request_ids": unique_request_ids,
            "unique_request_tags": unique_request_tags,
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
            "prompt_analysis": {
                "types": {ptype: len(prompts) for ptype, prompts in prompt_types.items()},
                "avg_prompt_length": statistics.mean(prompt_lengths) if prompt_lengths else 0,
                "avg_response_length": statistics.mean(response_lengths) if response_lengths else 0,
                "prompt_length_range": (min(prompt_lengths), max(prompt_lengths)) if prompt_lengths else (0, 0)
            },
            "tag_leak_analysis": {
                "total_leaked_tags": total_leaked_tags,
                "requests_with_leaks": len(requests_with_leaks),
                "leak_rate": len(requests_with_leaks) / total_calls if total_calls > 0 else 0,
                "leaked_tags_examples": [r.leaked_tags[:3] for r in requests_with_leaks[:5]]
            },
            "latency_stats": latency_stats,
            "throughput": total_calls / total_time if total_time > 0 else 0,
            "status": "PASS" if all_criteria_passed else "FAIL",
            "criteria": {
                "bucket_a_correct": bucket_a_correct,
                "bucket_b_correct": bucket_b_correct,
                "no_mismatches": mismatch_count == 0,
                "no_tag_leaks": tag_leak_count == 0,
                "unique_trace_ids": unique_trace_ids,
                "unique_request_ids": unique_request_ids,
                "unique_request_tags": unique_request_tags
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
        print("📱 RELATÓRIO DE VALIDAÇÃO DE CONCORRÊNCIA - TRÁFEGO REAL WHATSAPP")
        print("="*100)
        
        status_icon = "✅" if analysis["status"] == "PASS" else "❌"
        print(f"{status_icon} Status: {analysis['status']}")
        print(f"📊 Total Calls: {analysis['total_calls']}")
        print(f"✅ Success: {analysis['success_count']}")
        print(f"❌ Fail: {analysis['fail_count']}")
        print(f"🔍 Mismatch: {analysis['mismatch_count']}")
        print(f"🏷️ Tag Leak: {analysis['tag_leak_count']}")
        print(f"🆔 Unique Trace IDs: {analysis['unique_trace_ids']}")
        print(f"📋 Unique Request IDs: {analysis['unique_request_ids']}")
        print(f"🏷️ Unique Request Tags: {analysis['unique_request_tags']}")
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
        
        print("📝 Análise de Prompts:")
        prompt_analysis = analysis["prompt_analysis"]
        print(f"  Tipos: {dict(prompt_analysis['types'])}")
        print(f"  Tamanho Médio Prompt: {prompt_analysis['avg_prompt_length']:.1f} chars")
        print(f"  Tamanho Médio Response: {prompt_analysis['avg_response_length']:.1f} chars")
        print(f"  Range Prompt: {prompt_analysis['prompt_length_range']}")
        print()
        
        print("🏷️ Análise de Vazamento de Tags:")
        tag_leak = analysis["tag_leak_analysis"]
        print(f"  Tags Vazadas: {tag_leak['total_leaked_tags']}")
        print(f"  Requests com Vazamento: {tag_leak['requests_with_leaks']}")
        print(f"  Taxa de Vazamento: {tag_leak['leak_rate']:.2%}")
        if tag_leak['leaked_tags_examples']:
            print(f"  Exemplos: {tag_leak['leaked_tags_examples']}")
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
            prompt_preview = log.get("prompt_preview", "unknown")[:50] + "..."
            print(f"  {i+1}. Bucket {'A' if model == 'gpt-4o' else 'B'} → {model}")
            print(f"     Prompt: {prompt_preview}")
            print(f"     Trace: {trace_id}")
        
        print()
        print("🎯 Conclusão:")
        if analysis["status"] == "PASS":
            print("✅ VALIDAÇÃO DE TRÁFEGO REAL PASSOU")
            print("✅ Bucket A sempre usa gpt-4o")
            print("✅ Bucket B sempre usa gpt-5.1")
            print("✅ Nenhum vazamento entre buckets")
            print("✅ Nenhum vazamento de tags entre requests")
            print("✅ Todos os IDs são únicos")
            print("✅ Prompts variados funcionam corretamente")
            print("✅ Pipeline seguro para tráfego real WhatsApp")
        else:
            print("❌ VALIDAÇÃO DE TRÁFEGO REAL FALHOU")
            print("❌ Problemas detectados no tráfego real")
        
        print("="*100)

async def main():
    """Main execution"""
    validator = RealTrafficConcurrencyValidator()
    
    print("📱 VALIDAÇÃO DE CONCORRÊNCIA - TRÁFEGO REAL WHATSAPP")
    print("🎯 Objetivo: 300 chamadas com prompts reais de suporte")
    print("📋 Validar isolamento completo e sem vazamento de contexto")
    print()
    
    try:
        # Executar teste
        analysis = await validator.run_real_traffic_test(num_calls=300)
        
        # Salvar relatório
        reports_dir = Path("validation_reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"REAL_TRAFFIC_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
