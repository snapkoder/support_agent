#!/usr/bin/env python3
"""
SCRIPT DE VALIDAÇÃO DE PRODUÇÃO REAL DO PIPELINE DE LLM
ETAPA 5: Script Automático de Auditoria

Objetivo: Executa 50 chamadas reais e detecta automaticamente:
- Modelos não autorizados
- Divergência trace vs request
- Bucket usando modelo incorreto

Script deve imprimir:
- total_calls
- models_detected
- mismatch_count
- status PASS/FAIL
"""

import os
import sys
import json
import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter

# Adicionar o projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from support_agent.llm.llm_manager import LLMManager, resolve_model

class ModelAuditor:
    def __init__(self):
        self.results = []
        self.setup_logging()
        
        # Modelos autorizados para produção
        self.authorized_models = {
            "gpt-4o",
            "gpt-5.1", 
            "gpt-3.5-turbo"
        }
        
        # Modelos esperados por bucket
        self.bucket_models = {
            "A": "gpt-4o",
            "B": "gpt-5.1"
        }
    
    def setup_logging(self):
        """Configurar logs DEBUG completos para capturar tudo"""
        # Criar logger com nível DEBUG
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        
        # Limpar handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler com DEBUG
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Formato detalhado para capturar tudo
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler para salvar logs completos
        log_file = project_root / "validation_logs" / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
        self.logger.info(f"🔍 AUDIT LOG FILE: {log_file}")
    
    def get_bucket_assignment(self, user_id: str) -> str:
        """Simula atribuição de bucket A/B baseada em user_id"""
        hash_value = hash(user_id) % 100
        return "A" if hash_value < 50 else "B"
    
    def get_model_for_bucket(self, bucket: str) -> str:
        """Retorna o modelo correspondente ao bucket"""
        return self.bucket_models[bucket]
    
    async def audit_single_call(self, call_id: int, scenario: str) -> dict:
        """Audita uma única chamada com diferentes cenários"""
        
        result = {
            "call_id": call_id,
            "scenario": scenario,
            "timestamp": datetime.now().isoformat(),
            "bucket": None,
            "expected_model": None,
            "model_selected": None,
            "model_sent_to_api": None,
            "model_used": None,
            "api_success": False,
            "issues": [],
            "authorized": False,
            "consistent": False,
            "bucket_correct": False
        }
        
        try:
            if scenario == "normal_gpt4o":
                # Cenário normal com gpt-4o
                os.environ["LLM_MODEL_PRIMARY"] = "gpt-4o"
                result["expected_model"] = "gpt-4o"
                
            elif scenario == "normal_gpt51":
                # Cenário normal com gpt-5.1
                os.environ["LLM_MODEL_PRIMARY"] = "gpt-5.1"
                result["expected_model"] = "gpt-5.1"
                
            elif scenario == "ab_bucket_a":
                # Cenário A/B bucket A
                user_id = f"audit_user_a_{call_id}"
                result["bucket"] = "A"
                result["expected_model"] = self.bucket_models["A"]
                
                context = {
                    "ab_testing_active": True,
                    "ab_model": result["expected_model"],
                    "user_id": user_id,
                    "bucket": "A"
                }
                
            elif scenario == "ab_bucket_b":
                # Cenário A/B bucket B
                user_id = f"audit_user_b_{call_id}"
                result["bucket"] = "B"
                result["expected_model"] = self.bucket_models["B"]
                
                context = {
                    "ab_testing_active": True,
                    "ab_model": result["expected_model"],
                    "user_id": user_id,
                    "bucket": "B"
                }
                
            elif scenario == "unauthorized_model":
                # Cenário com modelo não autorizado (deve falhar)
                os.environ["LLM_MODEL_PRIMARY"] = "gpt-6.0-unauthorized"
                result["expected_model"] = "gpt-6.0-unauthorized"
            
            # Executar chamada
            if scenario.startswith("ab_"):
                # A/B testing com contexto
                resolved_model = resolve_model(context)
                result["model_selected"] = resolved_model
                
                llm_manager = LLMManager()
                await llm_manager.initialize(context)
                
                prompt = f"Audit call {call_id} - {scenario}. Respond just with 'OK'."
                response = await llm_manager.generate_response(prompt)
                
            else:
                # Normal sem contexto
                resolved_model = resolve_model()
                result["model_selected"] = resolved_model
                
                llm_manager = LLMManager()
                await llm_manager.initialize()
                
                prompt = f"Audit call {call_id} - {scenario}. Respond just with 'OK'."
                response = await llm_manager.generate_response(prompt)
            
            # Capturar resultados
            result["model_sent_to_api"] = getattr(response, 'model_sent_to_api', None)
            result["model_used"] = getattr(response, 'model_used', None)
            result["api_success"] = True
            
            # Auditoria de problemas
            # 1. Verificar se modelo é autorizado
            if result["model_sent_to_api"] in self.authorized_models:
                result["authorized"] = True
            else:
                result["issues"].append(f"Unauthorized model: {result['model_sent_to_api']}")
            
            # 2. Verificar consistência
            if (result["model_selected"] == result["model_sent_to_api"] == result["model_used"]):
                result["consistent"] = True
            else:
                result["issues"].append(f"Inconsistency: selected={result['model_selected']}, sent={result['model_sent_to_api']}, used={result['model_used']}")
            
            # 3. Verificar bucket correto (se aplicável)
            if result["bucket"]:
                expected_bucket_model = self.bucket_models[result["bucket"]]
                if result["model_sent_to_api"] == expected_bucket_model:
                    result["bucket_correct"] = True
                else:
                    result["issues"].append(f"Bucket {result['bucket']} should use {expected_bucket_model}, got {result['model_sent_to_api']}")
            
            # 4. Verificar modelo esperado vs usado
            if result["expected_model"] and result["model_sent_to_api"] != result["expected_model"]:
                result["issues"].append(f"Expected {result['expected_model']}, got {result['model_sent_to_api']}")
            
            self.logger.info(f"✅ AUDIT {call_id} ({scenario}): model={result['model_sent_to_api']}, issues={len(result['issues'])}")
            
        except Exception as e:
            result["error"] = str(e)
            result["issues"].append(f"API call failed: {e}")
            self.logger.error(f"❌ AUDIT {call_id} ({scenario}) failed: {e}")
        
        self.results.append(result)
        return result
    
    async def run_audit(self, total_calls: int = 50):
        """Executa auditoria completa com N chamadas"""
        self.logger.info(f"🔍 INICIANDO AUDITORIA COM {total_calls} CHAMADAS")
        
        # Distribuir cenários
        scenarios = []
        
        # 40% chamadas normais gpt-4o
        normal_gpt4o_count = int(total_calls * 0.4)
        scenarios.extend(["normal_gpt4o"] * normal_gpt4o_count)
        
        # 40% chamadas normais gpt-5.1
        normal_gpt51_count = int(total_calls * 0.4)
        scenarios.extend(["normal_gpt51"] * normal_gpt51_count)
        
        # 10% chamadas A/B bucket A
        ab_a_count = int(total_calls * 0.1)
        scenarios.extend(["ab_bucket_a"] * ab_a_count)
        
        # 10% chamadas A/B bucket B
        ab_b_count = total_calls - len(scenarios)
        scenarios.extend(["ab_bucket_b"] * ab_b_count)
        
        # Embaralhar cenários
        random.shuffle(scenarios)
        
        self.logger.info(f"📊 Cenários: {normal_gpt4o_count} gpt-4o, {normal_gpt51_count} gpt-5.1, {ab_a_count} A/B A, {ab_b_count} A/B B")
        
        # Executar chamadas em batches para não sobrecarregar
        batch_size = 10
        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i:i+batch_size]
            batch_tasks = []
            
            for j, scenario in enumerate(batch):
                call_id = i + j + 1
                task = self.audit_single_call(call_id, scenario)
                batch_tasks.append(task)
            
            await asyncio.gather(*batch_tasks)
            self.logger.info(f"✅ Batch {i//batch_size + 1} concluído")
        
        # Análise final
        await self.analyze_results()
    
    async def analyze_results(self):
        """Analisa resultados da auditoria"""
        self.logger.info("📊 ANALISANDO RESULTADOS DA AUDITORIA")
        
        total_calls = len(self.results)
        successful_calls = sum(1 for r in self.results if r["api_success"])
        
        # Modelos detectados
        models_detected = Counter(r.get("model_sent_to_api", "unknown") for r in self.results if r.get("model_sent_to_api"))
        
        # Problemas
        unauthorized_models = [r for r in self.results if not r["authorized"]]
        inconsistent_calls = [r for r in self.results if not r["consistent"]]
        bucket_incorrect = [r for r in self.results if r["bucket"] and not r["bucket_correct"]]
        total_issues = sum(len(r["issues"]) for r in self.results)
        
        # Status final
        audit_passed = (
            len(unauthorized_models) == 0 and
            len(inconsistent_calls) == 0 and
            len(bucket_incorrect) == 0 and
            total_issues == 0
        )
        
        # Imprimir resultados
        print("\n" + "="*80)
        print("🔍 RELATÓRIO DE AUDITORIA AUTOMÁTICA")
        print("="*80)
        print(f"total_calls: {total_calls}")
        print(f"successful_calls: {successful_calls}")
        print(f"models_detected: {dict(models_detected)}")
        print(f"unauthorized_models: {len(unauthorized_models)}")
        print(f"inconsistent_calls: {len(inconsistent_calls)}")
        print(f"bucket_incorrect: {len(bucket_incorrect)}")
        print(f"total_issues: {total_issues}")
        print(f"status: {'PASS' if audit_passed else 'FAIL'}")
        print("="*80)
        
        # Detalhar problemas se houver
        if unauthorized_models:
            print(f"❌ Modelos não autorizados detectados:")
            for r in unauthorized_models:
                print(f"   - Call {r['call_id']}: {r.get('model_sent_to_api', 'unknown')}")
        
        if inconsistent_calls:
            print(f"❌ Chamadas inconsistentes:")
            for r in inconsistent_calls:
                print(f"   - Call {r['call_id']}: {r['issues']}")
        
        if bucket_incorrect:
            print(f"❌ Buckets incorretos:")
            for r in bucket_incorrect:
                print(f"   - Call {r['call_id']}: Bucket {r['bucket']} usou {r.get('model_sent_to_api', 'unknown')}")
        
        # Gerar relatório JSON
        await self.generate_audit_report(audit_passed, {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "models_detected": dict(models_detected),
            "unauthorized_models": len(unauthorized_models),
            "inconsistent_calls": len(inconsistent_calls),
            "bucket_incorrect": len(bucket_incorrect),
            "total_issues": total_issues,
            "status": "PASS" if audit_passed else "FAIL"
        })
        
        return audit_passed
    
    async def generate_audit_report(self, audit_passed: bool, summary: dict):
        """Gera relatório JSON da auditoria"""
        report_file = project_root / "validation_reports" / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        report = {
            "etapa": "ETAPA 5 - Script Automático de Auditoria",
            "timestamp": datetime.now().isoformat(),
            "audit_config": {
                "authorized_models": list(self.authorized_models),
                "bucket_models": self.bucket_models,
                "total_calls": len(self.results)
            },
            "summary": summary,
            "status": "PASS" if audit_passed else "FAIL",
            "detailed_results": self.results,
            "issues_found": [
                {
                    "call_id": r["call_id"],
                    "scenario": r["scenario"],
                    "issues": r["issues"]
                }
                for r in self.results if r["issues"]
            ]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 AUDIT REPORT: {report_file}")
        print(f"📋 Relatório detalhado: {report_file}")

async def main():
    """Main execution"""
    auditor = ModelAuditor()
    
    print("🔍 INICIANDO AUDITORIA AUTOMÁTICA DO PIPELINE LLM")
    print("🎯 Objetivo: Detectar problemas em 50 chamadas reais")
    print("⚠️  ATENÇÃO: Esta auditoria fará 50 chamadas REAIS à API OpenAI")
    print("📊 Verificaremos modelos autorizados, consistência e buckets")
    print()
    
    try:
        audit_passed = await auditor.run_audit(50)
        
        if audit_passed:
            print("✅ AUDITORIA CONCLUÍDA COM SUCESSO - Nenhum problema detectado")
            return 0
        else:
            print("❌ AUDITORIA FALHOU - Problemas detectados no pipeline")
            return 1
            
    except Exception as e:
        print(f"❌ ERRO CRÍTICO NA AUDITORIA: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
