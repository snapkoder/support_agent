
# Jota Support Agent — Documento Técnico de Arquitetura (Oficial)

> Documento técnico completo do **estado atual** do repositório `support_agent/` (Python), cobrindo arquitetura, fluxo E2E, RAG ([core/rag/](cci:9://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag:0:0-0:0)), orquestração multi-agente, routing hierárquico L1/L2, cache, memória, CLI mock, testes, Docker e decisões arquiteturais.  
> **Escopo**: documentação fiel ao código presente. Onde houver discrepância com README/artefatos legados, isso é apontado explicitamente em **Limitações Conhecidas**.

---

## Table of Contents

1. [Visão Geral do Sistema](#1-visão-geral-do-sistema)  
2. [Arquitetura de Alto Nível](#2-arquitetura-de-alto-nível)  
3. [Estrutura de Diretórios](#3-estrutura-de-diretórios)  
4. [Core Layer](#4-core-layer)  
   4.1. [[core/agent_orchestrator.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:0:0-0:0)](#41-coreagent_orchestratorpy)  
   4.2. [Cache System](#42-cache-system)  
   4.3. [Memory](#43-memory)  
5. [RAG System ([core/rag/](cci:9://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag:0:0-0:0))](#5-rag-system-corerag)  
6. [Agents](#6-agents)  
7. [Hierarchical Evaluation](#7-hierarchical-evaluation)  
8. [CLI Mock](#8-cli-mock)  
9. [Testes](#9-testes)  
10. [Dockerização](#10-dockerização)  
11. [Segurança e Compliance](#11-segurança-e-compliance)  
12. [Decisões Arquiteturais e Justificativas](#12-decisões-arquiteturais-e-justificativas)  
13. [Limitações Conhecidas](#13-limitações-conhecidas)  
14. [Roadmap Técnico Futuro](#14-roadmap-técnico-futuro)  
15. [Confirmações Finais Obrigatórias](#15-confirmações-finais-obrigatórias)

---

## 1. Visão Geral do Sistema

### Objetivo do sistema
O **Jota Support Agent** é um sistema de atendimento automatizado multi-agente que:
- Recebe mensagens (via API/CLI mock).
- Determina roteamento determinístico e/ou por LLM.
- Recupera contexto (RAG) a partir de uma base de conhecimento local.
- Aciona um agente especializado (ou mantém atendimento geral).
- Produz uma decisão ([AgentDecision](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:367:0-390:78)) com resposta, confiança, sinalização de escalonamento e rastreabilidade mínima (`trace`).

### Escopo
- **Produção (API)**: [main.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/main.py:0:0-0:0) + [core/jota_agent_service.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/jota_agent_service.py:0:0-0:0) (FastAPI).
- **Orquestração principal**: [core/agent_orchestrator.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:0:0-0:0).
- **RAG novo (ports/adapters)**: [core/rag/](cci:9://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag:0:0-0:0).
- **Agentes especialistas**: `agents/*.py`.
- **Ferramentas de validação/teste**: [tests/](cci:9://file:///c:/Users/pedro/Desktop/case_jota/support_agent/tests:0:0-0:0), [smoke_test_final.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/smoke_test_final.py:0:0-0:0), [cli.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:0:0-0:0).

### Restrições arquiteturais (observadas no código)
- **Sem integração real** com WhatsApp/Zendesk no repo (existem stubs/mocks).
- **PII-safe**: existe redator de segredos/PII em [core/security.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/security.py:0:0-0:0) e sanitização na CLI ([cli.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:0:0-0:0)).
- **RAG local-first**: pipeline principal usa [InMemoryVectorStoreAdapter](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/adapters/inmemory_vector_store_adapter.py:14:0-120:19) com embeddings e cosine similarity.
- **Determinismo na orquestração**: existe routing determinístico por keywords antes de cair para LLM (reduz não-determinismo).
- **Sem fallback silencioso em pontos críticos**: há logging/indicadores explícitos de implementação e caminhos de erro no RAG.

### Premissas
- A base de conhecimento é mantida localmente em `support_agent/data/knowledge_base/`.
- O serviço pode operar com diferentes provedores de LLM (há infraestrutura para OpenAI/Ollama/fallback no [core/llm_manager.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/llm_manager.py:0:0-0:0)).
- O sistema visa suportar testes E2E e validações determinísticas no mesmo processo (cache e memória por processo).

---

## 2. Arquitetura de Alto Nível

### Diagrama textual do fluxo principal

```text
[Entrada]
  |
  |-- (API) FastAPI: POST /agent/message  -> core/jota_agent_service.py
  |-- (CLI) python cli.py                 -> cli.py
  |-- (Test) smoke_test_final.py          -> process_message_flow()
  |
[JotaAgentOrchestrator] core/agent_orchestrator.py
  |
  |-- Memory Load (MemoryOrchestrator) -> define new_session/existing_session
  |
  |-- Cache Check (decision/rag/classify)
  |
  |-- L1 Router (Fluxo): atendimento_geral como entrypoint + regras de fluxo
  |-- L2 Router (Determinístico): _classify_by_keywords() -> especialista ou fallthrough
  |-- (Fallback) Classificação por LLM (quando não há match determinístico)
  |
  |-- RAG Retrieve (core/rag via JotaRAGSystem shim)
  |
  |-- Execução do Agent (agents/*) + evidence_pack + geração LLM
  |
  |-- Grounding / Soft gates / Confidence reduction (sem “override” de escalonamento por LLM)
  |
  |-- Delegation (se aplicável) + Flow compliance
  |
[Saída]
  |
  |-- AgentDecision (response, confidence, rag_used, should_escalate, trace)
  |
  |-- (CLI) TicketStore mock cria/atualiza ticket em caso de escalonamento
```

### Entrada → Orchestrator → Routing → RAG → Agent → Decision → Escalation
- **Entrada**: [AgentMessage](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:356:0-365:35) com `content`, `user_id`, `session_id`, `context`.
- **Orchestrator**: conduz pipeline, coleta métricas e mantém invariantes de fluxo.
- **Routing**:
  - L1 (fluxo): entrada padrão em `atendimento_geral`, com possível delegação.
  - L2: classificação determinística por keywords ([_classify_by_keywords](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:1228:4-1343:27)).
  - Fallback: classificação via LLM quando não há match determinístico.
- **RAG**: [JotaRAGSystem.query](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:174:4-249:13) delega para [core/rag/RAGService.process_query](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:117:4-278:13).
- **Agent**: gera [AgentDecision](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:367:0-390:78), normalmente baseado em Evidence Pack + LLM.
- **Decision**: normalização, trace, métricas e guardrails (grounding/confidence gates).
- **Escalation**:
  - Especialistas implementam **override determinístico** ignorando `needs_escalation` vindo do JSON do LLM, escalando apenas com sinais estruturados (`issue_type`, `urgency`, etc.).
  - A CLI cria ticket mock baseado em `should_escalate`.

### Separação de responsabilidades (nível de módulos)
- **[core/agent_orchestrator.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:0:0-0:0)**: autoridade do fluxo, roteamento e integração.
- **`core/rag/*`**: RAG como caso de uso (use case), com portas/adapters.
- **`agents/*`**: lógica de conversação/nível de domínio e instruções para o LLM.
- **[core/llm_manager.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/llm_manager.py:0:0-0:0)**: resolução de modelo, chamada ao provedor, instrumentação.
- **[core/prompt_manager.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/prompt_manager.py:0:0-0:0)**: carregamento de prompts e criação do Evidence Pack.
- **`core/memory/*`**: memória simplificada + orquestração estruturada.
- **[core/cache_store.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:0:0-0:0)**: cache em memória do processo com TTL.

### Justificativa do design modular
- Permite:
  - Evoluir o RAG sem alterar orquestração (ports/adapters).
  - Controlar determinismo (routing e escalonamento com sinais estruturados).
  - Testabilidade: CLI mock e smoke tests exercitam o pipeline completo.

---

## 3. Estrutura de Diretórios

### Árvore simplificada (principal)
```text
support_agent/
  main.py
  cli.py
  smoke_test_final.py
  requirements.txt
  pytest.ini
  Dockerfile

  core/
    agent_orchestrator.py
    cache_store.py
    config.py
    jota_agent_service.py
    llm_manager.py
    policy_engine.py
    prompt_manager.py
    security.py
    memory/
      __init__.py
      simple_memory_adapter.py
      simple_session_memory.py
    rag/
      rag_service.py
      ports.py
      models.py
      rag_facade.py
      indexer.py
      embedding_signature.py
      adapters/
        inmemory_vector_store_adapter.py
        retriever_adapter.py
        knowledge_base_adapter.py
        local_embeddings_adapter.py
        openai_embeddings_adapter.py
        sqlite_vector_store_adapter.py
        chroma_vector_store_adapter.py
        external_kb_stub_adapter.py

  agents/
    base_agent.py
    atendimento_geral.py
    criacao_conta.py
    open_finance.py
    golpe_med.py

  data/
    knowledge_base/
      (arquivos .md, incluindo jota_kb_restructured.md)
    prompts/
      (artefatos/prompt sources; carregamento real depende de path)

  tests/
    test_mock_cli.py
    test_classification_robustness.py
    test_hardening_final.py
    test_hardening_comprehensive.py
    test_model_integrity.py
    test_model_error_no_fallback.py
    ...
```

### Produção vs. validação vs. CLI
- **Produção**:
  - [main.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/main.py:0:0-0:0) (entrypoint)
  - [core/jota_agent_service.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/jota_agent_service.py:0:0-0:0) (FastAPI)
  - [core/agent_orchestrator.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:0:0-0:0) (orquestração)
  - `core/rag/*` e `agents/*` (lógica)
- **Validação/Testes**:
  - `tests/*`
  - [smoke_test_final.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/smoke_test_final.py:0:0-0:0)
- **CLI Mock**:
  - [cli.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:0:0-0:0)

---

## 4. Core Layer

## 4.1 [core/agent_orchestrator.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:0:0-0:0)

### Responsabilidade central
É o **ponto de autoridade** do sistema:
- Define os *data contracts* (ex.: [AgentMessage](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:356:0-365:35), [AgentDecision](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:367:0-390:78), enums).
- Implementa fluxo end-to-end (especialmente [process_message_flow](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:642:4-745:13), usado em smoke test e CLI).
- Controla:
  - carregamento de memória,
  - roteamento L1/L2,
  - cache (decision/classify/rag),
  - execução de agentes,
  - compliance de fluxo,
  - instrumentação (trace, step timings),
  - sinais de delegação e escalonamento.

### Por que concentramos o contrato RAG aqui após remoção de `rag_system.py`
O arquivo contém:
- *Contracts* legados ([RAGQuery](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/ports.py:14:0-21:30), [RAGResult](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:68:0-75:34), [RAGDocument](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:46:0-55:22)) usados por agentes e Evidence Pack.
- O shim [JotaRAGSystem](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:78:0-302:38) que delega para `core/rag/RAGService`, preservando compatibilidade.

**Trade-off**:
- Prós: migração incremental e compatibilidade.
- Contras: arquivo grande e com múltiplas responsabilidades (documentado em Roadmap).

### Principais métodos (visão)
- **[get_agent_orchestrator()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:2880:0-2889:37)**: factory/singleton do orchestrator.
- **[process_message_flow(message)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:642:4-745:13)**: pipeline E2E (fluxo “enterprise”: entry agent, delegação, trace).
- **[process_message(message)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/base_agent.py:569:4-572:12)**: endpoint-oriented (usado pelo serviço FastAPI).
- **[_classify_by_keywords(text)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:1228:4-1343:27)**: roteamento determinístico (alta precisão).
- **[_classify_with_cache(message)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:1328:4-1432:76)**: classificação com cache (e fallback para LLM).
- **[_query_rag_optimized(message, agent_type)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:2522:4-2612:13)**: RAG com cache e top_k dinâmico.
- **[_execute_agent_specialist(message, rag_result, client_context, agent_type)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:1779:4-2192:54)**: invoca agentes.
- **[_verify_grounding_systematic(...)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:2290:4-2404:13)**: checagens de grounding e redução de confiança (soft gate).
- **Flow orchestration**:
  - mapeia entry agent, delegações e compliance.

### Fluxo [process_message_flow()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:642:4-745:13) (pontos importantes)
**1) Memory load**
- Carrega `memory_context` (via `core/memory`).
- Atualiza `message.context` com contexto estruturado.

**2) Cache check (multi-cache)**
- Verifica `decision_cache` e `rag_cache` antes da classificação.
- Registra em `decision.trace['cache_status']`.

**3) Routing e classificação**
- **Routing determinístico** via [_classify_by_keywords()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:1228:4-1343:27) tenta resolver casos de alta precisão (fraude, open finance, onboarding).
- Se não casar, cai para LLM/classificação.

**4) RAG retrieval**
- Executa RAG (sempre-on) via [JotaRAGSystem.query](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:174:4-249:13) → [core/rag/RAGService.process_query](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:117:4-278:13).
- Cria [RAGResult](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:68:0-75:34) compatível com agentes (documentos/chunks convertidos).

**5) Execução do agente**
- Executa o agente do domínio selecionado.
- Normaliza [AgentDecision](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:367:0-390:78), preserva `evidence_pack` e `trace`.

**6) Grounding / guardrails**
- Aplica verificação sistemática e, quando necessário, **reduz confiança** ao invés de “forçar escalonamento” por heurística textual.

### Hierarchical routing (L1 + L2)
- **L1 (fluxo)**: entrypoint é `atendimento_geral`; delegação para especialista quando há sinal.
- **L2 (classificação)**:
  - Determinística: [_classify_by_keywords()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:1228:4-1343:27) (regex/keywords/compounds).
  - Fallback: classificação por LLM (não detalhada aqui porque depende do prompt e do provider).

### [DelegationReason](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:401:0-406:39) enum
[DelegationReason](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:401:0-406:39) define razões canônicas, e o fluxo usa isso para:
- auditabilidade (`flow['delegation_reason']`)
- assertividade em testes e CLI

Exemplos (presentes no código):
- `OPEN_FINANCE_SIGNAL`
- `FRAUD_SIGNAL`
- `ONBOARDING_SIGNAL`
- `UNKNOWN_GENERAL`

### Flow compliance
O smoke test valida:
- Rodada 1: nova sessão deve entrar por `atendimento_geral`.
- Delegações são registradas em `flow`.

### Escalation logic (determinística)
- Especialistas (`criacao_conta`, `open_finance`, `golpe_med`) **não confiam** no campo `needs_escalation` do JSON do LLM.
- Escalonam somente por sinais estruturados:
  - `open_finance`: `issue_type == "ERRO"`
  - `criacao_conta`: `issue_type == "PROBLEMA"`
  - `golpe_med`: `urgency in ["URGENTE","EMERGENCIA"]` e/ou análise de risco adicional

O orchestrator **não** usa o LLM como autoridade para escalonamento via JSON; ele trata isso como não determinístico.

### Grounding
- Regras de citação `[C#]` são parte dos prompts e do Evidence Pack.
- O orchestrator mede cobertura de citação e grounding score e aplica *soft gates* (redução de confiança), mantendo o fluxo estável.

---

## 4.2 Cache System

### [core/cache_store.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:0:0-0:0) — CacheStore
**[CacheStore](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:11:0-128:40)** é um cache in-memory do processo, com:
- [get(key)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:22:4-40:36) → valida TTL, remove expirados, retorna apenas `value`.
- [set(key, value, ttl_seconds)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:42:4-52:13) → escreve com `cached_at`, TTL e lazy cleanup.
- [size()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:67:4-70:35), [clear()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:62:4-65:31), [delete()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:54:4-60:24).
- [_lazy_cleanup()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:94:4-128:40) (a cada intervalo) para expirar e limitar tamanho.

### Instâncias singleton por tipo
No módulo:
- [get_classify_cache()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:137:0-139:26) → `_classify_cache`
- [get_decision_cache()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:141:0-143:26) → `_decision_cache`
- [get_rag_cache()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/cache_store.py:145:0-147:21) → `_rag_cache`

**Decisão de design**: caches persistem pela vida do processo (bom para CLI e smoke test warm runs).

### Tipos de cache (no pipeline)
- **Classify cache**: evita recomputar classificação repetida para mensagens idênticas.
- **Decision cache**: reutiliza decisões quando apropriado (guardrails de confiança).
- **RAG cache**: cacheia [RAGResult](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:68:0-75:34) para reduzir retrieval repetido.

### TTL strategy (observado)
- `rag_cache`: TTL ~ `300s` (5 min) na orquestração.
- `classify_cache`: TTL lógico no wrapper (ex.: 3600s em algumas rotas internas).
- `decision_cache`: TTL variável calculado com heurística no orchestrator.

### Key normalization (ponto crítico)
**RAG cache** usa chave determinística baseada em:
- `normalize(query_text)` (lower + remove espaços duplicados)
- `kb_version` (hash do arquivo KB)
- `rag_signature` (assinatura do adapter de embeddings)

Isso evita:
- divergência GET/SET
- falsos misses por variação de whitespace/case
- hits errados quando KB/embeddings mudam

### Justificativa de design
- Cache triplo separa:
  - custo da classificação
  - custo do retrieval RAG
  - custo do pipeline completo (decisão)
- Evita “acoplar” hit rate de RAG ao agent_type, mantendo hit rate mais estável e observável.

---

## 4.3 Memory

### Onde vive
- `core/memory/*`
  - [simple_memory_adapter.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/memory/simple_memory_adapter.py:0:0-0:0) implementa:
    - [MemoryStore](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/memory/simple_memory_adapter.py:111:0-222:39) (persistência local via JSON no disco)
    - [MemorySnapshot](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/memory/simple_memory_adapter.py:38:0-90:9) (schema leve)
    - [MemoryOrchestrator](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/memory/simple_memory_adapter.py:325:0-443:24) (orquestração)
  - `simple_session_memory.py` (histórico básico por sessão/usuário)

### Session detection e stage control
- Orchestrator usa `memory_context` para inferir:
  - nova sessão vs. existente
  - estágio (`stage`) e `turn_count`
- A decisão “new session” é determinística pelo conteúdo do snapshot.

### Determinismo
- O snapshot tem `schema_version`, `turn_count`, `stage`, `entities` e `constraints`.
- Sanitização de entities/summary usa [redact_secrets](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/security.py:11:0-95:15).

### Por que não usamos memória persistente externa (no estado atual)
- Apesar de [requirements.txt](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/requirements.txt:0:0-0:0) listar `redis`, o caminho principal observado usa **store local** (`memory_store.json`) com locks por conversation_id.
- Benefício: simplicidade e determinismo local.
- Limitação: concorrência multi-processo e escalabilidade horizontal não são cobertas.

---

## 5. RAG System ([core/rag/](cci:9://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag:0:0-0:0))

### Visão geral
O RAG foi implementado em arquitetura **Ports & Adapters**, com [RAGService](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:20:0-461:66) como camada de caso de uso.

**Arquivos principais**:
- [core/rag/rag_service.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:0:0-0:0): orquestra o fluxo RAG.
- [core/rag/ports.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/ports.py:0:0-0:0): contratos (`Protocol`) e DTOs ([RAGQuery](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/ports.py:14:0-21:30), [RAGResult](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:68:0-75:34)).
- [core/rag/models.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/models.py:0:0-0:0): config, métricas e modelos auxiliares.
- [core/rag/rag_facade.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_facade.py:0:0-0:0): seleção de adapter, compatibilidade e metadata caching.
- `core/rag/adapters/*`: implementações concretas (embeddings, vector store, KB, retriever).

### RAGService
Responsabilidades:
- [process_query(query, agent_type, ...)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:117:4-278:13):
  - valida query (`validate_query`)
  - decide uso de RAG ([should_use_rag](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:55:4-115:20)) — com **RAG always on** quando habilitado
  - retrieval local-first ([_retrieve_local_first](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:332:4-395:27))
  - produz [RAGResult](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:68:0-75:34) com chunks, fontes, domínios e metadados
  - atualiza `RAGMetrics`

Observação: o [RAGService](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:20:0-461:66) mantém um `_query_cache` interno (quando `config.cache_enabled`), **além** do `rag_cache` do orchestrator. Na prática:
- `rag_cache` (orchestrator) é o cache “macro” do pipeline.
- `_query_cache` (RAGService) é um cache local do próprio serviço.

### Ports & Adapters
- **Ports**: [EmbeddingsPort](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/ports.py:52:0-83:11), [VectorStorePort](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/ports.py:85:0-131:11), [RetrieverPort](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/ports.py:133:0-158:11), [KnowledgeBasePort](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/ports.py:160:0-199:11).
- **Adapters**:
  - [RetrieverAdapter](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/adapters/retriever_adapter.py:18:0-183:24): combina embeddings + vector store; faz compatibility check (opcional).
  - [InMemoryVectorStoreAdapter](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/adapters/inmemory_vector_store_adapter.py:14:0-120:19): cosine similarity em memória (rápido, simples).
  - `KnowledgeBaseAdapter`: lê arquivos da KB.
  - `LocalEmbeddingsAdapter` / `OpenAIEmbeddingsAdapter`: geração de embeddings.
  - `SQLiteVectorStoreAdapter` / `ChromaVectorStoreAdapter`: opções alternativas.

### Vector store (in-memory adapter)
[InMemoryVectorStoreAdapter](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/adapters/inmemory_vector_store_adapter.py:14:0-120:19):
- armazena `chunks` + `embeddings`
- [similarity_search](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/ports.py:98:4-115:11) por cosine similarity
- `persist/load`: no-op (in-memory)

### Embeddings
- Controlados por adapters e pelo `RAGFacade` (seleção por config/env).
- Há assinatura estável (`EmbeddingSignature.stable_hash()`) para compatibilidade de índice.

### Retrieval pipeline
1. Embed da query
2. Similarity search top_k
3. Construção de `RetrievedChunk(chunk, score)`
4. Retorno como `RAGResult.chunks`

### Answerability gate e retry strategy
- Existem mecanismos de validação e logging de “empty context details” (ex.: `EmptyContextReason.RETRIEVAL_ZERO_HITS` via retriever).
- O comportamento do “answerability gate” é aplicado fortemente no nível de prompts/Evidence Pack e guardrails do orchestrator/agents (ex.: “não inventar; exigir citações”).

### Por que eliminamos `rag_system.py`
- `rag_system.py` foi substituído por:
  - `core/rag/*` como implementação real (ports/adapters)
  - [JotaRAGSystem](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:78:0-302:38) como shim/contrato compatível em [core/agent_orchestrator.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:0:0-0:0)

### Por que adotamos Ports & Adapters
- Troca de vector store (in-memory/SQLite/Chroma) sem mudar [RAGService](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag/rag_service.py:20:0-461:66).
- Troca de embeddings (local/OpenAI) sem acoplar o caso de uso.
- Testabilidade: ports podem ser mockados.

### Garantias de não fallback silencioso
- Logs explícitos de seleção de implementação (`impl=new_core_rag`).
- Em incompatibilidades ou erros, estrutura de retorno inclui `source="error"`/razões e logs estruturados em pontos críticos.

---

## 6. Agents

> Todos derivam de [agents/base_agent.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/base_agent.py:0:0-0:0) e usam:
> - [rag_system.query(RAGQuery)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:174:4-249:13) (contrato compatível)
> - [prompt_manager.create_evidence_pack(rag_result, query, top_k=...)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/prompt_manager.py:440:4-589:9)
> - [llm_manager.generate_response(prompt, agent_type=...)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/llm_manager.py:960:4-1264:63)
> - [_create_trace(...)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/base_agent.py:818:4-864:20) (método centralizado em [BaseAgent](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/base_agent.py:17:0-864:20))

### 6.1 Atendimento Geral — [agents/atendimento_geral.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/atendimento_geral.py:0:0-0:0)
**Responsabilidade funcional**
- Triagem, resposta geral e encaminhamento quando necessário.

**Quando é acionado**
- Entry agent do fluxo (L1). Mesmo quando delega, ele compõe o fluxo e razões.

**Lógica determinística (quando aplicável)**
- Possui “Fase 0” de **fatos explícitos** ([_check_explicit_facts](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/base_agent.py:235:4-322:19)) antes de geração.
- Modos:
  - “extractive” (perguntas objetivas) quando habilitado via env flags
  - “generative” (default) com Evidence Pack

**Uso de RAG**
- Sempre tenta obter contexto via [_get_rag_context](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/golpe_med.py:206:4-226:23).
- Usa Evidence Pack e citações.

**Limitações intencionais**
- Contém cache local `_response_cache` por mensagem+usuário (cache “micro”, por agente).

### 6.2 Criação de Conta — [agents/criacao_conta.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/criacao_conta.py:0:0-0:0)
**Responsabilidade funcional**
- Onboarding e problemas de cadastro/criação de conta.

**Quando é acionado**
- Delegação do atendimento geral quando L2 identifica onboarding.

**Regras de escalonamento**
- Ignora `needs_escalation` do LLM.
- Escala determinísticamente quando `issue_type == "PROBLEMA"`.

**Uso de RAG**
- Evidence Pack obrigatório, com regras de citar `[C#]`.

### 6.3 Open Finance — [agents/open_finance.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/open_finance.py:0:0-0:0)
**Responsabilidade funcional**
- Troubleshooting de conexão bancária e consentimento Open Finance.

**Quando é acionado**
- Delegação quando sinais fortes:
  - termos de conexão bancária
  - bancos + intenção
  - erros como `invalid_request_uri`, `err_unknown_url_scheme`

**Regras de escalonamento**
- Ignora `needs_escalation` do LLM.
- Escala determinísticamente quando `issue_type == "ERRO"`.

### 6.4 Golpe MED — [agents/golpe_med.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/golpe_med.py:0:0-0:0)
**Responsabilidade funcional**
- Segurança, fraude, MED e casos de alto risco.

**Quando é acionado**
- Sinais fortes de fraude/MED/BO etc.

**Regras de escalonamento**
- Ignora `needs_escalation` do LLM.
- Escala determinísticamente quando `urgency` ∈ {`URGENTE`, `EMERGENCIA`}.
- Possui análise adicional de risco ([_analyze_risk_level](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/golpe_med.py:228:4-344:72)) que pode recomendar `ESCALATE`.

**Limitações intencionais**
- Fallback conservador tende a escalonar, pois é domínio de segurança.

---

## 7. Hierarchical Evaluation

### L1 Router (general vs specialist)
- Entry agent do fluxo: `atendimento_geral`.
- O sistema mantém compliance do fluxo (validado no smoke test):
  - novas sessões entram por atendimento_geral
  - delegação é explícita e auditável

### L2 Specialist classifier
- [_classify_by_keywords(text)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:1228:4-1343:27) em [core/agent_orchestrator.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:0:0-0:0).
- Estratégia:
  - normaliza texto (lower, remove pontuação, remove acentos)
  - aplica regras de alta precisão por domínio:
    - `golpe_med`
    - `open_finance`
    - `criacao_conta`

### Strict vs tolerant scoring
- O código mistura:
  - regras determinísticas (strict)
  - fallback por LLM (tolerant)
- Soft gates de grounding reduzem confiança ao invés de bloquear duramente.

### Allowlist de overrides semânticos
- [BaseAgent._apply_overrides](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/agents/base_agent.py:48:4-149:27) implementa “Strong Evidence Override” com camadas (confidence/citação/alinhamento semântico).
- Objetivo: evitar “falso fallback” quando há evidência forte no Evidence Pack.

### Métricas obtidas (estado atual)
- [smoke_test_final.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/smoke_test_final.py:0:0-0:0) coleta:
  - `rag_cache_hit_rate`, `classify_cache_hit_rate`, latências por etapa
  - compliance do fluxo
- O repo também contém scripts/artefatos de validação sob `tests/validation_*`.

### Justificativa da abordagem hierárquica
- Reduz custo e variância do LLM.
- Aumenta determinismo (essencial para estabilidade operacional e testes reprodutíveis).
- Mantém precisão ao reservar LLM para casos ambíguos.

---

## 8. CLI Mock

### Objetivo
[cli.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:0:0-0:0) simula um canal de suporte (WhatsApp/Zendesk) **sem integrar APIs reais**, oferecendo:
- sessão ([CLISession](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:221:0-246:81)) com `user_id` e `session_id`
- ticket store local ([TicketStore](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:114:0-214:31)) com transcript e tags
- cenários multi-turn executados 2x (verificação de determinismo)

### Fluxo de sessão
- `/new`: nova sessão completa
- `/reset`: reseta sessão mas preserva ticket
- a cada mensagem, o CLI chama [orchestrator.process_message_flow(msg)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:642:4-745:13)

### TicketStore mock
- Criação de ticket “automática” quando `decision.should_escalate=True`
- Sanitização de PII nos transcripts ([_sanitize](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:79:0-85:15))

### Comandos
- `/scenario <name>`: executa cenário 2x
- `/ticket new|show|close`
- `/context`, `/debug on|off`

### Cenários multi-agente
Exemplos em `SCENARIOS`:
- `open_finance`: inclui `invalid_request_uri`
- `golpe_med`: inclui BO, MED e bloqueio
- `criacao_conta`
- `geral_especialista` (sequência geral → especialistas)

### Garantias de determinismo
- Cada cenário roda **duas vezes** e compara agent/escalation/rag flags (tabela final).

---

## 9. Testes

### Smoke test E2E
- [smoke_test_final.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/smoke_test_final.py:0:0-0:0)
- Executa 10 perguntas em 2 rodadas:
  - Rodada 1: cold start
  - Rodada 2: warm (mesmo processo)
- Coleta:
  - `rag_cache_hit_rate`, `classify_cache_hit_rate`
  - p50/p95 de latências gerais e por etapa

### Testes unitários da CLI
- [tests/test_mock_cli.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/tests/test_mock_cli.py:0:0-0:0)
- Cobertura:
  - lifecycle de sessão
  - TicketStore (create/close/events/transcript)
  - sanitização de CPF/CNPJ/UUID
  - estrutura de cenários

### Hardening / Model integrity
Arquivos relevantes:
- [tests/test_model_integrity.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/tests/test_model_integrity.py:0:0-0:0)
- [tests/test_model_error_no_fallback.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/tests/test_model_error_no_fallback.py:0:0-0:0)
- [tests/test_hardening_final.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/tests/test_hardening_final.py:0:0-0:0)
- [tests/test_hardening_comprehensive.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/tests/test_hardening_comprehensive.py:0:0-0:0)
- [tests/test_classification_robustness.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/tests/test_classification_robustness.py:0:0-0:0)

**Objetivo**: garantir consistência de modelo, ausência de fallback silencioso crítico e robustez do roteamento.

---

## 10. Dockerização

### Estrutura do Dockerfile
- Base: `python:3.11-slim`
- Instala deps do sistema: `gcc`, `g++`, `curl`
- Instala [requirements.txt](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/requirements.txt:0:0-0:0)
- Copia o projeto inteiro para `/app`
- Expõe porta `8000`
- `HEALTHCHECK` em `GET /health`
- CMD: `python main.py`

### Execução local (container)
- `docker build -t jota-support-agent .`
- `docker run -p 8000:8000 jota-support-agent`

### Estratégia de produção (implícita)
- Serviço FastAPI via uvicorn (in-process).
- Healthcheck é **bloqueante**: falha se provedor LLM primário não estiver saudável.

---

## 11. Segurança e Compliance

### PII-safe logging
- [core/security.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/security.py:0:0-0:0):
  - [redact_secrets(text)](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/security.py:11:0-95:15) mascara CPF/CNPJ, chaves OpenAI e tokens genéricos.
  - [SecureLogger](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/security.py:163:0-197:57) aplica redação automática também em `extra`.

### Sanitização no CLI
- [cli.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:0:0-0:0):
  - [_sanitize()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:79:0-85:15) remove padrões de CPF/CNPJ/UUID do texto exibido.
  - [_hash()](cci:1://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:88:0-90:62) mostra apenas hashes curtos de `session_id`/`user_id`.

### Fail-closed no RAG (comportamento observado)
- Em incompatibilidades e falhas, o pipeline retorna estruturas vazias com logs/razões; não “finge” sucesso silenciosamente.

---

## 12. Decisões Arquiteturais e Justificativas

### Por que removemos `rag_system.py`
- Centralizar implementação RAG em [core/rag/](cci:9://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/rag:0:0-0:0) com arquitetura ports/adapters.
- Manter compatibilidade por shim ([JotaRAGSystem](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/core/agent_orchestrator.py:78:0-302:38)) no orchestrator.

### Por que mantivemos contrato compatível
- Agentes e Evidence Pack trabalham com `RAGQuery/RAGResult` legados.
- Migração segura e incremental (reduz blast radius).

### Por que usamos cache triplo
- Separar custos e pontos de reuso:
  - classificação
  - retrieval RAG
  - decisão completa
- Melhor observabilidade e tuning de TTL.

### Por que routing determinístico antes do LLM
- Determinismo operacional.
- Reduz custo/latência e variação.
- Aumenta previsibilidade para testes.

### Por que avaliação hierárquica
- L1 controla o fluxo; L2 controla o domínio.
- Mantém compliance e auditabilidade.

### Por que CLI mock em vez de API real
- Permite testes offline e reproduzíveis.
- Exercita fluxo completo sem dependências externas.

### Por que temperatura controlada
- `LLMConfig.temperature` default ~0.2 em vários pontos.
- Menos criatividade, mais consistência e aderência à KB.

### Por que evitar arquivos desnecessários
- Mantém repo menor e reduz superfícies de manutenção.
- Evidente também na existência de smoke test único e CLI mock.

---

## 13. Limitações Conhecidas

### Divergências/legados de documentação (README)
O [README.md](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/README.md:0:0-0:0) contém referências e afirmações que **não batem 1:1** com o código atual, por exemplo:
- referências a módulos como `core.intelligent_agent` e `core.rag_complete` (não são parte do fluxo principal observado neste estado).
- linguagem de “production ready” e algumas métricas que não são garantias formais do código.

**Impacto**: a documentação oficial deve considerar **o código** como fonte de verdade (este documento faz isso).

### Dependência de LLM para geração
- Mesmo com roteamento determinístico e Evidence Pack, a geração final depende do provider.

### Não persistência real de tickets
- [TicketStore](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:114:0-214:31) é mock local em memória; não há integração com Zendesk real.

### Escalabilidade horizontal
- Cache e memória são principalmente “por processo”.
- Para múltiplas instâncias, seria necessário store externo consistente.

### Observabilidade estruturada
- Há logging estruturado em partes (cache events, model resolution, etc.), mas não há um pipeline único consolidado (ex.: Prometheus/OpenTelemetry).

---

## 14. Roadmap Técnico Futuro

### Refatoração do orchestrator (quando seguro)
- Separar responsabilidades (routing, caching, flow orchestration, grounding) em módulos menores.
- Reduzir tamanho do arquivo e risco de regressões.

### Persistência real de tickets
- Substituir [TicketStore](cci:2://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:114:0-214:31) por integração real (Zendesk/CRM).
- Garantir PII-safe e auditabilidade.

### Observabilidade
- Padronizar eventos e campos (cache, routing, rag, llm).
- Exportar métricas (Prometheus) e traces (OpenTelemetry) quando aplicável.

### RAG
- Persistência opcional de índice em store real (SQLite/Chroma) com controle de versão.
- Regras mais explícitas de “answerability” e “no hallucination” como gates formais.

### Split em serviços
- Separar API + orchestrator + RAG indexer (quando houver necessidade real de escala).

---

## 15. Confirmações Finais Obrigatórias

- **Documento reflete arquitetura real**: **Sim** — baseado em leitura direta de `core/*`, `core/rag/*`, `agents/*`, [cli.py](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/cli.py:0:0-0:0), `tests/*`, [Dockerfile](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/Dockerfile:0:0-0:0), [requirements.txt](cci:7://file:///c:/Users/pedro/Desktop/case_jota/support_agent/requirements.txt:0:0-0:0).
- **Nenhum prompt alterado**: **Confirmado** — apenas documentação produzida; nenhum patch aplicado.
- **Nenhuma KB alterada**: **Confirmado** — apenas documentação produzida; nenhum patch aplicado.
- **Nenhuma funcionalidade inventada**: **Confirmado** — onde há divergência (ex.: README vs. código), isso foi explicitamente apontado em *Limitações Conhecidas*.
- **Todas as decisões justificadas tecnicamente**: **Sim** — justificativas foram descritas com trade-offs e implicações operacionais.

---

## Status
- **Documento entregue em Markdown**, com sumário, diagramas textuais e descrição profunda por módulos.