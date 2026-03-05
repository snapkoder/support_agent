# Jota Support Agent (support_agent)

Serviço de atendimento automatizado com **orquestração multi-agente**, **RAG (Retrieval-Augmented Generation)** e **camadas de segurança/observabilidade**, disponibilizando:

- API HTTP (FastAPI) para processamento de mensagens
- CLI local para simular conversas (WhatsApp/Zendesk mock)
- Suíte de scripts/testes para validação de qualidade, determinismo e performance

## Visão geral

O `support_agent` processa mensagens de clientes e decide **qual agente especialista** deve responder (ex.: fraude/MED, Open Finance, criação de conta), podendo:

- **Delegar** para agentes especialistas
- **Acoplar contexto RAG** (base de conhecimento) à resposta
- **Escalar** para atendimento humano com base em sinais estruturados

Entry points principais:

- `main.py`: sobe o serviço HTTP (FastAPI + Uvicorn)
- `cli.py`: REPL local para testes manuais e cenários

## Requisitos

- Python 3.10+ (recomendado: 3.11)
- (Opcional) Ollama rodando localmente, caso utilize provider `ollama`
- Chave da OpenAI, caso utilize provider `openai`

Dependências Python estão em `requirements.txt`.

## Setup (ambiente local)

1) Crie e ative um virtualenv

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

2) Instale as dependências

```bash
pip install -r requirements.txt
```

3) Configure variáveis de ambiente

- Copie `./.env.example` para `./.env`
- Ajuste os valores conforme seu ambiente

```bash
# Exemplo (Windows)
copy .env.example .env
```

### Variáveis de ambiente (mínimo recomendado)

O arquivo `.env.example` contém uma configuração extensa (produção). Em dev, o mínimo costuma ser:

- `OPENAI_API_KEY` (se usar OpenAI)
- `LLM_PRIMARY_PROVIDER` (`openai` ou `ollama`)
- `OLLAMA_BASE_URL` e `OLLAMA_MODEL` (se usar Ollama)
- `OPENAI_EMBEDDINGS_MODEL` / `OPENAI_EMBEDDINGS_ENABLED` (se embeddings via OpenAI)

Observação: há flags e parâmetros adicionais (RAG, fallback, timeouts, rate limiting, etc.) documentados no próprio `.env.example`.

## Executando (modo API)

O modo API sobe um FastAPI com endpoints de mensagem, healthcheck e métricas.

### Subir o serviço

```bash
python main.py
```

Por padrão, lê `API_HOST` e `API_PORT` (fallback para `localhost:8000`).

### Endpoints

- `POST /agent/message`
- `GET /health`
- `GET /metrics`

Em ambiente **não-produção**, a documentação Swagger fica disponível em:

- `GET /docs`
- `GET /redoc`

### Exemplo de chamada

```bash
curl -X POST http://localhost:8000/agent/message \
  -H "Content-Type: application/json" \
  -d "{\"content\":\"Cai em um golpe no Pix\",\"user_id\":\"u1\",\"session_id\":\"s1\",\"context\":{},\"metadata\":{}}"
```

## Executando (modo CLI)

A CLI simula um canal de suporte sem integrações reais e é o caminho recomendado para:

- validação manual
- inspeção de roteamento/delegação
- checagem de determinismo

Rodar a CLI:

```bash
python cli.py
```

Rodar a CLI com logs:

```bash
python cli.py --debug
```

Rodar um cenário automatizado:

```bash
python cli.py --scenario golpe_med
```

Comandos úteis dentro do REPL:

- `/help`
- `/new` (nova sessão)
- `/reset` (reseta sessão mantendo ticket)
- `/scenario <name>` / `/scenarios`
- `/context` (exibe memória sanitizada)

## Arquitetura (alto nível)

- `core/agent_orchestrator.py`
  - Orquestra o fluxo: classificação/roteamento, delegação, RAG, geração de resposta, métricas/trace.

- `core/rag/`
  - Camada de RAG (serviço/ports/adapters) para indexação e retrieval.
  - Índice local em `rag_index/`.

- `core/memory/`
  - Memória de sessão e armazenamento simples (ex.: adaptadores in-memory/JSON).

- `agents/`
  - Agentes especialistas e agente geral.

- `core/jota_agent_service.py`
  - API FastAPI com middleware de segurança (rate limiting simples, headers, CORS).

## RAG e índice local

- O diretório `rag_index/` contém artefatos do índice (ex.: Chroma + metadata).
- Scripts em `scripts/` suportam rebuild/validações.

Se houver mudanças em embeddings/modelo que causem incompatibilidade de dimensão, verifique as flags relacionadas no `.env` (ex.: auto-rebuild em mismatch).

## Testes e validação

Há documentação específica em `tests/README.md`.

Execução típica (exemplos):

```bash
python -m pytest
```

Ou scripts diretos (conforme `tests/README.md`):

```bash
python tests/test_pipeline_integration.py
python tests/test_regression_suite.py
```

## Observabilidade e troubleshooting

- **Healthcheck**: `GET /health`
- **Métricas**: `GET /metrics`

Problemas comuns:

- Provider LLM indisponível (OpenAI sem chave / Ollama parado)
- Timeout por configuração agressiva de `*_TIMEOUT`
- RAG sem índice local válido (rebuild necessário)

## Segurança

- Não commite `.env` com segredos.
- Prefira rodar em produção com `JOTA_ENV=production` para desativar docs públicas.
- Revise CORS via `JOTA_CORS_ORIGINS`.

## Licença / uso

Repositório destinado ao case/projeto interno. Ajuste conforme o contexto de distribuição.
