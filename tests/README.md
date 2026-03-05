# 🧪 Testes do Jota Support Agent

Esta pasta contém os testes essenciais para validação do sistema Jota Support Agent.

## 📋 Estrutura dos Testes

### 🚀 Benchmark A/B
- **Arquivo**: `benchmark_ab.py`
- **Finalidade**: Comparar desempenho entre modelos LLM (gemma2:2b vs qwen2:1.5b)
- **Uso**: `python tests/benchmark_ab.py`
- **Saída**: `tests/benchmark_results.json`

### 🔍 Teste de Integração
- **Arquivo**: `test_pipeline_integration.py`
- **Finalidade**: Teste ponta-a-ponta do pipeline completo
- **Uso**: `python tests/test_pipeline_integration.py`
- **Saída**: Console

### 🧪 Suíte de Regressão
- **Arquivo**: `test_regression_suite.py`
- **Finalidade**: Validação contínua de funcionalidades críticas
- **Uso**: `python tests/test_regression_suite.py`
- **Saída**: `tests/regression_results.json`

### 📊 Baseline RAG
- **Arquivo**: `test_rag_precision_baseline.py`
- **Finalidade**: Medir precisão do sistema RAG
- **Uso**: `python tests/test_rag_precision_baseline.py`
- **Saída**: `tests/last_baseline.json`

### 📝 Golden Set
- **Arquivo**: `test_questions.md`
- **Finalidade**: Conjunto de perguntas de referência
- **Uso**: Referência para testes

## 🚀 Como Executar

### Executar Benchmark A/B
```bash
cd tests
python benchmark_ab.py
```

### Executar Teste de Integração
```bash
cd tests
python test_pipeline_integration.py
```

### Executar Suíte de Regressão
```bash
cd tests
python test_regression_suite.py
```

### Executar Baseline RAG
```bash
cd tests
python test_rag_precision_baseline.py
```

## 📊 Métricas Avaliadas

### Benchmark A/B
- Tempo médio por query
- Taxa de grounding (citasções presentes)
- Taxa de fallback
- Qualidade manual (0-5)
- Taxa de sucesso

### Teste de Integração
- Tempo total do pipeline
- Presença de Evidence Pack
- Status de grounding
- Confiança da resposta
- Agente correto

### Suíte de Regressão
- Presença de keywords esperadas
- Agente correto
- Tempo de resposta (< 30s)
- Confiança (> 0.5)
- Resposta não vazia
- Taxa de pass (mínimo 80%)

## 🔧 Configuração

Os testes usam as configurações definidas no ambiente:
- `OLLAMA_MODEL`: Modelo a ser testado
- `OLLAMA_TIMEOUT`: Timeout do LLM
- `OLLAMA_MAX_TOKENS`: Tokens máximos

## 📋 Histórico de Resultados

- `benchmark_results.json`: Últimos resultados do benchmark A/B
- `regression_results.json`: Últimos resultados da suíte de regressão
- `last_baseline.json`: Último baseline RAG

## 🚨 Critérios de Sucesso

- **Benchmark A/B**: Queries completas sem erros
- **Teste de Integração**: Pipeline completo funcional
- **Suíte de Regressão**: Taxa de pass ≥ 80%
- **Baseline RAG**: Métricas dentro dos limites esperados

## 🐛 Troubleshooting

### Erros Comuns
- **Timeout**: Verificar se Ollama está rodando
- **Import Error**: Verificar se PYTHONPATH está correto
- **Agent Error**: Verificar logs do orquestrador

### Logs Detalhados
- Ative logging com variáveis de ambiente:
  ```bash
  export LOG_LEVEL=DEBUG
  python tests/benchmark_ab.py
  ```

## 📝 Manutenção

### Adicionar Novos Testes
1. Criar arquivo em `tests/`
2. Seguir padrão de nomenclatura
3. Adicionar ao README.md
4. Atualizar suíte de regressão se necessário

### Atualizar Testes Existentes
1. Modificar arquivo correspondente
2. Testar localmente
3. Atualizar documentação
4. Commitar mudanças

## 📞 Contato

Para dúvidas ou problemas, verifique os logs do sistema ou contate a equipe de desenvolvimento.
