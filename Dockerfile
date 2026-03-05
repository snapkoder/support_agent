# JOTA SUPPORT AGENT - VERSÃO MODULAR COM VECTOR RAG
# Arquitetura modular focada nos 4 objetivos principais

FROM python:3.11-slim

# Configurações de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV JOTA_ENV=production
ENV JOTA_CONFIDENCE_THRESHOLD=0.7
ENV JOTA_USE_VECTOR_RAG=true

# Criar diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro (cache do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código modular
COPY . .

# Criar usuário não-root
RUN useradd -r -s /bin/false -d /app appuser

# Criar diretórios necessários e ajustar permissões
RUN mkdir -p logs data/chroma rag_index \
    && chown -R appuser:appuser /app

# Trocar para usuário não-root
USER appuser

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de execução
CMD ["python", "main.py"]
