from __future__ import annotations

import os
import time as _time
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import CustomerMessage, AgentResponse, WebhookPayload


_SAFE_ERROR = "Erro interno. Tente novamente ou contate o suporte."


class _RateLimiter:
    def __init__(self, rate: float = 10.0, burst: int = 20):
        self._rate = rate
        self._burst = burst
        self._buckets: Dict[str, list] = {}

    def allow(self, ip: str) -> bool:
        now = _time.monotonic()
        if ip not in self._buckets:
            self._buckets[ip] = [self._burst - 1, now]
            return True
        tokens, last = self._buckets[ip]
        elapsed = now - last
        tokens = min(self._burst, tokens + elapsed * self._rate)
        self._buckets[ip][1] = now
        if tokens >= 1:
            self._buckets[ip][0] = tokens - 1
            return True
        return False


_rate_limiter = _RateLimiter()


def register_routes(app: FastAPI, service: Any) -> None:
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.allow(client_ip):
            return JSONResponse(status_code=429, content={"detail": "Too many requests"})

        start_time = datetime.now()
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"

        service.logger.info(
            f"Request: {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
        )
        return response

    @app.post("/agent/message", response_model=AgentResponse)
    async def process_message(message: CustomerMessage) -> AgentResponse:
        try:
            from support_agent.orchestrator.agent_orchestrator import AgentMessage
            from support_agent.security.output_filter import filter_llm_output

            agent_message = AgentMessage(
                content=message.content,
                user_id=message.user_id,
                session_id=message.session_id,
                timestamp=datetime.now(),
                context=message.context,
                metadata=message.metadata or {},
            )

            decision = await service.orchestrator.process_message(agent_message)

            service._update_service_metrics(True, decision.processing_time)

            safe_response = filter_llm_output(decision.response)
            return AgentResponse(
                response=safe_response,
                agent_type=decision.agent_type,
                confidence=decision.confidence,
                processing_time=decision.processing_time,
                rag_used=decision.rag_used,
                should_escalate=decision.should_escalate,
                escalation_reason=decision.escalation_reason,
                reasoning=decision.reasoning,
            )

        except Exception as e:
            service.logger.error(f"Error processing message: {type(e).__name__}")
            service._update_service_metrics(False, 0.0)
            raise HTTPException(status_code=500, detail=_SAFE_ERROR)

    @app.post("/webhook/whatsapp")
    async def whatsapp_webhook(payload: WebhookPayload):
        try:
            service.logger.info(f"WhatsApp webhook received: {payload.object}")
            return {"status": "received"}
        except Exception as e:
            service.logger.error(f"Error processing WhatsApp webhook: {type(e).__name__}")
            raise HTTPException(status_code=500, detail=_SAFE_ERROR)

    @app.get("/health")
    async def health_check():
        try:
            if not service.orchestrator:
                raise HTTPException(status_code=503, detail="Service unavailable")

            orchestrator_metrics = await service.orchestrator.get_metrics()
            llm_metrics = orchestrator_metrics.get("llm_metrics", {})
            providers = llm_metrics.get("providers", {})
            primary_provider = llm_metrics.get("primary_provider")

            if not primary_provider:
                raise HTTPException(status_code=503, detail="Service unavailable")

            primary_healthy = providers.get(primary_provider, {}).get("healthy", False)
            if not primary_healthy:
                raise HTTPException(status_code=503, detail="Service degraded")

            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail="Service unavailable")

    @app.get("/metrics")
    async def get_metrics():
        try:
            orchestrator_metrics = await service.orchestrator.get_metrics() if service.orchestrator else {}
            return {
                "service": service.service_metrics,
                "orchestrator": orchestrator_metrics,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            service.logger.error(f"Error getting metrics: {type(e).__name__}")
            raise HTTPException(status_code=500, detail=_SAFE_ERROR)
