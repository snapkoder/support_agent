#!/usr/bin/env python3
"""
JOTA MOCK CLI - WhatsApp + Zendesk Mock Interface
Simulates the support channel without real APIs.

Usage:
  python cli.py
  python cli.py --debug
  python cli.py --scenario golpe_med

Commands:
  /new                 -> nova sessao (novo session_id, zera estado)
  /reset               -> reseta sessao atual (nova session_id interna, mantém ticket)
  /ticket new          -> cria ticket mock
  /ticket show         -> exibe ticket atual
  /ticket close        -> fecha ticket como solved
  /context             -> exibe contexto de memoria (sanitizado)
  /debug on|off        -> toggle logs verbosos
  /scenario <name>     -> executa cenario multi-turn (2x para check determinismo)
  /scenarios           -> lista cenarios disponíveis
  /help                -> este menu
  /exit                -> sai
"""

import sys
import os
import asyncio
import hashlib
import uuid
import logging
import time
import re
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


# ============================================================================
# LOGGING
# ============================================================================

def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        root.addHandler(handler)
    else:
        for h in root.handlers:
            h.setLevel(level)
    # Silence noisy third-party loggers unless debug
    noise = [
        "httpx", "httpcore", "openai",
        "core.llm_manager", "core.agent_orchestrator",
        "core.rag", "agents", "core.memory",
    ]
    for name in noise:
        logging.getLogger(name).setLevel(logging.DEBUG if debug else logging.CRITICAL)


# ============================================================================
# PII SAFETY
# ============================================================================

_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b'), "[CPF]"),
    (re.compile(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b'), "[CNPJ]"),
    (re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.I), "[UUID]"),
]


def _sanitize(text: str) -> str:
    """Remove known PII patterns from displayed text."""
    if not text:
        return text
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _hash(value: str, length: int = 8) -> str:
    """Short display-safe hash of a sensitive string."""
    return hashlib.sha256(value.encode()).hexdigest()[:length]


# ============================================================================
# TICKET STORE (FASE 2)
# ============================================================================

_MAX_TRANSCRIPT = 20


@dataclass
class MockTicket:
    ticket_id: str
    status: str = "open"
    created_at: str = ""
    updated_at: str = ""
    last_agent: str = ""
    escalation_reason: Optional[str] = None
    transcript: List[Dict] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    session_hash: str = ""
    events: List[Dict] = field(default_factory=list)


class TicketStore:
    def __init__(self) -> None:
        self._tickets: Dict[str, MockTicket] = {}
        self._counter = 0

    def create(
        self,
        session_hash: str,
        tags: Optional[List[str]] = None,
        escalation_reason: Optional[str] = None,
    ) -> MockTicket:
        self._counter += 1
        tid = f"TKT-{self._counter:04d}"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t = MockTicket(
            ticket_id=tid,
            status="open",
            created_at=now,
            updated_at=now,
            session_hash=session_hash,
            tags=list(tags or []),
            escalation_reason=escalation_reason,
        )
        self._tickets[tid] = t
        return t

    def get(self, ticket_id: str) -> Optional[MockTicket]:
        return self._tickets.get(ticket_id)

    def close(self, ticket_id: str) -> bool:
        t = self._tickets.get(ticket_id)
        if t:
            t.status = "solved"
            t.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False

    def mark_reset(self, ticket_id: str) -> None:
        t = self._tickets.get(ticket_id)
        if t:
            t.tags.append("session_reset")
            t.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.add_event(ticket_id, {"type": "session_reset"})

    def add_event(self, ticket_id: str, event: Dict) -> None:
        t = self._tickets.get(ticket_id)
        if t:
            t.events.append({**event, "ts": datetime.now().strftime("%H:%M:%S")})
            t.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_turn(
        self, ticket_id: str, user_msg: str, agent_msg: str, agent_type: str
    ) -> None:
        t = self._tickets.get(ticket_id)
        if not t:
            return
        t.transcript.append({
            "ts": datetime.now().strftime("%H:%M:%S"),
            "user": _sanitize(user_msg[:120]),
            "agent": _sanitize(agent_msg[:120]),
            "agent_type": agent_type,
        })
        if len(t.transcript) > _MAX_TRANSCRIPT:
            t.transcript = t.transcript[-_MAX_TRANSCRIPT:]
        t.last_agent = agent_type
        t.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def show(self, ticket_id: Optional[str]) -> str:
        if not ticket_id:
            return "Nenhum ticket ativo. Use /ticket new para criar."
        t = self._tickets.get(ticket_id)
        if not t:
            return f"Ticket {ticket_id} nao encontrado."
        lines = [
            "=" * 60,
            f"TICKET : {t.ticket_id}   STATUS: {t.status.upper()}",
            f"Criado : {t.created_at}",
            f"Update : {t.updated_at}",
            f"Session: {t.session_hash}   Agente: {t.last_agent or '-'}",
            f"Tags   : {', '.join(t.tags) or 'none'}",
        ]
        if t.escalation_reason:
            lines.append(f"Escalation: {t.escalation_reason}")
        if t.events:
            lines.append("Eventos (ultimos 5):")
            for e in t.events[-5:]:
                etype = e.get("type", "")
                agt = e.get("agent_type", "")
                rid = e.get("request_id", "")
                ts = e.get("ts", "")
                lines.append(f"  [{ts}] {etype} agent={agt} req={rid}")
        if t.transcript:
            lines.append("Transcript (ultimas 3 interacoes):")
            for turn in t.transcript[-3:]:
                lines.append(
                    f"  [{turn['ts']}] [{turn['agent_type']}]"
                )
                lines.append(f"    U: {turn['user'][:70]}")
                lines.append(f"    A: {turn['agent'][:70]}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# SESSION STATE (FASE 3)
# ============================================================================

class CLISession:
    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.user_id: str = "cli_user_001"
        self.turn_count: int = 0
        self.current_ticket_id: Optional[str] = None

    @property
    def session_hash(self) -> str:
        return _hash(self.session_id)

    @property
    def user_hash(self) -> str:
        return _hash(self.user_id)

    def new_session(self) -> None:
        """Full new session: fresh session_id, clear everything."""
        self.session_id = str(uuid.uuid4())
        self.turn_count = 0
        self.current_ticket_id = None

    def reset_session(self) -> None:
        """Reset: new internal session_id (clears memory) — ticket kept, caller marks it."""
        self.session_id = str(uuid.uuid4())
        self.turn_count = 0
        # current_ticket_id intentionally preserved — caller handles mark_reset


# ============================================================================
# SCENARIOS (FASE 4)
# ============================================================================

SCENARIOS: Dict[str, Dict] = {
    "golpe_med": {
        "name": "CENARIO_1: Golpe + MED + Bloqueio",
        "description": "Vitima de golpe Pix, duvidas MED, pedido de bloqueio de conta",
        "messages": [
            "Cai em um golpe pelo Pix",
            "O BO e obrigatorio?",
            "Quanto tempo demora o MED?",
            "Preciso bloquear minha conta, invadiram",
        ],
        "expected_delegation": "golpe_med",
        "expect_escalate": True,
    },
    "open_finance": {
        "name": "CENARIO_2: Open Finance + erro + frustracao",
        "description": "Falha ao conectar banco Inter via Open Finance, frustracao",
        "messages": [
            "Nao consigo conectar meu banco Inter",
            "Erro invalid_request_uri",
            "Ja tentei tudo e nada funciona, estou muito cansado",
        ],
        "expected_delegation": "open_finance",
        "expect_escalate": True,
    },
    "criacao_conta": {
        "name": "CENARIO_3: Criacao de conta + camera + CPF",
        "description": "Abertura conta PJ, camera nao abre no cadastro, mudanca de CPF",
        "messages": [
            "Quero abrir uma conta pessoa juridica",
            "A camera nao abre durante o cadastro",
            "Quero usar outro CPF na minha conta",
        ],
        "expected_delegation": "criacao_conta",
        "expect_escalate": False,
    },
    "geral_especialista": {
        "name": "CENARIO_4: Geral -> Especialista(s)",
        "description": "Inicia geral, delega para criacao_conta, depois golpe_med",
        "messages": [
            "Como funciona o Jota?",
            "Quero abrir uma conta",
            "Nao consigo fazer meu cadastro",
            "Fui vitima de um golpe",
        ],
        "expected_delegation": ["criacao_conta", "golpe_med"],
        "expect_escalate": True,
    },
}


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

_W = 68
SEP = "=" * _W
SEP_THIN = "-" * _W


def _fmt_bool(v: bool) -> str:
    return "T" if v else "F"


def _safe_float(v: Any) -> float:
    try:
        return float(str(v).split("/")[0].strip())
    except (ValueError, TypeError):
        return 0.0


def format_response(
    turn: int,
    session_hash: str,
    user_msg: str,
    result: Dict,
    latency_ms: float,
) -> str:
    decision = result["decision"]
    flow = result["flow"]
    trace = getattr(decision, "trace", {}) or {}
    step_timings = trace.get("step_timings_ms", {})
    cache_status = trace.get("cache_status", {})

    # Routing line
    entry = flow.get("entry_agent", "?")
    delegated_to = flow.get("delegated_to")
    delegation_reason = flow.get("delegation_reason") or ""
    final_agent = decision.agent_type
    if delegated_to and delegated_to != entry:
        routing = f"{entry} -> {delegated_to}  [reason: {delegation_reason}]"
    else:
        routing = f"{final_agent}  [no delegation]"

    # RAG info from evidence_pack
    evidence = getattr(decision, "evidence_pack", {}) or {}
    rag_chunks = evidence.get("chunks_used", [])
    rag_scores = [_safe_float(s) for s in evidence.get("scores", [])]
    top_score = max(rag_scores) if rag_scores else 0.0
    rag_hits = len(rag_chunks)

    req_id = (trace.get("request_id") or "n/a")[:12]

    cl_ms = step_timings.get("classify_llm", 0)
    rl_ms = step_timings.get("rag_retrieve", 0)
    gl_ms = step_timings.get("response_generate_llm", 0)

    esc_reason = getattr(decision, "escalation_reason", "") or ""

    lines = [
        SEP,
        f"[TURN {turn}] session={session_hash}  req={req_id}  ts={datetime.now().strftime('%H:%M:%S')}",
        SEP_THIN,
        f"Voce  : {_sanitize(user_msg)}",
        SEP_THIN,
        f"AGENT   : {routing}",
        f"RAG     : used={_fmt_bool(decision.rag_used)}  hits={rag_hits}  top_score={top_score:.3f}",
        f"CACHE   : classify={_fmt_bool(cache_status.get('classify_cache_hit', False))}"
        f"  decision={_fmt_bool(cache_status.get('decision_cache_hit', False))}"
        f"  rag={_fmt_bool(cache_status.get('rag_cache_hit', False))}",
        f"ESCALATE: {_fmt_bool(decision.should_escalate)}"
        + (f"  reason={esc_reason}" if decision.should_escalate and esc_reason else ""),
        f"LATENCY : {latency_ms:.0f}ms"
        f"  [classify={cl_ms:.0f}ms  rag={rl_ms:.0f}ms  gen={gl_ms:.0f}ms]",
        SEP_THIN,
        f"Jota  : {_sanitize(decision.response)}",
        SEP,
    ]
    return "\n".join(lines)


def format_scenario_table(scen_name: str, runs: List[List[Dict]]) -> str:
    lines = [
        "",
        SEP,
        f"RESULTADO DO CENARIO: {scen_name}",
        SEP_THIN,
        f"{'#':<4} {'AGENT':<22} {'DELEGATED_TO':<18} {'RAG':<5} {'ESC':<5} {'LAT(ms)':<10}",
        SEP_THIN,
    ]
    run1 = runs[0]
    for i, row in enumerate(run1):
        agent = (row.get("agent_type") or "?")[:20]
        delg = (row.get("delegated_to") or "-")[:16]
        rag = _fmt_bool(row.get("rag_used", False))
        esc = _fmt_bool(row.get("should_escalate", False))
        lat = f"{row.get('latency_ms', 0):.0f}"
        msg = _sanitize((row.get("message") or "")[:30])
        lines.append(f"{i+1:<4} {agent:<22} {delg:<18} {rag:<5} {esc:<5} {lat:<10}  \"{msg}\"")

    # Determinism check
    if len(runs) == 2:
        run2 = runs[1]
        lines += ["", SEP_THIN, "DETERMINISMO (run1 vs run2):"]
        all_match = True
        for i, (r1, r2) in enumerate(zip(run1, run2)):
            agent_ok = r1.get("agent_type") == r2.get("agent_type")
            esc_ok = r1.get("should_escalate") == r2.get("should_escalate")
            rag_ok = r1.get("rag_used") == r2.get("rag_used")
            ok = agent_ok and esc_ok and rag_ok
            if not ok:
                all_match = False
            diffs = []
            if not agent_ok:
                diffs.append(f"agent:{r1.get('agent_type')}!={r2.get('agent_type')}")
            if not esc_ok:
                diffs.append(f"esc:{r1.get('should_escalate')}!={r2.get('should_escalate')}")
            if not rag_ok:
                diffs.append(f"rag:{r1.get('rag_used')}!={r2.get('rag_used')}")
            status = "OK   " if ok else "DIFF "
            detail = " | ".join(diffs) if diffs else "stable"
            lines.append(f"  Turn {i+1}: {status}  {detail}")
        verdict = "DETERMINISTICO" if all_match else "NAO-DETERMINISTICO"
        lines.append(f"  Veredicto: {verdict}")

    lines.append(SEP)
    return "\n".join(lines)


# ============================================================================
# MOCK CLI (FASE 1 + 3 + 4)
# ============================================================================

HELP_TEXT = """
Comandos disponiveis:
  /new                 -> nova sessao (novo session_id, zera estado)
  /reset               -> reseta sessao atual (nova session_id interna, ticket mantido)
  /ticket new          -> cria ticket mock
  /ticket show         -> exibe ticket atual
  /ticket close        -> fecha ticket (solved)
  /context             -> exibe contexto de memoria (sanitizado)
  /debug on|off        -> toggle logs verbosos no console
  /scenario <name>     -> executa cenario multi-turn (roda 2x, verifica determinismo)
  /scenarios           -> lista cenarios disponíveis
  /help                -> este menu
  /exit                -> sai
"""


class MockCLI:
    def __init__(self, debug: bool = False) -> None:
        self.orchestrator = None
        self.session = CLISession()
        self.ticket_store = TicketStore()
        self.debug = debug
        configure_logging(debug)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        print("Inicializando Jota Support Agent...")
        try:
            from support_agent.orchestrator.agent_orchestrator import get_agent_orchestrator
            self.orchestrator = await get_agent_orchestrator()
            print(f"Sistema pronto.  session={self.session.session_hash}  user={self.session.user_hash}")
            print("Digite /help para comandos ou escreva uma mensagem para comecar.\n")
        except Exception as exc:
            print(f"[ERRO] Falha ao inicializar orchestrator: {exc}")
            raise

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    async def _send(self, content: str) -> Tuple[Dict, float]:
        from support_agent.orchestrator.agent_orchestrator import AgentMessage, AgentPriority

        self.session.turn_count += 1
        msg = AgentMessage(
            content=content,
            user_id=self.session.user_id,
            session_id=self.session.session_id,
            timestamp=datetime.now(),
            context={},
            priority=AgentPriority.MEDIUM,
        )
        t0 = time.perf_counter()
        result = await self.orchestrator.process_message_flow(msg)
        latency_ms = (time.perf_counter() - t0) * 1000
        return result, latency_ms

    def _extract_row(self, result: Dict, latency_ms: float) -> Dict:
        decision = result["decision"]
        flow = result["flow"]
        trace = getattr(decision, "trace", {}) or {}
        step_timings = trace.get("step_timings_ms", {})
        return {
            "agent_type": decision.agent_type,
            "entry_agent": flow.get("entry_agent"),
            "delegated_to": flow.get("delegated_to"),
            "delegation_reason": flow.get("delegation_reason"),
            "rag_used": decision.rag_used,
            "should_escalate": decision.should_escalate,
            "escalation_reason": getattr(decision, "escalation_reason", "") or "",
            "confidence": decision.confidence,
            "latency_ms": latency_ms,
            "classify_ms": step_timings.get("classify_llm", 0),
            "rag_ms": step_timings.get("rag_retrieve", 0),
            "gen_ms": step_timings.get("response_generate_llm", 0),
        }

    # ------------------------------------------------------------------
    # Ticket auto-management
    # ------------------------------------------------------------------

    def _auto_ticket(self, result: Dict, user_msg: str) -> None:
        """Auto-create ticket on escalation; add event if ticket already exists."""
        decision = result["decision"]
        if not decision.should_escalate:
            return

        reason = getattr(decision, "escalation_reason", "") or ""
        agent = decision.agent_type
        trace = getattr(decision, "trace", {}) or {}
        req_id = (trace.get("request_id") or "")[:12]

        tags: List[str] = [agent]
        if agent == "golpe_med" or "fraud" in reason.lower():
            tags.append("fraud")
        if agent == "open_finance":
            tags.append("open_finance")
        if agent == "criacao_conta":
            tags.append("onboarding")
        if "frustration" in reason.lower() or "frustration" in agent.lower():
            tags.append("user_frustration")

        if not self.session.current_ticket_id:
            t = self.ticket_store.create(
                session_hash=self.session.session_hash,
                tags=tags,
                escalation_reason=reason,
            )
            self.session.current_ticket_id = t.ticket_id
            print(f"[TICKET] Auto-criado: {t.ticket_id}  escalation={reason}  agent={agent}")
        else:
            self.ticket_store.add_event(
                self.session.current_ticket_id,
                {
                    "type": "escalation",
                    "agent_type": agent,
                    "reason": reason,
                    "request_id": req_id,
                },
            )

    def _update_ticket_transcript(self, result: Dict, user_msg: str) -> None:
        if not self.session.current_ticket_id:
            return
        decision = result["decision"]
        self.ticket_store.add_turn(
            self.session.current_ticket_id,
            user_msg,
            decision.response,
            decision.agent_type,
        )

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def handle_message(self, content: str, silent: bool = False) -> Tuple[Dict, float]:
        result, latency_ms = await self._send(content)
        self._auto_ticket(result, content)
        self._update_ticket_transcript(result, content)
        if not silent:
            print(format_response(
                self.session.turn_count,
                self.session.session_hash,
                content,
                result,
                latency_ms,
            ))
        return result, latency_ms

    # ------------------------------------------------------------------
    # Command handler (FASE 3)
    # ------------------------------------------------------------------

    async def handle_command(self, line: str) -> bool:
        """Process a /command. Returns False to exit the loop."""
        parts = line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/exit":
            print("Saindo. Tchau!")
            return False

        elif cmd == "/help":
            print(HELP_TEXT)

        elif cmd == "/new":
            self.session.new_session()
            print(
                f"[NEW] Nova sessao. "
                f"session={self.session.session_hash}  turn=0"
            )

        elif cmd == "/reset":
            # Preserve ticket but mark as reset
            old_ticket = self.session.current_ticket_id
            if old_ticket:
                self.ticket_store.mark_reset(old_ticket)
            self.session.reset_session()
            print(
                f"[RESET] Sessao resetada. "
                f"session={self.session.session_hash}  (nova session_id interna)"
            )
            if old_ticket:
                print(f"[RESET] Ticket {old_ticket} mantido e marcado como session_reset.")
            print("[RESET] Proxima mensagem sera tratada como nova sessao pelo fluxo.")

        elif cmd == "/ticket":
            sub = args[0] if args else ""
            if sub == "new":
                t = self.ticket_store.create(
                    session_hash=self.session.session_hash,
                    tags=["manual"],
                )
                self.session.current_ticket_id = t.ticket_id
                print(f"[TICKET] Criado: {t.ticket_id}  status={t.status}")
            elif sub == "show":
                print(self.ticket_store.show(self.session.current_ticket_id))
            elif sub == "close":
                if self.session.current_ticket_id:
                    self.ticket_store.close(self.session.current_ticket_id)
                    print(f"[TICKET] {self.session.current_ticket_id} fechado (solved).")
                    self.session.current_ticket_id = None
                else:
                    print("[TICKET] Nenhum ticket ativo.")
            else:
                print("[TICKET] Uso: /ticket new|show|close")

        elif cmd == "/context":
            print(f"session_hash : {self.session.session_hash}")
            print(f"user_hash    : {self.session.user_hash}")
            print(f"turn_count   : {self.session.turn_count}")
            print(f"ticket       : {self.session.current_ticket_id or 'none'}")
            try:
                from support_agent.memory import get_memory_orchestrator
                mem = get_memory_orchestrator()
                ctx = mem.load_memory_context(self.session.session_id)
                if ctx:
                    summary = _sanitize(str(ctx)[:400])
                    print(f"memory_ctx   : {summary}")
                else:
                    print("memory_ctx   : (vazio)")
            except Exception as exc:
                print(f"memory_ctx   : (indisponivel: {exc})")

        elif cmd == "/debug":
            sub = args[0].lower() if args else ""
            if sub == "on":
                self.debug = True
                configure_logging(True)
                print("[DEBUG] Logs verbosos ativados.")
            elif sub == "off":
                self.debug = False
                configure_logging(False)
                print("[DEBUG] Logs verbosos desativados.")
            else:
                print("/debug on|off")

        elif cmd == "/scenarios":
            print("Cenarios disponíveis:")
            for k, v in SCENARIOS.items():
                print(f"  {k:<25}  {v['name']}")

        elif cmd == "/scenario":
            name = args[0] if args else ""
            if name not in SCENARIOS:
                avail = ", ".join(SCENARIOS.keys())
                print(f"Cenario '{name}' nao encontrado. Disponiveis: {avail}")
            else:
                await self.run_scenario(name)

        else:
            print(f"Comando desconhecido: '{cmd}'. Use /help.")

        return True

    # ------------------------------------------------------------------
    # Scenario runner (FASE 4)
    # ------------------------------------------------------------------

    async def run_scenario(self, name: str) -> List[List[Dict]]:
        """Run scenario twice and print determinism table."""
        scen = SCENARIOS[name]
        print(f"\n{SEP}")
        print(f"CENARIO: {scen['name']}")
        print(f"  {scen['description']}")
        print(f"  Mensagens: {len(scen['messages'])} turns")
        print(SEP)

        all_runs: List[List[Dict]] = []

        for run_idx in range(2):
            run_label = "cold" if run_idx == 0 else "warm-repeat"
            print(f"\n  --- Run {run_idx + 1}/2 ({run_label}) ---")
            # Fresh session for each run (determinism test)
            self.session.new_session()

            run_rows: List[Dict] = []
            for msg in scen["messages"]:
                # Print user message for run 1; silent for run 2
                verbose = run_idx == 0
                result, latency_ms = await self.handle_message(msg, silent=not verbose)
                row = self._extract_row(result, latency_ms)
                row["message"] = msg
                run_rows.append(row)
                await asyncio.sleep(0.05)

            all_runs.append(run_rows)

        print(format_scenario_table(scen["name"], all_runs))
        return all_runs

    # ------------------------------------------------------------------
    # Main REPL
    # ------------------------------------------------------------------

    async def run(self) -> None:
        print(SEP)
        print("JOTA MOCK CLI  -  WhatsApp + Zendesk Simulator")
        print(f"session={self.session.session_hash}  Digite /help para comandos")
        print(SEP)

        while True:
            try:
                line = input("\nVoce: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrompido. Saindo.")
                break

            if not line:
                continue

            if line.startswith("/"):
                try:
                    should_continue = await self.handle_command(line)
                except Exception as exc:
                    print(f"[ERRO] Comando falhou: {exc}")
                    logging.getLogger(__name__).exception("handle_command error")
                    should_continue = True
                if not should_continue:
                    break
            else:
                try:
                    await self.handle_message(line)
                except Exception as exc:
                    print(f"[ERRO] {exc}")
                    logging.getLogger(__name__).exception("handle_message error")


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main() -> int:
    # Ensure UTF-8 output on Windows (prevents UnicodeEncodeError from agent responses)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Jota Mock CLI - WhatsApp + Zendesk Simulator")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--scenario", metavar="NAME",
        help=f"Run a specific scenario and exit. Options: {list(SCENARIOS)}"
    )
    args = parser.parse_args()

    cli = MockCLI(debug=args.debug)
    await cli.setup()

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Cenario '{args.scenario}' nao encontrado. Disponiveis: {list(SCENARIOS)}")
            return 1
        await cli.run_scenario(args.scenario)
        return 0

    await cli.run()
    return 0


def async_main() -> int:
    """Entry point for setuptools console_scripts"""
    return asyncio.run(main())

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
