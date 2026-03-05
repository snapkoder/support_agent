"""
POLICY ENGINE - JOTA SUPPORT AGENT
Motor de decisões inteligentes simplificado e focado
Objetivo: Maximizar resolução automática com regras claras e objetivas
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class JotaPolicyEngine:
    """
    Policy Engine Simplificado e Focado
    Objetivo: Tomar decisões inteligentes com regras claras e objetivas
    Foco: Maximizar resolução automática, minimizar escalonamento desnecessário
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(f"{__name__}.JotaPolicyEngine")
        
        # Estatísticas simples
        self.stats = {
            'total_evaluations': 0,
            'escalations': 0,
            'auto_resolutions': 0
        }
    
    def evaluate_response(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avaliação única e simplificada da resposta do agente
        Retorna decisão clara sobre escalonamento, prioridade e ações
        """
        try:
            self.stats['total_evaluations'] += 1
            
            # Garantir que agent_response não seja None
            # Converter AgentDecision para dict se necessário
            if hasattr(agent_response, 'agent_type'):
                # É AgentDecision - extrair atributos
                agent_type = agent_response.agent_type
                confidence = agent_response.confidence
                response_preview = str(agent_response.response)[:100] + '...'
                needs_escalation = getattr(agent_response, 'needs_escalation', False)
            elif isinstance(agent_response, dict):
                # É dict - usar get
                agent_type = agent_response.get('agent_type', 'unknown')
                confidence = agent_response.get('confidence', 0.0) if agent_response.get('confidence') is not None else 0.0
                response_preview = str(agent_response.get('response', ''))[:100] + '...'
                needs_escalation = agent_response.get('needs_escalation', False)
            else:
                # Tipo desconhecido
                agent_type = 'unknown'
                confidence = 0.0
                response_preview = 'Unknown response type'
                needs_escalation = False
            
            decision = {
                'timestamp': datetime.now().isoformat(),
                'agent_type': agent_type,
                'confidence': confidence,
                'response_preview': response_preview,
                'should_escalate': False,
                'priority': 'medium',
                'actions': [],
                'reasoning': []
            }
            
            # 1. Verificar escalonamento (apenas casos essenciais)
            escalation_decision = self._check_escalation(agent_response, context)
            decision['should_escalate'] = escalation_decision['should_escalate']
            decision['reasoning'].append(f"Escalation: {escalation_decision['reason']}")
            
            if escalation_decision['should_escalate']:
                self.stats['escalations'] += 1
            else:
                self.stats['auto_resolutions'] += 1
            
            # 2. Determinar prioridade (lógica clara)
            priority = self._determine_priority(agent_response, context)
            decision['priority'] = priority
            decision['reasoning'].append(f"Priority: {priority}")
            
            # 3. Determinar ações (mínimo necessário)
            actions = self._get_actions(agent_response, context)
            decision['actions'] = actions
            decision['reasoning'].append(f"Actions: {len(actions)} actions identified")
            
            self.logger.info(f"Decision: {decision['agent_type']} -> {decision['priority']} (conf: {decision['confidence']:.2f})")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'agent_type': 'error',
                'confidence': 0.0,
                'should_escalate': True,
                'priority': 'critical',
                'actions': [{'type': 'escalate_to_human'}],
                'reasoning': [f'Error: {str(e)}']
            }
    
    def _check_escalation(self, agent_response, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica se deve escalar para humano
        Regra: Apenas casos críticos ou quando solicitado
        """
        # Extrair informações de forma compatível
        if hasattr(agent_response, 'agent_type'):
            agent_type = agent_response.agent_type
            confidence = agent_response.confidence
            needs_escalation = getattr(agent_response, 'needs_escalation', False)
        elif isinstance(agent_response, dict):
            agent_type = agent_response.get('agent_type', '')
            confidence = agent_response.get('confidence', 0.0)
            needs_escalation = agent_response.get('needs_escalation', False)
        else:
            agent_type = 'unknown'
            confidence = 0.0
            needs_escalation = False
            
        message = context.get('message', '').lower()
        
        # 1. Segurança sempre escala (não negociável)
        if agent_type == 'golpe_med':
            return {
                'should_escalate': True,
                'reason': 'Security case - always escalate for human review'
            }
        
        # 2. Cliente pediu escalonamento explicitamente
        if needs_escalation:
            return {
                'should_escalate': True,
                'reason': 'Agent requested escalation'
            }
        
        # 3. Confiança muito baixa (threshold ajustável)
        if confidence < self.confidence_threshold:
            return {
                'should_escalate': True,
                'reason': f'Low confidence: {confidence:.2f} < {self.confidence_threshold}'
            }
        
        # 4. Urgência explícita do cliente
        urgency_keywords = ['urgente', 'emergência', 'bloquear', 'invadiram', 'hackearam', 'roubo', 'perda', 'desesperado']
        if any(keyword in message for keyword in urgency_keywords):
            return {
                'should_escalate': True,
                'reason': f'Urgency detected: {[kw for kw in urgency_keywords if kw in message]}'
            }
        
        # 5. Problemas técnicos (apenas se complexos)
        technical_keywords = ['api', 'endpoint', 'rest', 'programação', 'código', 'desenvolvedor', 'sistema']
        if any(keyword in message for keyword in technical_keywords) and len(message) > 100:
            return {
                'should_escalate': True,
                'reason': 'Complex technical issue detected'
            }
        
        return {
            'should_escalate': False,
            'reason': 'No escalation criteria met'
        }
    
    def _determine_priority(self, agent_response, context: Dict[str, Any]) -> str:
        """
        Determina prioridade de forma clara e objetiva
        Regra: Baseada no tipo de agente e urgência
        """
        # Extrair informações de forma compatível
        if hasattr(agent_response, 'agent_type'):
            agent_type = agent_response.agent_type
        elif isinstance(agent_response, dict):
            agent_type = agent_response.get('agent_type', '')
        else:
            agent_type = 'unknown'
            
        message = context.get('message', '').lower()
        
        # 1. Prioridade baseada no agente
        if agent_type == 'golpe_med':
            return 'critical'
        elif agent_type in ['criacao_conta', 'open_finance']:
            return 'high'
        
        # 2. Urgência explícita
        urgency_keywords = ['urgente', 'emergência', 'bloquear', 'invadiram', 'hackearam', 'roubo', 'perda', 'desesperado']
        if any(keyword in message for keyword in urgency_keywords):
            return 'critical'
        
        # 3. Fraude ou segurança
        fraud_keywords = ['golpe', 'fraude', 'estornar', 'devolver', 'med', 'valor roubado', 'atividade suspeita']
        if any(keyword in message for keyword in fraud_keywords):
            return 'high'
        
        # 4. Elogios ou feedback positivo
        praise_keywords = ['maravilhoso', 'excelente', 'parabéns', 'perfeito', 'ótimo', 'bom']
        if any(keyword in message for keyword in praise_keywords):
            return 'low'
        
        # 5. Dúvidas simples
        simple_keywords = ['o que é', 'como funciona', 'ajuda', 'dúvida', 'informação']
        if any(keyword in message for keyword in simple_keywords) and len(message) < 50:
            return 'low'
        
        # 6. Padrão: média
        return 'medium'
    
    def _get_actions(self, agent_response, context: Dict[str, Any]) -> list:
        """
        Determina ações necessárias de forma minimalista
        Regra: Apenas ações essenciais para cada caso
        """
        actions = []
        
        # Extrair informações de forma compatível
        if hasattr(agent_response, 'agent_type'):
            agent_type = agent_response.agent_type
            needs_escalation = getattr(agent_response, 'needs_escalation', False)
            confidence = agent_response.confidence
        elif isinstance(agent_response, dict):
            agent_type = agent_response.get('agent_type', '')
            needs_escalation = agent_response.get('needs_escalation', False)
            confidence = agent_response.get('confidence', 0.0)
        else:
            agent_type = 'unknown'
            needs_escalation = False
            confidence = 0.0
        
        # 1. Ações de segurança (sempre)
        if agent_type == 'golpe_med':
            actions.append({
                'type': 'create_security_ticket',
                'priority': 'critical',
                'description': 'Create security ticket for human review'
            })
            actions.append({
                'type': 'flag_suspicious',
                'priority': 'high',
                'description': 'Flag as suspicious activity'
            })
        
        # 2. Ação de escalonamento (se necessário)
        if needs_escalation:
            actions.append({
                'type': 'escalate_to_human',
                'priority': 'high',
                'description': 'Escalate to human agent'
            })
        
        # 3. Ação de baixa confiança
        if confidence < self.confidence_threshold:
            actions.append({
                'type': 'low_confidence_flag',
                'priority': 'medium',
                'description': f'Low confidence: {confidence:.2f} < {self.confidence_threshold}'
            })
        
        return actions
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas simples do policy engine
        """
        return {
            'total_evaluations': self.stats['total_evaluations'],
            'escalations': self.stats['escalations'],
            'auto_resolutions': self.stats['auto_resolutions'],
            'escalation_rate': self.stats['escalations'] / max(1, self.stats['total_evaluations']),
            'auto_resolution_rate': self.stats['auto_resolutions'] / max(1, self.stats['total_evaluations']),
            'confidence_threshold': self.confidence_threshold
        }
    
    def reset_stats(self):
        """Reseta estatísticas"""
        self.stats = {
            'total_evaluations': 0,
            'escalations': 0,
            'auto_resolutions': 0
        }

# Instância global para uso no sistema
_policy_engine = None

def get_policy_engine(confidence_threshold: float = 0.3) -> JotaPolicyEngine:
    """
    Retorna instância do Policy Engine (singleton)
    """
    global _policy_engine
    
    if _policy_engine is None:
        _policy_engine = JotaPolicyEngine(confidence_threshold)
    
    return _policy_engine
