#!/usr/bin/env python3
"""
Gerenciador de Prompts dos Agentes
Carrega prompts dos arquivos MD e fornece exemplos para few-shot learning
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from support_agent.config.settings import _get_env_fallback

logger = logging.getLogger(__name__)

@dataclass
class PromptExample:
    """Exemplo de interação para few-shot learning"""
    user_input: str
    agent_response: str
    context: Optional[str] = None

@dataclass
class AgentPrompt:
    """Prompt completo do agente com exemplos"""
    name: str
    personality: str
    description: str
    responsibilities: List[str]
    system_prompt: str
    examples: List[PromptExample]
    keywords: Dict[str, List[str]]
    standard_responses: Dict[str, str]

class PromptManager:
    """Gerenciador de prompts dos agentes"""
    
    def __init__(self, prompts_dir: str = None):
        # 🆕 CORREÇÃO: Resolver caminho relativo ao arquivo do módulo
        if prompts_dir is None:
            # Try assets/prompts first (canonical location), fallback to data/prompts then ../prompts
            assets_prompts = os.path.join(
                os.path.dirname(__file__),
                "..", "assets", "prompts"
            )
            data_prompts = os.path.join(
                os.path.dirname(__file__),
                "..", "data", "prompts"
            )
            legacy_prompts = os.path.join(
                os.path.dirname(__file__),
                "..", "prompts"
            )
            if os.path.exists(assets_prompts):
                prompts_dir = assets_prompts
            elif os.path.exists(data_prompts):
                prompts_dir = data_prompts
            else:
                prompts_dir = legacy_prompts
        
        self.prompts_dir = prompts_dir
        self.logger = logging.getLogger(f"{__name__}.PromptManager")
        self._prompts: Dict[str, AgentPrompt] = {}
        
        # 🆕 CORREÇÃO: Verificar se diretório existe antes de carregar
        if os.path.exists(self.prompts_dir):
            self._load_all_prompts()
        else:
            self.logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            # Criar prompts padrão
            self._create_default_prompts()
    
    def _load_all_prompts(self):
        """Carrega todos os prompts do diretório"""
        try:
            for filename in os.listdir(self.prompts_dir):
                if filename.endswith(".md"):
                    # 🆕 CORREÇÃO: Remover prefixo agent_ e aceitar qualquer arquivo .md
                    agent_name = filename.replace(".md", "")
                    self._prompts[agent_name] = self._load_prompt_file(agent_name)
            
            self.logger.info(f"Loaded {len(self._prompts)} agent prompts")
            
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")
    
    def _create_default_prompts(self):
        """Cria prompts padrão quando diretório não existe"""
        self.logger.info("Creating default prompts")
        
        # Prompt padrão para atendimento_geral
        default_prompt = AgentPrompt(
            name="atendimento_geral",
            personality="Você é um assistente de atendimento do Jota. Seja útil, educado e prestativo.",
            description="Agente de atendimento geral",
            responsibilities=["Responder perguntas gerais", "Auxiliar clientes"],
            system_prompt="Você é um assistente de atendimento do Jota. Seja útil e educado.",
            examples=[],
            keywords={},
            standard_responses={}
        )
        
        self._prompts["atendimento_geral"] = default_prompt
        self.logger.info(f"Created {len(self._prompts)} default prompts")
    
    def _load_prompt_file(self, agent_name: str) -> AgentPrompt:
        """Carrega prompt de um arquivo específico"""
        # 🆕 CORREÇÃO: Remover prefixo agent_ do nome do arquivo
        file_path = os.path.join(self.prompts_dir, f"{agent_name}.md")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extrair seções
        description = self._extract_section(content, "Descrição")
        personality = self._extract_personality(content)
        responsibilities = self._extract_responsibilities(content)
        system_prompt = self._build_system_prompt(content, agent_name)
        examples = self._extract_examples(content)
        keywords = self._extract_keywords(content)
        standard_responses = self._extract_standard_responses(content)
        
        return AgentPrompt(
            name=agent_name,
            personality=personality,
            description=description,
            responsibilities=responsibilities,
            system_prompt=system_prompt,
            examples=examples,
            keywords=keywords,
            standard_responses=standard_responses
        )
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extrai uma seção do conteúdo"""
        pattern = rf"## {section_name}\n\n(.*?)(?=\n## |\n\n# |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_personality(self, content: str) -> str:
        """Extrai informações de personalidade"""
        section = self._extract_section(content, "Personalidade")
        lines = section.split('\n')
        personality_parts = []
        
        for line in lines:
            if line.startswith('- **Nome:**'):
                personality_parts.append(f"Nome: {line.split('**Nome:**')[1].strip()}")
            elif line.startswith('- **Tom:**'):
                personality_parts.append(f"Tom: {line.split('**Tom:**')[1].strip()}")
            elif line.startswith('- **Estilo:**'):
                personality_parts.append(f"Estilo: {line.split('**Estilo:**')[1].strip()}")
            elif line.startswith('- **Foco:**'):
                personality_parts.append(f"Foco: {line.split('**Foco:**')[1].strip()}")
        
        return "\n".join(personality_parts)
    
    def _extract_responsibilities(self, content: str) -> List[str]:
        """Extrai lista de responsabilidades"""
        section = self._extract_section(content, "Responsabilidades Principais")
        lines = section.split('\n')
        responsibilities = []
        
        for line in lines:
            if line.strip().startswith(tuple(str(i) for i in range(1, 10))):
                # Remove número e ponto
                resp = re.sub(r'^\d+\.\s*', '', line.strip())
                if resp:
                    responsibilities.append(resp)
        
        return responsibilities
    
    def _build_system_prompt(self, content: str, agent_name: str) -> str:
        """Constrói o system prompt completo com regras de citação e answerability gate"""
        description = self._extract_section(content, "Descrição")
        personality = self._extract_personality(content)
        responsibilities = self._extract_responsibilities(content)
        
        # Adicionar regras específicas se existirem
        rules_section = ""
        if "## Regras de Precedência" in content:
            rules_section = f"\n\n{self._extract_section(content, 'Regras de Precedência')}"
        elif "## Etapas Progressivas" in content:
            rules_section = f"\n\n{self._extract_section(content, 'Etapas Progressivas')}"
        
        # Construir prompt com regras de qualidade e ancoragem
        prompt = f"""REGRA OBRIGATÓRIA DE RESPOSTA:

* Cada afirmação factual DEVE conter citação no formato [C#].
* Respostas sem citações são inválidas.
* Se não houver informação suficiente no contexto para citar, responda explicitamente que não encontrou evidência suficiente.
* NÃO utilize conhecimento externo.
* NÃO gere resposta sem pelo menos uma citação válida.

Você é {agent_name.replace('_', ' ').title()} do Jota.

{description}

{personality}

RESPONSABILIDADES PRINCIPAIS:
{chr(10).join(f"- {resp}" for resp in responsibilities)}{rules_section}

REGRAS DE QUALIDADE OBRIGATÓRIAS:

1) ANCORAGEM EM FATOS - OBRIGATÓRIO:
- Use apenas informações da base oficial do Jota
- CADA afirmação factual DEVE conter citação no formato [C#] onde # é o número da seção
- Exemplo: "O Jota oferece rendimento de 100% do CDI [C27]"
- NENHUMA informação factual pode ser apresentada sem citação [C#]
- Se não houver citação disponível, diga explicitamente: "Não encontrei informações específicas sobre isso"

2) USO OBRIGATÓRIO DO EVIDENCE PACK:
- Se o EVIDENCE PACK contém informações, USE-AS para responder
- NUNCA diga "não encontrei" se houver trechos relevantes no EVIDENCE PACK
- Se a informação não estiver no EVIDENCE PACK, diga: "Não encontrei informações específicas sobre isso na base do Jota"
- NUNCA invente informações ou generalize sem suporte no EVIDENCE PACK

3) ESTRUTURA HIERÁRQUICA:
- Organize respostas usando títulos e subtítulos claros
- Use listas numeradas para passos sequenciais
- Use bullet points para informações não sequenciais

4) PRECISÃO TERMINOLÓGICA:
- Use termos exatos da base de conhecimento (ex: "pagamento de pix", não "fazer pix")
- Mantenha consistência com a linguagem oficial do Jota

5) RECUSA CONTROLADA:
- Para perguntas fora do escopo, explique educadamente o que o Jota oferece
- Ofereça alternativas dentro do suporte disponível

Responda sempre em português brasileiro de forma clara, profissional e 100% baseada na base oficial do Jota."""
        
        return prompt
    
    def _extract_examples(self, content: str) -> List[PromptExample]:
        """Extrai exemplos de interação"""
        examples = []
        
        # Encontrar todos os exemplos - formato atualizado com Agent: ou nome específico
        example_pattern = r"#### Exemplo \d+: (.+?)\n\nCliente: (.+?)\n\n(?:Agent|Aline): (.+?)(?=\n\n####|\n\n###|\Z)"
        matches = re.findall(example_pattern, content, re.DOTALL)
        
        for title, user_input, agent_response in matches:
            examples.append(PromptExample(
                user_input=user_input.strip(),
                agent_response=agent_response.strip(),
                context=title.strip()
            ))
        
        return examples
    
    def _extract_keywords(self, content: str) -> Dict[str, List[str]]:
        """Extrai palavras-chave por categoria"""
        keywords = {}
        
        # Encontrar seção de palavras-chave
        section_pattern = r"## Palavras-Chave para Detecção\n\n(.*?)(?=\n## |\n\n# |\Z)"
        section_match = re.search(section_pattern, content, re.DOTALL)
        
        if section_match:
            section = section_match.group(1)
            
            # Extrair categorias
            category_pattern = r"\*\*([^*]+):\*\*\n- (.+?)(?=\n\*\*|\n##|\n\n#|\Z)"
            category_matches = re.findall(category_pattern, section, re.DOTALL)
            
            for category, keyword_list in category_matches:
                keywords_list = [kw.strip().replace('"', '') for kw in keyword_list.split('\n- ')]
                keywords[category.lower()] = keywords_list
        
        return keywords
    
    def _extract_standard_responses(self, content: str) -> Dict[str, str]:
        """Extrai respostas padrão"""
        responses = {}
        
        # Encontrar seção de respostas padrão
        section_pattern = r"## Respostas Padrão\n\n(.*?)(?=\n## |\n\n# |\Z)"
        section_match = re.search(section_pattern, content, re.DOTALL)
        
        if section_match:
            section = section_match.group(1)
            
            # Extrair respostas
            response_pattern = r"\*\*([^*]+):\*\*\n\"(.+?)\""
            response_matches = re.findall(response_pattern, section)
            
            for response_type, response_text in response_matches:
                responses[response_type.lower()] = response_text
        
        return responses
    
    def get_available_agents(self) -> List[str]:
        """Retorna lista de agentes disponíveis dinamicamente"""
        return list(self._prompts.keys())
    
    def get_agent_descriptions(self) -> Dict[str, str]:
        """Retorna descrições de todos os agentes para classificação"""
        descriptions = {}
        for agent_name, prompt in self._prompts.items():
            descriptions[agent_name] = prompt.description or f"Agente {agent_name}"
        return descriptions
    
    def get_prompt(self, agent_name: str) -> Optional[AgentPrompt]:
        """Obtém prompt do agente"""
        return self._prompts.get(agent_name)
    
    def get_system_prompt(self, agent_name: str) -> str:
        """Obtém apenas o system prompt"""
        prompt = self.get_prompt(agent_name)
        return prompt.system_prompt if prompt else ""
    
    def get_examples(self, agent_name: str) -> List[PromptExample]:
        """Obtém exemplos do agente"""
        prompt = self.get_prompt(agent_name)
        return prompt.examples if prompt else []
    
    def format_examples_for_few_shot(self, agent_name: str, max_examples: int = 3) -> str:
        """Formata exemplos para few-shot learning"""
        examples = self.get_examples(agent_name)
        if not examples:
            return ""
        
        # Limitar número de exemplos
        examples = examples[:max_examples]
        
        formatted = "\n\nEXEMPLOS DE INTERAÇÃO:\n"
        for i, example in enumerate(examples, 1):
            formatted += f"\n--- Exemplo {i} ---\n"
            formatted += f"Cliente: {example.user_input}\n"
            formatted += f"Agente: {example.agent_response}\n"
        
        return formatted
    
    def get_keywords(self, agent_name: str) -> Dict[str, List[str]]:
        """Obtém palavras-chave do agente"""
        prompt = self.get_prompt(agent_name)
        return prompt.keywords if prompt else {}
    
    def get_standard_response(self, agent_name: str, response_type: str) -> str:
        """Obtém resposta padrão específica"""
        prompt = self.get_prompt(agent_name)
        if prompt:
            return prompt.standard_responses.get(response_type.lower(), "")
        return ""
    
    def format_context_with_citations(self, documents: List, query: str) -> str:
        """
        Formata contexto do RAG com citações [C#] para ancoragem
        Usado pelos agentes para incluir contexto nas respostas
        """
        if not documents:
            return "Não encontrei informações específicas na base do conhecimento do Jota para esta pergunta."
        
        context_parts = []
        available_citations = []
        
        for i, doc in enumerate(documents[:3], 1):  # Limitar aos 3 melhores resultados
            # Extrair número da seção do metadata se disponível
            section_num = doc.metadata.get('section', i)
            citation = f"[C{section_num}]"
            available_citations.append(citation)
            
            # Formatar conteúdo com citação e breadcrumbs
            content = doc.content.strip()
            if len(content) > 300:  # Limitar tamanho para não sobrecarregar
                content = content[:300] + "..."
            
            # Adicionar breadcrumbs se disponível
            breadcrumbs = ""
            if 'title' in doc.metadata:
                breadcrumbs = f" (Seção: {doc.metadata['title']})"
            
            context_parts.append(f"{citation} {content}{breadcrumbs}")
        
        formatted_context = f"""EVIDENCE PACK - Base de Conhecimento Oficial Jota:
{chr(10).join(context_parts)}

CITAÇÕES DISPONÍVEIS: {', '.join(available_citations)}"""
        
        return formatted_context
    
    def detect_strong_evidence_match(self, rag_result, query: str) -> Dict[str, Any]:
        """
        Detecta strong match no top-1/top-2 para evitar false fallbacks
        """
        if not rag_result or not rag_result.documents:
            return {"strong_match": False, "reason": "no_documents"}
        
        # Keywords para detecção de evidência direta
        keyword_patterns = {
            "aplicativo": ["aplicativo", "app", "celular", "mobile"],
            "cartão": ["cartão", "credito", "débito", "bandeira"],
            "rendimento": ["rendimento", "juros", "cdi", "percentual", "taxa"],
            "horário": ["horário", "atendimento", "funcionamento"],
            "pix": ["pix", "limite", "noturno", "transferência"],
            "conta": ["conta", "abertura", "criação", "cadastro"]
        }
        
        # Verificar top-2 documentos
        for i, doc in enumerate(rag_result.documents[:2]):
            content = doc.content.lower()
            query_lower = query.lower()
            
            # Encontrar keywords correspondentes
            matched_keywords = []
            for key, patterns in keyword_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    if any(pattern in content for pattern in patterns):
                        matched_keywords.append(key)
            
            # Critérios de evidência forte
            if matched_keywords:
                # Verificar afirmações diretas
                direct_statements = [
                    "não há aplicativo", "não existe aplicativo", "funciona pelo whatsapp",
                    "não emite cartão", "não oferece cartão", "sem cartão",
                    "100% cd", "100% do cdi", "rende 100%",
                    "não somos instituição financeira", "não há limite"
                ]
                
                if any(stmt in content for stmt in direct_statements):
                    # Extrair citation ID do metadata
                    section_num = doc.metadata.get('section', i + 1)
                    citation_id = f"[C{section_num}]"
                    
                    return {
                        "strong_match": True,
                        "doc_rank": i + 1,
                        "chunk_id": doc.chunk_id,
                        "section": section_num,
                        "citation_id": citation_id,
                        "snippet_original": doc.content[:300],
                        "matched_patterns": matched_keywords,
                        "direct_statement": next(stmt for stmt in direct_statements if stmt in content)
                    }
        
        return {"strong_match": False, "reason": "no_strong_match"}
    
    def create_evidence_pack(self, rag_result, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Cria Evidence Pack completo com metadados para logging e validação
        """
        # Usar env config como fallback
        if top_k is None:
            top_k = int(_get_env_fallback("CONTEXT_TOP_K_FINAL", "3"))
        
        # Detectar strong evidence match
        strong_match = self.detect_strong_evidence_match(rag_result, query)
        
        if not rag_result or not rag_result.documents:
            return {
                "context": "Não encontrei informações específicas na base do conhecimento do Jota para esta pergunta.",
                "citations": [],
                "chunks_used": [],
                "scores": [],
                "top_k": 0,
                "query": query,
                "answerable": False,
                "strong_match": strong_match
            }
        
        # Selecionar top_k documentos
        top_docs = rag_result.documents[:top_k]
        
        # 🆕 Metadados do context select
        retrieval_top_k = int(_get_env_fallback("RETRIEVAL_TOP_K", "8"))
        context_top_k_final = int(_get_env_fallback("CONTEXT_TOP_K_FINAL", "3"))
        
        # 🆕 Informações dos documentos selecionados
        selected_docs = []
        for doc in top_docs:
            doc_info = {
                "chunk_id": doc.chunk_id or doc.doc_id,
                "score_cosine": doc.metadata.get("score_cosine", doc.score),
                "score_lexical": doc.metadata.get("score_lexical", 0),
                "coverage": doc.metadata.get("coverage", 0),
                "final_score": doc.metadata.get("final_score", doc.score),
                "h_path": doc.metadata.get("h_path", ""),
                "title": doc.metadata.get("title", "")
            }
            selected_docs.append(doc_info)
        
        context_parts = []
        citations = []
        chunks_used = []
        scores = []
        
        import re
        
        for i, doc in enumerate(top_docs, 1):
            # 🆕 PARTE 2: Padronizar para índices consistentes [C1], [C2], etc.
            citation = f"[C{i+1}]"
            citations.append(citation)
            chunks_used.append(doc.chunk_id or f"chunk_{i}")
            scores.append(doc.score)
            
            # Guardar mapeamento para validação posterior
            if not hasattr(self, '_citation_mapping'):
                self._citation_mapping = {}
            self._citation_mapping[citation] = {
                'section_num': doc.metadata.get('section', str(i)),
                'chunk_id': doc.chunk_id or f"chunk_{i}",
                'score': doc.score
            }
            
            # 🆕 PARTE 1: Transformar conteúdo em frases declarativas completas
            content = doc.content.strip()
            
            # Remover cabeçalhos markdown e bullet points
            lines = content.split('\n')
            declarative_sentences = []
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Remover formatação visual
                line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)  # Remove bold
                line = re.sub(r'`([^`]+)`', r'\1', line)   # Remove inline code
                line = re.sub(r'^#+\s*', '', line)        # Remove markdown headers
                line = re.sub(r'^-\s*', '', line)         # Remove bullet points
                line = re.sub(r'^\*\s*', '', line)         # Remove bullet points
                line = re.sub(r'^\*\*\s*', '', line)       # Remove bold markers
                
                # Identificar conteúdo factual
                if line and not line.startswith('('):  # Evitar breadcrumbs no início
                    # 🆕 RAG PURO: Preservar conteúdo literal sem interpretação semântica
                    declarative_sentence = f"{citation} {line}"
                    declarative_sentences.append(declarative_sentence)
            
            # 🆕 PARTE 3: Manter formato - cada evidência como frase completa
            if declarative_sentences:
                # Usar a primeira frase declarativa como principal
                main_sentence = declarative_sentences[0]
                
                # Adicionar contexto adicional se houver
                if len(declarative_sentences) > 1:
                    additional_info = " ".join(declarative_sentences[1:])
                    main_sentence += f" {additional_info}"
                
                # Adicionar breadcrumbs simplificados
                if 'title' in doc.metadata:
                    main_sentence += f" ({doc.metadata['title']})"
                elif 'h_path' in doc.metadata:
                    main_sentence += f" ({doc.metadata['h_path']})"
                
                context_parts.append(main_sentence)
            else:
                # Fallback: se não conseguir extrair frases, usar conteúdo original simplificado
                simplified_content = re.sub(r'\n+', ' ', content)  # Remove quebras de linha
                simplified_content = re.sub(r'\s+', ' ', simplified_content)  # Remove espaços extras
                simplified_content = simplified_content[:200] + "..." if len(simplified_content) > 200 else simplified_content
                
                declarative_sentence = f"{citation} {simplified_content}"
                context_parts.append(declarative_sentence)
        
        # 🆕 PARTE 1: Criar contexto declarativo linear
        context_text = '\n'.join(context_parts)
        
        return {
            "context": f"""EVIDENCE PACK - Base de Conhecimento Oficial Jota:

{context_text}

CITAÇÕES DISPONÍVEIS: {', '.join(citations)}""",
            "citations": citations,
            "chunks_used": chunks_used,
            "scores": scores,
            "top_k": len(top_docs),
            "query": query,
            "answerable": len(top_docs) > 0 and max(scores) > 0.3,
            "strong_match": strong_match,
            # 🆕 Metadados do context select
            "retrieval_top_k": retrieval_top_k,
            "context_top_k_final": context_top_k_final,
            "selected_docs": selected_docs
        }
    def list_agents(self) -> List[str]:
        """Lista todos os agentes disponíveis"""
        return list(self._prompts.keys())
    
    def reload_prompts(self):
        """Recarrega todos os prompts"""
        self._prompts.clear()
        self._load_all_prompts()

# Instância global
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Obtém instância global do PromptManager"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
