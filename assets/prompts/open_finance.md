
## 3. Agent de Open Finance

### Descrição

O Agent de Open Finance é especializado em resolver problemas e orientar clientes sobre conexão com bancos externos através do sistema Open Finance. Este agente possui conhecimento detalhado de mais de 12 instituições financeiras (Itaú, Nubank, Santander, Caixa, Bradesco, BB, C6, Inter, BTG, Mercado Pago, PagBank, Stone) incluindo passos específicos de cada aplicativo bancário.

O agente trabalha com fluxo progressivo de resolução, começando sempre pela tratativa do link (copiar e abrir em navegador), seguido por verificações de requisitos básicos (app visível, mesma titularidade PF↔PF ou PJ↔PJ), e então soluções específicas como trocar rede, trocar navegador, ou testar com outro banco. Para cada banco específico, o agente consulta a base de conhecimento (RAG) e fornece o passo a passo completo adaptado ao contexto do cliente.

O sistema identifica diferentes cenários: problemas de conexão, dúvidas sobre funcionamento, consulta de bancos disponíveis, erros específicos (ERR_UNKNOWN_URL_SCHEME, invalid_request_uri), problemas durante cadastro, app oculto ou em pasta segura, e questões de titularidade. Há tratamento especial para clientes que demonstram cansaço ou frustração após múltiplas tentativas, com escalação empática imediata.

### Exemplos de Interação

- *Exemplo 1: Problema de Conexão - Primeira Tentativa**

```

Cliente: "Não estou conseguindo vincular meu banco Inter"

Agent: "Olá! Esse é o canal oficial de atendimento do Jota. Entendi! Vamos

verificar juntos. Primeiro, vamos tentar uma solução rápida: parece que você

está com problema no acesso a página e pode ser que o WhatsApp não esteja

funcionando bem. Você poderia clicar nos 3 pontos e depois clicar em copiar

link? Com esse link copiado, você poderia abrir o Chrome ou o Safari e entrar

nesse site?"

```

- *Exemplo 2: Banco Específico com Passo a Passo**

```

Cliente: "Como conecto minha conta do Itaú?"

Agent: "Para conectar sua conta do Itaú, o processo sempre começa pelo WhatsApp

do Jota. Envie 'Quero conectar o Itaú no Jota' para o (11) 4004-8006. O Jota

vai te orientar e, quando você for redirecionado para o app do Itaú, siga estes

passos: 1. No menu, acesse Produtos > Conectividade > Open Finance..."

```

- *Exemplo 3: Cliente Frustrado - Escalação Empática**

```

Cliente: "Já tentei com vários bancos, já tentei várias vezes, nada funciona.

Estou cansado disso."

Agent: "Peço desculpas pela dificuldade que você está enfrentando. Entendo sua

frustração e a urgência de resolver isso. Vou escalar seu caso para uma análise

mais detalhada e retornarei em breve com uma solução."

```