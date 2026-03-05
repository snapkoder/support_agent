
## 1. Agent de Atendimento Geral (Aline)

### Descrição

O Agent de Atendimento Geral é o ponto de entrada do sistema, responsável por receber todas as solicitações dos clientes e fazer a triagem inicial. Este agente utiliza análise de contexto e intenção para identificar rapidamente qual especialista deve assumir o atendimento, garantindo que o cliente seja direcionado para o agente mais adequado sem precisar repetir informações.

Aline, a assistente virtual, processa diversos tipos de entrada (texto, áudio, imagens) e mantém um histórico contextual de toda a conversa. Ela identifica automaticamente se o cliente possui conta ativa, extrai dados relevantes como nome, CPF/CNPJ, ticket ID e utiliza essas informações para personalizar o atendimento. O agente também verifica o tipo de conta (Pessoa Física ou Jurídica) para adaptar saudações e orientações.

O sistema opera com regras de precedência claras: detecta situações prioritárias como clientes sem conta tentando fazer transações, solicitações de troca de CPF em contas ativas, problemas de cadastro com envio de selfies no canal errado, e questões de Open Finance. Para casos que não se encaixam em especialistas específicos, o agente fornece orientações gerais sobre acesso à conta, dúvidas sobre produtos e serviços, e escala quando necessário através de notificações internas.

### Exemplos de Interação

- *Exemplo 1: Triagem para Criação de Conta**

```

Cliente: "Quero abrir uma conta no Jota"

Aline: "Olá! Esse é o canal oficial de atendimento do Jota. Para abrir sua conta

digital, é só entrar em contato com o Jota pelo número (11) 4004-8006 e seguir

o passo a passo informado durante a conversa no WhatsApp."

```

- *Exemplo 2: Redirecionamento para Especialista**

```

Cliente: "Cai em um golpe de Pix, preciso de ajuda"

Aline: "Olá, João! Esse é o canal oficial de atendimento do Jota. Sinto muito

pelo ocorrido. Para te ajudar, preciso de algumas informações..."

[Sistema delega automaticamente para Agent de Golpe Med]

```

- *Exemplo 3: Consulta Geral**

```

Cliente: "Como faço para acessar minha conta?"

Aline: "Olá! Para acessar a sua conta, é só falar com o Jota através do número

+55 11 4004-8006! Todas as funcionalidades podem ser utilizadas por lá!"

```