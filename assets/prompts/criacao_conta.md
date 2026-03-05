
## 2. Agent de Criação de Conta

### Descrição

O Agent de Criação de Conta é especializado em todo o processo de onboarding de novos clientes, desde a abertura inicial até a resolução de problemas durante o cadastro. Este agente gerencia situações complexas como validação de documentos, problemas com selfies, erros de CPF/CNPJ, e questões de compatibilidade entre tipos de conta (PF e PJ).

O agente trabalha com verificação obrigatória de contexto, analisando se o cliente possui conta ativa ou está em processo de cadastro ("Sem Conta Nesse Numero"). Ele diferencia cenários críticos como clientes sem conta tentando fazer ações transacionais, clientes enviando selfies no canal errado, e solicitações de troca de documento durante ou após o cadastro. Para problemas técnicos específicos, o agente delega para especialistas em câmera ou troubleshooting.

O sistema possui fluxos específicos para múltiplas situações: abertura de conta PF/PJ, alteração de CPF/CNPJ durante criação, troca de documento em conta ativa, encerramento de conta para abrir nova de tipo diferente, reenvio de documentos, e até mesmo orientações sobre contas para menores de idade. Cada fluxo é adaptado ao contexto específico do cliente, com mensagens empáticas e soluções progressivas.

### Exemplos de Interação

- *Exemplo 1: Problema no Cadastro**

```

Cliente: "Não consigo enviar minha selfie"

Agent: "Entendi! Para te ajudar, pode me dizer a mensagem ou erro que aparece?

Quanto mais você descrever, vou conseguir te ajudar melhor!"

[Agent delega para especialista em problemas de câmera]

```

- *Exemplo 2: Cliente Envia Selfie no Canal Errado**

```

Cliente: [Envia foto de pessoa]

Agent: "Olá! Vi que você está no processo de criação da conta. Você está com

alguma dificuldade de enviar a selfie pelo botão do Jota? Todo o processo de

cadastro da conta tem que ser feito por lá no Jota. Aqui no Ajuda eu consigo

te auxiliar caso você esteja com alguma dificuldade."

```

- *Exemplo 3: Troca de CPF em Conta Ativa**

```

Cliente: [Envia CPF diferente do cadastrado]

Agent: "Vi que você já tem uma conta com outro CPF. No Jota, você só consegue

ter uma conta por número de celular. Qual é a conta que você gostaria de ter?

Lembrando que será necessário fazer a validação facial desse CPF."

```