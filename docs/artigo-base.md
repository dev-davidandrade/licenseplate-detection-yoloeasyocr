# Reconhecimento temporal de placas veiculares brasileiras em vídeo com YOLO e EasyOCR

> Rascunho de trabalho. Os trechos marcados com **[DECIDIR]** ou **[INSERIR]**
> devem ser discutidos com o orientador antes de virar texto final.

## Resumo

Este trabalho investiga um sistema de reconhecimento automático de placas
veiculares brasileiras para cenários de controle de acesso. A proposta combina
um detector YOLOv8 Nano, reconhecimento de caracteres com EasyOCR, normalização
dos padrões brasileiro antigo e Mercosul e uma política de confirmação temporal
em vídeo. O objetivo não é tomar uma decisão com base em um único quadro, mas
acumular evidências de uma mesma passagem até obter uma leitura suficientemente
confiável. A baseline recuperada apresentou mAP@50 de 0,9482 no detector e
acerto exato de 64,29% no fluxo OCR normalizado sobre uma amostra inicial de 28
placas com rótulo inferido pelo nome do arquivo. Esses números são preliminares,
pois o conjunto OCR ainda é pequeno e precisa de revisão manual. Como
contribuição de engenharia, o projeto disponibiliza pipeline reproduzível,
interface para imagens, vídeos e câmera, testes automatizados e scripts de
avaliação. **[INSERIR resultados finais da base revisada e da câmera-alvo.]**

**Palavras-chave:** reconhecimento de placas; visão computacional; YOLO;
reconhecimento óptico de caracteres; vídeo; controle de acesso.

## 1. Introdução

O controle de entrada de veículos normalmente depende de identificação manual,
cartões, controles remotos ou sistemas comerciais de leitura automática de
placas. Em ambientes institucionais, uma alternativa baseada em visão
computacional pode reduzir etapas manuais e registrar evidências da passagem.
Entretanto, reconhecer a placa em uma imagem isolada não representa todo o
problema. Câmeras reais produzem quadros com movimento, reflexos, baixa
resolução, perspectiva e oclusões, e um OCR pode repetir o mesmo erro em vários
quadros.

Este trabalho parte de um protótipo com YOLO e EasyOCR e o reorganiza como uma
baseline reproduzível. A hipótese central é que a combinação entre detecção,
seleção de evidências, validação estrutural da placa e confirmação temporal pode
ser mais segura do que a aceitação imediata de uma única leitura.

### 1.1 Problema de pesquisa

Como construir e avaliar um pipeline reproduzível capaz de reconhecer placas
brasileiras em vídeo e produzir uma decisão de autorização com baixa incidência
de falsos aceites em hardware de baixo custo?

### 1.2 Questões de pesquisa

- **QP1:** qual é a contribuição relativa do detector e do OCR para os erros do
  sistema completo?
- **QP2:** a normalização dos formatos brasileiros melhora o acerto exato sem
  elevar falsos aceites?
- **QP3:** a confirmação temporal melhora a decisão por passagem quando
  comparada à leitura de um único quadro?
- **QP4:** qual frequência de análise oferece o melhor compromisso entre
  latência, acurácia e uso de CPU?
- **QP5:** **[DECIDIR com o orientador se haverá comparação de outro OCR ou novo
  treinamento.]**

### 1.3 Objetivo geral

Desenvolver e avaliar um sistema reproduzível de reconhecimento temporal de
placas brasileiras em vídeo para simulação de controle de acesso veicular.

### 1.4 Objetivos específicos

1. auditar o dataset e o modelo recuperados;
2. medir separadamente detector, OCR e pipeline ponta a ponta;
3. construir uma base OCR brasileira revisada manualmente;
4. comparar estratégias de pré-processamento e seleção de quadros;
5. implementar confirmação temporal e rejeição de leituras incompletas;
6. medir desempenho e tempo até a decisão em CPU;
7. disponibilizar uma demonstração reproduzível com câmera e vídeo.

## 2. Fundamentação e trabalhos relacionados

Esta seção deve contextualizar:

- reconhecimento automático de placas veiculares (ALPR/LPR);
- detectores de objetos de estágio único e família YOLO;
- OCR genérico e reconhecedores especializados em placas;
- placas brasileiras antigas e padrão Mercosul;
- agregação temporal, rastreamento e decisão por trajetória;
- requisitos de controle de acesso e custo de falsos aceites.

**[INSERIR revisão bibliográfica e referências definidas com o orientador. Não
usar esta lista como fundamentação final sem pesquisa de fontes.]**

## 3. Materiais e métodos

### 3.1 Arquitetura

O pipeline recebe uma imagem ou quadro de vídeo, detecta regiões de placa com
YOLOv8 Nano, recorta cada região, executa o EasyOCR com uma lista restrita a
caracteres alfanuméricos e normaliza o resultado conforme os formatos
`ABC1234` e `ABC1D23`. Leituras incompletas são mantidas para diagnóstico, mas
não participam da identificação ou autorização. Em vídeo, leituras válidas são
acumuladas por passagem e comparadas com uma lista de veículos autorizados.
Na câmera fixa, uma região de interesse configurável reduz a área enviada ao
detector. Antes do consenso, uma barreira de evidência aplica limites mínimos
de confiança do detector, confiança do OCR e qualidade do recorte. Esses
limites devem ser definidos na validação e mantidos fixos durante o teste.

### 3.2 Dataset do detector

O dataset recuperado possui 1.073 imagens exportadas: 939 para treino, 89 para
validação e 45 para teste. As imagens de treino correspondem a 313 origens com
aumentação de dados. Portanto, as 1.073 imagens não devem ser descritas como
amostras independentes. A divisão de teste contém 45 imagens e 46 instâncias.

### 3.3 Base revisada do OCR

As anotações originais contêm caixas, mas não transcrições. Foi criado um fluxo
que extrai os recortes das caixas verdadeiras e exige revisão manual do texto.
A primeira execução gerou 135 recortes de validação e teste. Após inspeção
manual, 27 foram mantidos com transcrição confirmada e 108 foram ignorados por
serem placas estrangeiras ou inadequadas ao escopo. As 27 amostras mantidas são
do padrão brasileiro antigo: 19 pertencem à validação e 8 ao teste. Portanto,
essa base ainda não permite concluir sobre placas Mercosul.

### 3.4 Política temporal

A política inicial exige duas ou mais confirmações da placa cadastrada, pelo
menos uma leitura bruta exatamente igual ao cadastro e intervalo máximo de três
segundos entre evidências da mesma passagem. Esses parâmetros serão tratados
como configuração experimental e não como valores universais.

### 3.5 Ambiente experimental

- CPU: AMD Ryzen 3 3100, quatro núcleos;
- Python: 3.12;
- execução do PyTorch em CPU;
- entrada do detector: 1024 pixels;
- **[INSERIR câmera, resolução, distância e iluminação do teste externo.]**

### 3.6 Métricas

| Camada | Métricas |
| --- | --- |
| Detector | precisão, recall, mAP@50 e mAP@50-95 |
| OCR | acerto exato, acurácia por caractere e distância de edição |
| Vídeo | taxa de detecção por passagem e estabilidade da leitura |
| Decisão | falso aceite, falsa rejeição e tempo até confirmação |
| Desempenho | segundos por inferência e quadros analisados por segundo |

### 3.7 Protocolo experimental proposto

1. congelar as divisões por imagem de origem ou trajetória;
2. revisar manualmente os rótulos OCR;
3. medir a baseline sem alterar parâmetros no teste;
4. ajustar alternativas apenas em treino e validação;
5. selecionar a configuração antes de abrir o teste;
6. avaliar imagens e vídeos externos da câmera-alvo;
7. repetir cada comparação com a mesma divisão e as mesmas métricas.

## 4. Resultados preliminares

### 4.1 Detector

| Métrica | Resultado |
| --- | ---: |
| Precisão | 0,9323 |
| Recall | 0,9565 |
| mAP@50 | 0,9482 |
| mAP@50-95 | 0,8267 |
| Inferência por imagem em CPU | 134,3 ms |

### 4.2 OCR e pipeline completo

Onze configurações do EasyOCR foram comparadas usando os rótulos revisados. O
recorte usado pela aplicação, com lista restrita a caracteres alfanuméricos,
foi selecionado pelo resultado da validação. A normalização estrutural converte
confusões compatíveis com as posições de letras e dígitos, mas o sistema de
autorização ainda exige ao menos uma leitura bruta exatamente igual ao cadastro.

| Divisão | Amostras | Exato bruto | Exato normalizado | Caracteres normalizados | Tempo médio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validação | 19 | 21,05% | 31,58% | 78,95% | 46,1 ms |
| Teste preliminar | 8 | 12,50% | 62,50% | 82,14% | 44,1 ms |
| Total descritivo | 27 | 18,52% | 40,74% | 79,89% | 45,5 ms |

O teste contém somente oito placas e já havia sido explorado em etapas
anteriores. Assim, esses números documentam a baseline revisada, mas não são a
estimativa final de generalização. O experimento definitivo deverá usar placas
Mercosul e trajetórias externas mantidas fora da escolha de configuração.

Na validação, elevar o limite mínimo de confiança do OCR de 0,30 para 0,60
reduziu as evidências aceitas de 12 para 7 e aumentou a precisão entre as
evidências aceitas de 50,0% para 85,7%, preservando os seis acertos
normalizados. Esse limite é provisório e deverá ser recalibrado com placas
Mercosul; ele não deve ser interpretado como universal.

### 4.3 Vídeos exploratórios

Cinco vídeos, totalizando 215,1 segundos, produziram 441 quadros amostrados e
229 detecções. O fluxo completo sustentou aproximadamente 7,2 análises por
segundo em CPU. Como esses vídeos ainda não possuem rótulos completos por
trajetória, o resultado mede viabilidade operacional, não acurácia.

## 5. Experimentos planejados

| Experimento | Alteração | Comparação principal |
| --- | --- | --- |
| E0 | Pipeline atual | baseline revisada |
| E1 | Seleção do melhor recorte por qualidade | acerto e tempo |
| E2 | Pré-processamentos definidos na validação | acerto e caracteres |
| E3 | Outro reconhecedor ou treinamento específico | **[DECIDIR]** |
| E4 | Confirmação temporal | quadro único versus passagem |
| E5 | Frequência de análise | latência versus CPU |

## 6. Ameaças à validade

- conjunto de teste pequeno;
- mistura de países no dataset original;
- base OCR revisada limitada a 27 placas brasileiras do padrão antigo;
- conjunto de teste preliminar já observado durante o desenvolvimento;
- poucas placas Mercosul confirmadas;
- vídeos exploratórios sem verdade de referência completa;
- risco de vazamento se quadros da mesma trajetória forem separados entre
  treino e teste;
- resultados de desempenho dependentes do hardware e da resolução.

## 7. Discussão

Os resultados preliminares sugerem que o detector não é o principal gargalo.
O reconhecimento dos caracteres concentra os erros e pode repetir confusões
sistemáticas, como `T/I`, mesmo em vários quadros. Isso limita a eficácia de um
consenso baseado apenas na frequência do texto. A discussão final deve separar
ganhos provenientes de melhor imagem, melhor OCR, normalização estrutural e
política temporal.

## 8. Conclusão

**[REDIGIR após os experimentos finais.]** A conclusão deverá responder às
questões de pesquisa, apresentar as limitações e evitar generalizar resultados
obtidos em uma única câmera ou em uma base pequena.

## 9. Pontos para conversar com o orientador

1. O trabalho será apresentado como sistema aplicado, comparação experimental
   ou estudo de viabilidade?
2. Qual é a pergunta de pesquisa principal?
3. Será necessário comparar EasyOCR, Tesseract e outro reconhecedor?
4. O acionamento físico do portão faz parte do escopo ou somente a simulação?
5. Qual tamanho mínimo de base revisada é aceitável?
6. Haverá submissão a comitê de ética ou procedimento institucional para
   gravação e tratamento das imagens?
7. Qual formato e veículo de publicação devem orientar o texto?

## Referências

**[INSERIR somente após levantamento bibliográfico em fontes acadêmicas.]**
