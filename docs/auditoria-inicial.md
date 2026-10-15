# Auditoria inicial e baseline

Data da auditoria: 15 de outubro de 2026.

## Objetivo

Registrar o estado inicial reproduzível do projeto antes de refatorações,
alterações no OCR ou novo treinamento. As métricas desta página formam a
baseline contra a qual versões futuras devem ser comparadas.

## Ambiente avaliado

- Sistema operacional: Windows.
- Processador: AMD Ryzen 3 3100 4-Core Processor.
- Python do projeto: 3.12.14, gerenciado pelo `uv` em `.venv`.
- PyTorch: 2.14.1, CPU.
- Ultralytics instalada para a avaliação: 8.4.177.
- A avaliação não utilizou GPU NVIDIA.

## Dataset recuperado

O arquivo recuperado é uma exportação Roboflow no formato YOLOv8, versão 2,
gerada em 4 de novembro de 2025 e licenciada sob CC BY 4.0.

Origem declarada pelo arquivo de metadados:

`https://universe.roboflow.com/site-yolo-easyocr-em-streamlit/yolo-easyocr-on-streamlit-ebns6/dataset/2`

| Divisão | Imagens | Arquivos de anotação |
| --- | ---: | ---: |
| Treino | 939 | 939 |
| Validação | 89 | 89 |
| Teste | 45 | 45 |
| Total | 1.073 | 1.073 |

O dataset contém uma única classe, `license plate`, e 1.073 caixas anotadas.
Não foram encontradas linhas malformadas nem coordenadas fora do intervalo
normalizado. Uma imagem de treino possui anotação vazia e uma imagem de teste
possui duas placas.

As 939 imagens de treino foram geradas a partir de 313 imagens de origem, com
três versões aumentadas de cada uma. Há 312 nomes-base porque duas imagens de
origem diferentes foram enviadas como `PAG5219.jpg` e `PAG5219.png`.

Antes das aumentações, a divisão inferida é de 313 imagens para treino, 89 para
validação e 45 para teste: aproximadamente 70%, 20% e 10% de 447 originais. A
comparação dos nomes-base não encontrou origens repetidas entre as divisões.
Portanto, as 1.073 imagens exportadas não devem ser apresentadas como 1.073
amostras independentes em trabalhos acadêmicos.

O aumento de dados declarado pelo Roboflow inclui:

- espelhamento horizontal com 50% de probabilidade;
- rotação aleatória entre -10 e +10 graus;
- ajuste aleatório de brilho entre -15% e +15%;
- redimensionamento para 640 x 640;
- auto-orientação e auto-contraste.

### Limitação para o OCR

As anotações descrevem somente as caixas das placas; não existe transcrição dos
caracteres em um arquivo de rótulos OCR. Foram identificadas 114 imagens cujo
nome-base parece conter o texto da placa. Elas podem formar uma primeira amostra
de avaliação, mas será necessário criar um conjunto OCR revisado manualmente.

A inspeção visual também encontrou placas brasileiras antigas e placas de outros
países. A cobertura de placas brasileiras no padrão Mercosul ainda precisa ser
medida e provavelmente ampliada.

## Modelo recuperado

Arquivo avaliado: `backend/weights/weights_best.pt`.

- Tarefa: detecção.
- Arquitetura: YOLOv8 Nano.
- Classe: `license plate`.
- Parâmetros: 3.005.843.
- Ultralytics usada no treinamento: 8.3.224.
- Data registrada no checkpoint: 4 de novembro de 2025.
- Épocas configuradas: 100.
- Tamanho de entrada: 1024.
- Batch: 8.
- Dispositivo original: GPU 0.
- Otimizador: automático.
- Seed: 0, com execução determinística habilitada.

## Baseline do detector

Avaliação feita sobre a divisão `test`, com 45 imagens e 46 instâncias,
resolução 1024, batch 1 e CPU.

| Métrica | Resultado |
| --- | ---: |
| Precisão | 0,9323 |
| Recall | 0,9565 |
| mAP@50 | 0,9482 |
| mAP@50-95 | 0,8267 |
| Pré-processamento por imagem | 2,6 ms |
| Inferência por imagem | 134,3 ms |
| Pós-processamento por imagem | 1,7 ms |

O tempo observado equivale a aproximadamente 7 quadros por segundo para a
detecção isolada nessa CPU e nessa resolução. Uma interface pode continuar
fluida usando detecção amostrada e rastreamento nos quadros intermediários.

## Baseline do OCR atual

O OCR foi avaliado separadamente, usando as caixas verdadeiras das anotações
para não misturar erros de detecção com erros de leitura. A amostra contém as
28 imagens de validação e teste cujo nome-base informa uma placa brasileira no
formato antigo. O processamento reproduziu o comportamento atual do frontend:
recorte de 25% do topo, seis variações de pré-processamento e EasyOCR em CPU.

| Métrica | Resultado |
| --- | ---: |
| Amostras | 28 |
| Acerto exato da placa | 0,3571 |
| Acurácia de caracteres | 0,8520 |
| Tempo médio por placa | 0,936 s |
| Acerto exato na validação | 7/19 (36,84%) |
| Acerto exato no teste | 3/9 (33,33%) |

O detector já apresenta resultado forte, mas o OCR é o principal gargalo de
qualidade e desempenho. A baseline confirma que executar seis leituras por
placa em todos os quadros não é adequado para tempo real. O arquivo detalhado
de resultados é gerado localmente em `artifacts/baseline/ocr.csv` pelo script
`scripts/benchmark_ocr.py`.

### Primeiro experimento de otimização

Como o detector YOLO já fornece uma região apertada, o segundo experimento
removeu a detecção de texto interna do EasyOCR e chamou diretamente o
reconhecedor, restringindo a saída a letras e números.

| Configuração | Acerto exato | Tempo médio por placa |
| --- | ---: | ---: |
| Pipeline original, seis leituras | 35,71% | 0,936 s |
| Reconhecimento direto | 50,00% | 0,046 s |
| Direto + normalização de formato | 67,86% | 0,046 s |

A normalização utiliza as posições válidas dos formatos brasileiro antigo e
Mercosul para corrigir ambiguidades como `0/O`, `5/S` e `2/Z`. Ela não tenta
adivinhar trocas entre caracteres da mesma categoria, como `P/F`.

Uma busca de margens feita somente na validação chegou a 73,68%, mas a margem
escolhida obteve apenas 11,11% nas nove imagens de teste. O resultado não foi
adotado: ele evidencia sobreajuste e a necessidade de uma amostra OCR maior e
mais representativa.

### Avaliação ponta a ponta

O fluxo otimizado também foi avaliado usando as caixas previstas pelo YOLO, em
vez das caixas perfeitas das anotações.

| Métrica | Resultado |
| --- | ---: |
| Placas detectadas | 28/28 (100%) |
| Texto bruto exatamente correto | 13/28 (46,43%) |
| Texto normalizado exatamente correto | 18/28 (64,29%) |
| Acurácia de caracteres normalizados | 88,27% |
| Tempo médio do detector | 0,271 s |
| Tempo médio do OCR | 0,112 s |

Essa amostra contém somente imagens com placa conhecida no nome do arquivo e,
portanto, não substitui um teste externo. Mesmo assim, ela confirma que o ganho
do reconhecimento direto permanece quando usamos as caixas reais produzidas
pelo detector.

## Teste externo em vídeos

Cinco vídeos externos foram usados como teste exploratório, sem incorporá-los
ao treino. O conjunto soma aproximadamente 215 segundos e contém cenas
brasileiras e internacionais, iluminação diurna e noturna, tráfego intenso,
movimento e placas pequenas. Como os vídeos não possuem anotações quadro a
quadro, esta etapa mede cobertura e estabilidade operacional, não acurácia.

O pipeline analisou duas imagens por segundo, com resolução de inferência 1024
e lotes de quatro imagens em CPU.

| Cena | Duração | Quadros amostrados | Quadros com detecção | Detecções |
| --- | ---: | ---: | ---: | ---: |
| Hikvision, carro em movimento | 11,0 s | 22 | 4 | 4 |
| Câmera LPR, pista noturna | 70,2 s | 147 | 14 | 14 |
| Tráfego intenso | 60,0 s | 120 | 91 | 130 |
| CCTV viário | 30,8 s | 65 | 28 | 33 |
| Câmera AHD, cena brasileira | 43,1 s | 87 | 48 | 48 |
| **Total** | **215,1 s** | **441** | **185** | **229** |

O detector levou em média 0,103 segundo por quadro amostrado quando executado
em lote. O OCR direto levou 0,070 segundo por detecção. Considerando a
frequência observada de placas, o fluxo completo sustentou aproximadamente 7,2
quadros analisados por segundo nessa CPU. Portanto, uma câmera de 30 FPS não
precisa bloquear a interface: cinco análises por segundo, com rastreamento nos
quadros intermediários, são viáveis no equipamento avaliado.

### Análise manual das cenas brasileiras

Na cena AHD foi possível confirmar visualmente a placa antiga `CXT0522`. O
EasyOCR produziu a leitura exata em 2 das 48 detecções. A alternativa mais
frequente foi `CXI0522`, mostrando uma confusão sistemática entre `T` e `I`.
O consenso temporal simples também escolheu a alternativa incorreta; repetir
quadros reduz ruído aleatório, mas não corrige sozinho um erro sistemático do
reconhecedor.

Uma comparação entre margens de recorte de 0%, 10%, 15%, 20% e 25% no topo não
justificou trocar a configuração global. A margem de 20% aumentou a confiança
dos dois acertos nesse vídeo, mas obteve desempenho inferior ao recorte atual
no conjunto de imagens. A alteração foi rejeitada para evitar sobreajuste.

No vídeo curto foram identificadas visualmente as placas `BCV8345` e
`DSZ1057`. Ao processar todos os 132 quadros, o YOLO gerou 23 detecções, mas o
OCR não recuperou nenhuma das duas placas por completo. Os recortes preservam
movimento, desfoque e perspectiva; várias leituras recuperaram parte dos
sufixos numéricos. Esse caso indica que o próximo ganho deve vir da seleção de
quadros por qualidade e de um reconhecedor ajustado para caracteres de placas,
não apenas do aumento da frequência de leitura.

Os vídeos anotados, o CSV de cada detecção e o resumo por trajetória são
gerados localmente por `scripts/process_videos.py`. O experimento de margens é
reproduzido por `scripts/evaluate_video_ocr_crops.py`.

## Base de aplicação após a auditoria

O frontend, a API e os novos experimentos passaram a usar o mesmo serviço em
`backend/services/recognition.py`. Esse serviço concentra carregamento dos
modelos, detecção, OCR direto, normalização e uma pontuação de qualidade do
recorte. A mudança evita que a interface apresente um comportamento diferente
do backend.

A interface Streamlit aceita imagem, fotografia do navegador e vídeo. No modo
de vídeo, ela permite controlar a frequência de amostragem, limita a duração
analisada, apresenta os melhores recortes e simula uma lista de placas
autorizadas. A simulação não representa um acionamento físico.

A política temporal inicial exige:

- pelo menos duas confirmações da placa autorizada;
- ao menos uma leitura bruta exatamente igual ao cadastro;
- confirmações separadas por no máximo três segundos.

Essa política prioriza evitar falsos aceites. No trecho inicial do vídeo
brasileiro, a leitura correta `CXT0522` apareceu apenas uma vez, enquanto o erro
`CXI0522` apareceu três vezes. O sistema recusou corretamente a autorização. A
regra e a separação entre passagens possuem testes automatizados.

A API deixou de salvar uploads usando o nome fornecido pelo cliente. As imagens
são decodificadas em memória, limitadas a 15 MB e encaminhadas ao mesmo serviço
de reconhecimento usado pela interface.

## Riscos metodológicos identificados

1. O conjunto de teste é pequeno: 45 imagens.
2. O dataset mistura países e padrões de placa diferentes.
3. A anotação atual não permite medir diretamente a acurácia do OCR.
4. Ainda não há um conjunto independente capturado na câmera-alvo.
5. As dependências originais não possuem versões fixadas.
6. O frontend e o backend implementam pipelines de OCR diferentes.

## Próximos marcos

1. Revisar manualmente os 28 rótulos usados na baseline do OCR.
2. Montar uma pequena base OCR brasileira revisada, incluindo os vídeos.
3. Selecionar os melhores recortes de cada trajetória por nitidez e tamanho.
4. Substituir o agrupamento global por rastreamento robusto de cada veículo.
5. Coletar um conjunto brasileiro independente, incluindo padrão Mercosul.
6. Comparar reconhecedores especializados de caracteres de placas.
7. Implementar câmera em tempo real com processamento assíncrono.
8. Criar o container após estabilizar o fluxo de vídeo e as dependências.
9. Comparar o modelo atual com um novo treinamento usando a mesma divisão de
   teste e as mesmas métricas.
