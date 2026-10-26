# Protocolo da base OCR brasileira revisada

## Objetivo

Criar uma referência confiável para medir a leitura dos caracteres sem misturar
erros do detector YOLO com erros do OCR. A unidade de avaliação é um recorte da
placa associado a uma transcrição confirmada visualmente por uma pessoa.

## A base precisa de vídeo?

Não para começar. As caixas verdadeiras do dataset permitem extrair recortes
sem depender do detector e formam uma primeira base controlada. Os vídeos são
uma segunda etapa importante porque representam movimento, desfoque, distância,
perspectiva e iluminação do cenário de uso.

Vídeos sem transcrição servem para teste operacional, mas não permitem calcular
acurácia do OCR. Para entrarem na avaliação científica, os recortes selecionados
dos vídeos também precisam de revisão manual.

## Estado inicial

O comando abaixo gera recortes das divisões `valid` e `test`:

```powershell
uv run python -m scripts.build_ocr_review_set
```

Na execução inicial foram geradas 135 amostras. Em 28 delas, o nome do arquivo
continha uma placa brasileira aparentemente válida. Depois da revisão manual,
27 recortes foram mantidos com transcrição confirmada e 108 foram ignorados.
Todos os recortes mantidos são de placas brasileiras do padrão antigo; ainda é
necessário adicionar placas Mercosul para medir o cenário-alvo.

Os arquivos são locais e ignorados pelo Git:

```text
artifacts/ocr-review/
├── images/
└── manifest.csv
```

## Revisão manual

Inicie a interface principal:

```powershell
uv run streamlit run frontend/app.py
```

Abra a página **Revisar base OCR** na navegação do Streamlit. Para cada recorte:

1. transcreva somente os sete caracteres visíveis;
2. confirme `ABC1234` para o padrão antigo ou `ABC1D23` para Mercosul;
3. marque como **ilegível** se não houver informação visual suficiente;
4. marque como **ignorada** se não for uma placa brasileira ou se o recorte não
   servir ao objetivo do estudo;
5. use as observações para registrar reflexo, desfoque, oclusão ou dúvida.

Não complete caracteres por contexto e não aceite automaticamente a sugestão
do arquivo. Um rótulo incorreto contamina todas as comparações posteriores.

## Separação dos experimentos

- Ajustes de parâmetros devem usar treino e validação.
- O teste deve permanecer fechado até a configuração estar definida.
- Imagens aumentadas da mesma origem pertencem sempre à mesma divisão.
- Em vídeos, quadros da mesma passagem ou trajetória pertencem à mesma divisão.
- O resultado principal deve ser apresentado por placa ou trajetória, não como
  se cada quadro fosse uma amostra independente.

## Métricas recomendadas

- acerto exato da placa;
- acurácia por caractere e distância de edição;
- resultado separado entre placa antiga e Mercosul;
- resultado por condição: estática, movimento, noite, reflexo e placa pequena;
- tempo médio do OCR por recorte;
- taxa de autorização correta por passagem;
- tempo até a confirmação temporal.

## Inclusão futura de vídeos

Quando houver vídeo representativo da câmera-alvo:

1. executar o detector em frequência controlada;
2. agrupar detecções da mesma trajetória;
3. selecionar os melhores recortes por nitidez, tamanho e enquadramento;
4. revisar manualmente a transcrição;
5. reservar trajetórias completas para teste externo;
6. registrar câmera, resolução, distância, horário e condição de iluminação.

Essa etapa será adicionada sem misturar seus quadros com o teste atual.
