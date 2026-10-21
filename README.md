# Leitor de placas com YOLO + EasyOCR

Projeto experimental para detectar placas com YOLOv8 e reconhecer seus
caracteres com EasyOCR. O repositório contém uma interface Streamlit para
imagens, câmera e vídeos, uma API FastAPI, scripts de avaliação reproduzível e
um primeiro mecanismo de decisão temporal.

## Preparação no Windows

O projeto usa Python 3.12.10 e mantém versões reproduzíveis em `uv.lock`.

```powershell
uv sync
```

O `uv sync` lê `pyproject.toml` e `uv.lock`, instala o Python indicado em
`.python-version` quando necessário e cria/atualiza `.venv`. Para ativar esse
ambiente manualmente no PowerShell, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

A ativação é opcional: `uv run <comando>` já executa o comando dentro da
`.venv` correta. Não é necessário rodar `python -m venv .venv` antes de
`uv sync`.

Para quem ainda não usa `uv`, o `requirements.txt` permanece disponível com
as mesmas versões fixadas.

## Interface

```powershell
uv run streamlit run frontend\app.py
```

Abra `http://localhost:8501` se o navegador não abrir automaticamente. Na
primeira análise, o carregamento dos modelos pode levar alguns segundos.

Na barra lateral é possível cadastrar placas autorizadas. A interface oferece
imagem, fotografia, vídeo e uma câmera contínua experimental via WebRTC. Nos
modos temporais, a simulação exige leituras repetidas e pelo menos uma leitura
bruta exata antes de indicar autorização. Essa decisão não controla hardware.

## API

```powershell
uv run uvicorn backend.main:app --reload
```

A documentação interativa fica em `http://localhost:8000/docs`. O endpoint
`POST /ocr` recebe a imagem em memória e não preserva o arquivo enviado.

## Critério de identificação

O detector pode localizar a região de uma placa mesmo quando o OCR lê apenas
parte do texto. O sistema mantém essa leitura bruta para diagnóstico, mas só
considera uma placa identificada quando há exatamente 7 caracteres e o texto
corresponde a um dos formatos brasileiros suportados:

- placa antiga: `ABC1234`;
- placa Mercosul: `ABC1D23`.

Leituras parciais, longas demais ou fora desses formatos não entram na decisão
temporal, não autorizam acesso e não são persistidas como placas identificadas.

## Processamento de vídeos

O comando abaixo analisa cinco quadros por segundo, cria vídeos anotados e
salva cada detecção em CSV. A pasta de entrada pode estar fora do repositório.

```powershell
uv run python -m scripts.process_videos `
  --input S:\videos_teste `
  --sample-fps 5 `
  --output-dir artifacts\video-results
```

Para fazer somente uma triagem rápida do detector, acrescente `--skip-ocr`.
Os resultados locais ficam em `artifacts/`, que não é versionada.

## Base OCR revisada

Gere 135 recortes das caixas verdadeiras de validação e teste:

```powershell
uv run python -m scripts.build_ocr_review_set
```

Depois, execute a interface e abra a página **Revisar base OCR**. O manifesto e
os recortes ficam em `artifacts/ocr-review/`; as sugestões derivadas dos nomes
dos arquivos só viram rótulos após confirmação manual. O protocolo completo
está em `docs/base-ocr-revisada.md`.

## Testes

```powershell
uv run python -m unittest discover -s tests -v
```

Os números da baseline, as limitações do dataset e os experimentos em imagens
e vídeos estão registrados em `docs/auditoria-inicial.md`.

## Organização principal

- `backend/services/recognition.py`: detector e OCR compartilhados;
- `backend/services/realtime.py`: amostragem e estado da câmera contínua;
- `backend/services/temporal.py`: consenso e política de autorização;
- `frontend/app.py`: demonstração interativa;
- `frontend/pages/1_Revisar_base_OCR.py`: revisão manual dos rótulos OCR;
- `scripts/`: benchmarks e processamento em lote;
- `tests/`: regras de placa, qualidade, decisão temporal e API.
