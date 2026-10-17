from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from av import VideoFrame
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.recognition import (  # noqa: E402
    PlateDetection,
    load_default_recognizer,
)
from backend.services.realtime import RealtimePlateSession  # noqa: E402
from backend.services.plate_utils import PLATE_CHARACTER_COUNT  # noqa: E402
from backend.services.temporal import (  # noqa: E402
    PlateReading,
    TemporalPlateAggregator,
    parse_authorized_plates,
)


@st.cache_resource(show_spinner=False)
def load_recognizer():
    return load_default_recognizer()


def annotate_image(image_rgb: np.ndarray, detections: list[PlateDetection]) -> np.ndarray:
    annotated = image_rgb.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = (
            (0, 190, 0) if detection.valid_brazilian_plate else (255, 165, 0)
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        if detection.valid_brazilian_plate:
            label = detection.normalized_text
        elif detection.recognition_status == "partial":
            label = f"parcial: {detection.raw_text}"
        elif detection.recognition_status == "invalid":
            label = f"invalida: {detection.raw_text}"
        else:
            label = "sem leitura"
        cv2.putText(
            annotated,
            f"{label} {detection.detector_confidence:.2f}",
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def add_detection_to_aggregator(
    aggregator: TemporalPlateAggregator,
    detection: PlateDetection,
    frame_index: int,
    time_seconds: float,
) -> None:
    aggregator.add(
        PlateReading(
            raw_text=detection.raw_text,
            normalized_text=detection.normalized_text,
            detector_confidence=detection.detector_confidence,
            ocr_confidence=detection.ocr_confidence,
            quality_score=detection.quality_score,
            frame_index=frame_index,
            time_seconds=time_seconds,
        )
    )


def analyze_video(
    uploaded_file,
    recognizer,
    sample_fps: float,
    max_seconds: int,
    progress,
) -> dict[str, object]:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temporary_path = None
    capture = None
    started_at = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(uploaded_file.getbuffer())
            temporary_path = Path(temporary.name)

        capture = cv2.VideoCapture(str(temporary_path))
        if not capture.isOpened():
            raise ValueError("O vídeo não pôde ser aberto pelo OpenCV.")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_limit = min(source_frames, int(max_seconds * source_fps))
        stride = max(1, round(source_fps / sample_fps))
        expected_samples = max(1, (frame_limit + stride - 1) // stride)

        aggregator = TemporalPlateAggregator()
        evidence = []
        frame_index = 0
        sampled_frames = 0
        frames_with_detections = 0
        while frame_index < frame_limit:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = recognizer.analyze_image(frame_rgb)
                sampled_frames += 1
                if detections:
                    frames_with_detections += 1
                for detection in detections:
                    timestamp = frame_index / source_fps
                    add_detection_to_aggregator(
                        aggregator, detection, frame_index, timestamp
                    )
                    evidence.append(
                        {
                            "detection": detection,
                            "frame_index": frame_index,
                            "time_seconds": timestamp,
                        }
                    )
                progress.progress(
                    min(sampled_frames / expected_samples, 1.0),
                    text=f"Analisando amostra {sampled_frames} de {expected_samples}",
                )
            frame_index += 1
        capture.release()

        evidence.sort(
            key=lambda item: item["detection"].evidence_score,
            reverse=True,
        )
        elapsed = time.perf_counter() - started_at
        return {
            "source_name": uploaded_file.name,
            "source_size": uploaded_file.size,
            "source_fps": source_fps,
            "resolution": f"{width} × {height}",
            "duration_analyzed": frame_index / source_fps,
            "sampled_frames": sampled_frames,
            "frames_with_detections": frames_with_detections,
            "detections": len(evidence),
            "processing_seconds": elapsed,
            "processing_fps": sampled_frames / elapsed if elapsed else 0.0,
            "aggregator": aggregator,
            "evidence": evidence[:12],
        }
    finally:
        if capture is not None:
            capture.release()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def show_detection_cards(detections: list[PlateDetection]) -> None:
    for index, detection in enumerate(detections, start=1):
        st.divider()
        image_column, result_column = st.columns([1, 2])
        with image_column:
            st.image(
                detection.crop_rgb,
                caption=f"Recorte #{index}",
                use_container_width=True,
            )
        with result_column:
            displayed_text = (
                detection.normalized_text
                or detection.raw_text
                or "Não reconhecida"
            )
            if detection.valid_brazilian_plate:
                metric_label = "Placa identificada"
            elif detection.recognition_status == "partial":
                metric_label = "Leitura parcial do OCR"
            else:
                metric_label = "Leitura inválida do OCR"
            st.metric(metric_label, displayed_text)
            st.write(f"Texto bruto: `{detection.raw_text or 'vazio'}`")
            st.write(
                "Caracteres reconhecidos: "
                f"{detection.character_count}/{PLATE_CHARACTER_COUNT}"
            )
            st.write(f"Confiança do detector: {detection.detector_confidence:.1%}")
            st.write(f"Confiança do OCR: {detection.ocr_confidence:.1%}")
            st.write(f"Qualidade do recorte: {detection.quality_score:.1%}")
            if detection.valid_brazilian_plate:
                st.success(
                    "Placa brasileira completa. Esta leitura pode participar "
                    "da confirmação temporal."
                )
            elif detection.recognition_status == "partial":
                st.warning(
                    "Leitura parcial: são necessários exatamente 7 caracteres. "
                    "Este resultado não será usado para identificar ou autorizar."
                )
            elif detection.recognition_status == "unreadable":
                st.warning(
                    "A região da placa foi localizada, mas o OCR não reconheceu "
                    "caracteres."
                )
            else:
                st.warning(
                    "A leitura tem tamanho ou formato incompatível com uma placa "
                    "brasileira e não será usada para identificar ou autorizar."
                )


@st.fragment(run_every=1.0)
def show_realtime_status(session: RealtimePlateSession) -> None:
    snapshot = session.snapshot()
    columns = st.columns(4)
    columns[0].metric("Quadros recebidos", snapshot.frames_received)
    columns[1].metric("Quadros analisados", snapshot.frames_processed)
    columns[2].metric("Detecções", snapshot.detections)
    columns[3].metric(
        "Última análise",
        f"{snapshot.last_processing_seconds:.2f} s",
    )

    if snapshot.last_error:
        st.error(f"Erro durante a análise: {snapshot.last_error}")

    if snapshot.candidates:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Placa": candidate.plate,
                        "Leituras": candidate.votes,
                        "Leituras exatas": candidate.raw_exact_votes,
                        "Melhor confiança": candidate.best_confidence,
                    }
                    for candidate in snapshot.candidates
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Nenhuma placa brasileira completa foi acumulada ainda.")

    decision = snapshot.decision
    if decision.authorized:
        st.success(
            f"✅ Acesso simulado autorizado para {decision.plate}: "
            f"{decision.votes} confirmações."
        )
    elif decision.plate:
        st.warning(
            f"⏳ {decision.plate} detectada, aguardando mais evidências "
            f"({decision.reason})."
        )
    elif decision.reason != "lista_vazia":
        st.info("Nenhuma placa cadastrada foi confirmada.")


def show_realtime_camera(
    authorized_plates: set[str],
    analyses_per_second: int,
    min_confirmations: int,
) -> None:
    st.subheader("Câmera em tempo real")
    st.caption(
        "O vídeo continua fluido, mas YOLO e OCR analisam apenas a frequência "
        "selecionada. Clique em START e permita o acesso à câmera."
    )
    with st.spinner("Carregando os modelos para a câmera..."):
        recognizer = load_recognizer()

    session_key = "realtime_plate_session"
    session = st.session_state.get(session_key)
    if session is None or session.recognizer is not recognizer:
        session = RealtimePlateSession(recognizer)
        st.session_state[session_key] = session
    session.configure(
        analyses_per_second=analyses_per_second,
        authorized_plates=authorized_plates,
        min_confirmations=min_confirmations,
    )

    if st.button("Limpar histórico da passagem", use_container_width=True):
        session.reset()

    def process_video_frame(frame: VideoFrame) -> VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = session.analyze_if_due(frame_rgb)
        if detections is not None:
            frame_rgb = annotate_image(frame_rgb, detections)
        output_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return VideoFrame.from_ndarray(output_bgr, format="bgr24")

    webrtc_streamer(
        key="license-plate-realtime-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=process_video_frame,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
        },
    )
    show_realtime_status(session)


def show_video_result(
    result: dict[str, object],
    authorized_plates: set[str],
    min_confirmations: int,
) -> None:
    st.subheader("Resumo do vídeo")
    columns = st.columns(4)
    columns[0].metric("Quadros analisados", result["sampled_frames"])
    columns[1].metric("Com detecção", result["frames_with_detections"])
    columns[2].metric("Detecções", result["detections"])
    columns[3].metric("Velocidade", f"{result['processing_fps']:.1f} FPS")
    st.caption(
        f"{result['resolution']} · {result['source_fps']:.1f} FPS de origem · "
        f"{result['duration_analyzed']:.1f} s analisados em "
        f"{result['processing_seconds']:.1f} s"
    )

    aggregator = result["aggregator"]
    candidates = aggregator.candidates()
    if candidates:
        st.write("#### Candidatos acumulados")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Placa": candidate.plate,
                        "Leituras": candidate.votes,
                        "Leituras exatas": candidate.raw_exact_votes,
                        "Evidência ponderada": candidate.weighted_score,
                        "Melhor confiança": candidate.best_confidence,
                    }
                    for candidate in candidates
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("Nenhuma leitura formou uma placa brasileira válida.")

    if authorized_plates:
        decision = aggregator.decide(
            authorized_plates,
            min_votes=min_confirmations,
            min_raw_exact_votes=1,
        )
        if decision.authorized:
            st.success(
                f"✅ Acesso simulado autorizado para {decision.plate}: "
                f"{decision.votes} confirmações."
            )
        elif decision.plate:
            st.warning(
                f"⏳ {decision.plate} apareceu, mas ainda não atingiu os critérios "
                f"de confirmação ({decision.reason})."
            )
        else:
            st.error("⛔ Nenhuma placa autorizada foi confirmada.")
    else:
        st.info("Cadastre placas na barra lateral para simular a decisão de acesso.")

    evidence = result["evidence"]
    if evidence:
        st.write("#### Melhores recortes")
        columns = st.columns(3)
        for index, item in enumerate(evidence[:9]):
            detection = item["detection"]
            label = detection.normalized_text or detection.raw_text or "sem leitura"
            columns[index % 3].image(
                detection.crop_rgb,
                caption=(
                    f"{item['time_seconds']:.1f}s · {label} · "
                    f"qualidade {detection.quality_score:.0%}"
                ),
                use_container_width=True,
            )


st.set_page_config(
    page_title="Leitor de placas",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Leitor de placas")
st.caption("Protótipo experimental com YOLOv8, EasyOCR e decisão temporal")

with st.sidebar:
    st.header("Entrada")
    input_mode = st.radio(
        "Escolha a origem",
        [
            "Enviar imagem",
            "Fotografar com câmera",
            "Câmera em tempo real",
            "Analisar vídeo",
        ],
    )
    st.divider()
    st.header("Veículos autorizados")
    authorized_raw = st.text_area(
        "Uma placa por linha",
        placeholder="ABC1234\nBRA2E19",
        help="Somente placas brasileiras válidas entram na simulação.",
    )
    authorized_plates = parse_authorized_plates(authorized_raw)
    if authorized_raw and not authorized_plates:
        st.warning("Nenhuma placa cadastrada possui formato válido.")
    elif authorized_plates:
        st.caption(f"{len(authorized_plates)} placa(s) válida(s)")

    if input_mode == "Analisar vídeo":
        st.divider()
        st.header("Amostragem")
        sample_fps = st.slider("Análises por segundo", 1, 5, 2)
        max_seconds = st.slider("Limite do vídeo (segundos)", 5, 120, 30, 5)
        min_confirmations = st.slider("Confirmações para autorizar", 2, 5, 2)
    elif input_mode == "Câmera em tempo real":
        st.divider()
        st.header("Processamento ao vivo")
        realtime_analyses_per_second = st.slider(
            "Análises por segundo",
            1,
            5,
            2,
        )
        realtime_min_confirmations = st.slider(
            "Confirmações para autorizar",
            2,
            5,
            2,
        )

if input_mode == "Câmera em tempo real":
    show_realtime_camera(
        authorized_plates,
        realtime_analyses_per_second,
        realtime_min_confirmations,
    )
    st.stop()

if input_mode == "Enviar imagem":
    source = st.file_uploader(
        "Selecione uma imagem de veículo",
        type=["jpg", "jpeg", "png", "webp"],
    )
elif input_mode == "Fotografar com câmera":
    source = st.camera_input("Fotografe a placa do veículo")
else:
    video_source = st.file_uploader(
        "Selecione um vídeo",
        type=["mp4", "mov", "avi", "m4v", "mkv"],
    )
    st.caption(
        "Nesta fase, use preferencialmente um vídeo com apenas um veículo-alvo. "
        "O rastreamento de múltiplos veículos ainda é experimental."
    )
    if video_source is None:
        st.info("Envie um vídeo para iniciar a análise temporal.")
        st.stop()

    if st.button("Analisar vídeo", type="primary", use_container_width=True):
        progress = st.progress(0.0, text="Preparando o vídeo")
        try:
            with st.spinner("Carregando os modelos..."):
                recognizer = load_recognizer()
            result = analyze_video(
                video_source,
                recognizer,
                float(sample_fps),
                max_seconds,
                progress,
            )
            st.session_state["video_result"] = result
            progress.empty()
        except Exception as error:
            progress.empty()
            st.exception(error)

    saved_result = st.session_state.get("video_result")
    if (
        saved_result
        and saved_result["source_name"] == video_source.name
        and saved_result["source_size"] == video_source.size
    ):
        show_video_result(saved_result, authorized_plates, min_confirmations)
    st.stop()

if source is None:
    st.info("Envie uma imagem ou faça uma fotografia para iniciar a análise.")
    st.stop()

image_rgb = np.array(Image.open(source).convert("RGB"))
with st.spinner("Carregando os modelos e analisando a imagem..."):
    recognizer = load_recognizer()
    detections = recognizer.analyze_image(image_rgb)

st.subheader("Resultado")
st.image(
    annotate_image(image_rgb, detections),
    caption="Imagem analisada",
    use_container_width=True,
)
if not detections:
    st.warning("Nenhuma placa foi localizada nesta imagem.")
    st.stop()

if authorized_plates:
    preliminary_matches = [
        detection.normalized_text
        for detection in detections
        if detection.normalized_text in authorized_plates
    ]
    if preliminary_matches:
        st.info(
            f"Correspondência preliminar com {preliminary_matches[0]}. "
            "Uma imagem isolada não autoriza o acesso na política temporal."
        )
show_detection_cards(detections)
