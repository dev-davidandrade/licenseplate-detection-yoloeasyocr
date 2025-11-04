import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
from PIL import Image
import numpy as np
import re


# ---------------------------
# Helpers
# ---------------------------
def clean_text(raw: str) -> str:
    """Remove tudo que não é letra ou número e retorna em maiúsculas."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def is_valid_plate(text: str) -> bool:
    """Valida formato de placa (padrão brasileiro Mercosul ou antigo)."""
    return bool(re.match(r"^[A-Z0-9]{6,7}$", text))


def best_plate_from_ocr_results(ocr_results):
    """
    Recebe lista de tuples (bbox, text, conf).
    Retorna o melhor texto, confiança e lista ordenada de candidatos.
    """
    candidates = []
    for _, text, conf in ocr_results:
        txt = clean_text(text)
        if not txt:
            continue
        # ignora palavras como BR, BRASIL, MERCOSUL
        if txt in ["BR", "BRASIL", "MERCOSUL"]:
            continue
        candidates.append((txt, float(conf)))

    if not candidates:
        return None, 0.0, []

    # separa em válidos (formato de placa) e genéricos
    valid = [c for c in candidates if is_valid_plate(c[0])]
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    if valid:
        # pega o de maior confiança entre os válidos
        best = max(valid, key=lambda x: x[1])
    else:
        # se nenhum válido, pega o mais confiante geral
        best = sorted_candidates[0]

    return best[0], best[1], sorted_candidates


def preprocess_variants(plate_rgb):
    """Gera variantes preprocessadas para tentar OCR."""
    h, w = plate_rgb.shape[:2]
    gray = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Bilateral
    bilateral = cv2.bilateralFilter(gray_clahe, 9, 75, 75)

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize
    scale = 2.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_rgb = cv2.resize(plate_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    resized_bilateral = cv2.bilateralFilter(resized_gray, 9, 75, 75)
    resized_thresh = cv2.adaptiveThreshold(
        resized_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return [
        ("gray_clahe", gray_clahe),
        ("bilateral", bilateral),
        ("thresh", thresh),
        ("morph", morph),
        ("resized_gray", resized_gray),
        ("resized_thresh", resized_thresh),
    ]


# ---------------------------
# Streamlit UI + lógica
# ---------------------------
st.set_page_config(page_title="Leitor de Placas", page_icon="🚗")
st.title("🚗 Leitor de Placas - YOLO + OCR (versão refinada)")
st.write(
    "Envie uma imagem de um veículo para detectar e ler a placa com precisão aprimorada."
)

model = YOLO("../backend/weights/weights_best.pt")
reader = easyocr.Reader(["pt"], gpu=False)

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    if len(img_np.shape) == 2 or (len(img_np.shape) == 3 and img_np.shape[2] == 1):
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    st.image(img_np, caption="Imagem Enviada", use_container_width=True)

    results = model(img_np)

    if len(results[0].boxes) == 0:
        st.warning("Nenhuma placa detectada. Tente outra imagem!")
    else:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0].item())

            display_img = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_img,
                f"Conf: {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Recorte refinado
            h, w = y2 - y1, x2 - x1
            cut_top, cut_bottom, cut_sides = int(h * 0.25), int(h * 0.05), int(w * 0.05)
            y1n, y2n = max(0, y1 + cut_top), min(img_np.shape[0], y2 - cut_bottom)
            x1n, x2n = max(0, x1 + cut_sides), min(img_np.shape[1], x2 - cut_sides)

            plate_crop = img_np[y1n:y2n, x1n:x2n]
            if plate_crop.size == 0:
                st.error("Recorte inválido — coordenadas fora da imagem.")
                continue

            variants = preprocess_variants(plate_crop)
            all_candidates = []

            for name, var_img in variants:
                try:
                    ocr_res = reader.readtext(var_img)
                except Exception:
                    try:
                        ocr_res = reader.readtext(
                            cv2.cvtColor(var_img, cv2.COLOR_GRAY2RGB)
                        )
                    except Exception:
                        ocr_res = []
                for _, txt, conf_txt in ocr_res:
                    all_candidates.append((None, txt, conf_txt))

            best_text, best_conf, sorted_candidates = best_plate_from_ocr_results(
                all_candidates
            )

            if not best_text:
                best_text = "Não detectado"
                best_conf = 0.0

            st.markdown(f"### Detecção #{i+1} — Conf (YOLO): {conf:.2f}")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(
                    cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                    caption="Imagem com box",
                    use_container_width=True,
                )
                st.image(
                    plate_crop, caption="Recorte da placa", use_container_width=True
                )

            with col2:
                st.write("**Melhor leitura:**")
                st.write(f"**{best_text}** (conf OCR: {best_conf:.2f})")

                st.write("**Candidatos:**")
                if sorted_candidates:
                    for txt, conf_txt in sorted_candidates[:8]:
                        st.write(f"- `{txt}` — conf {conf_txt:.2f}")
                else:
                    st.write("Nenhum candidato OCR encontrado.")

            st.write("---")
