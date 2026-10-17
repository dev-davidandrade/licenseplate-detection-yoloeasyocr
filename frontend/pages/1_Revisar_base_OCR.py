from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.ocr_dataset import (  # noqa: E402
    load_manifest,
    review_progress,
    save_manifest,
    update_review,
)


MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "ocr-review" / "manifest.csv"

st.set_page_config(page_title="Revisar base OCR", page_icon="🔤", layout="wide")
st.title("🔤 Revisar base OCR")
st.caption(
    "Confirme manualmente o texto de cada placa. Sugestões automáticas não são "
    "tratadas como verdade de referência."
)

rows = load_manifest(MANIFEST_PATH)
if not rows:
    st.warning("A base de revisão ainda não foi gerada.")
    st.code(
        "uv run python -m scripts.build_ocr_review_set",
        language="powershell",
    )
    st.stop()

completed, total = review_progress(rows)
st.progress(completed / total if total else 0.0, text=f"{completed} de {total} revisadas")

status_filter = st.selectbox(
    "Mostrar",
    ["pending", "reviewed", "illegible", "ignored", "all"],
    format_func={
        "pending": "Pendentes",
        "reviewed": "Revisadas",
        "illegible": "Ilegíveis",
        "ignored": "Ignoradas",
        "all": "Todas",
    }.get,
)
filtered = [
    row for row in rows if status_filter == "all" or row["status"] == status_filter
]
if not filtered:
    st.success("Não há amostras nesse filtro.")
    st.stop()

index_key = f"ocr_review_index_{status_filter}"
index = min(st.session_state.get(index_key, 0), len(filtered) - 1)
st.session_state[index_key] = index
row = filtered[index]

st.write(f"Amostra **{index + 1} de {len(filtered)}** · `{row['sample_id']}`")
image_path = MANIFEST_PATH.parent / row["crop_path"]
image_column, form_column = st.columns([3, 2])
with image_column:
    st.image(str(image_path), use_container_width=True)
    st.caption(
        f"Divisão: {row['split']} · Origem: {row['source_group']} · "
        f"Status: {row['status']}"
    )

with form_column:
    initial_label = row["reviewed_label"] or row["suggested_label"]
    label = st.text_input(
        "Texto correto da placa",
        value=initial_label,
        max_chars=7,
        key=f"label_{row['sample_id']}",
        help="Use ABC1234 ou ABC1D23. Confira visualmente antes de salvar.",
    )
    notes = st.text_area(
        "Observação opcional",
        value=row["notes"],
        key=f"notes_{row['sample_id']}",
    )
    if row["suggested_label"]:
        st.info(f"Sugestão derivada do nome do arquivo: {row['suggested_label']}")

    if st.button("Confirmar rótulo", type="primary", use_container_width=True):
        try:
            update_review(rows, row["sample_id"], "reviewed", label, notes)
            save_manifest(MANIFEST_PATH, rows)
            st.rerun()
        except ValueError as error:
            st.error(str(error))

    action_columns = st.columns(2)
    if action_columns[0].button("Marcar ilegível", use_container_width=True):
        update_review(rows, row["sample_id"], "illegible", notes=notes)
        save_manifest(MANIFEST_PATH, rows)
        st.rerun()
    if action_columns[1].button("Ignorar", use_container_width=True):
        update_review(rows, row["sample_id"], "ignored", notes=notes)
        save_manifest(MANIFEST_PATH, rows)
        st.rerun()

navigation = st.columns(2)
if navigation[0].button("← Anterior", disabled=index == 0, use_container_width=True):
    st.session_state[index_key] = index - 1
    st.rerun()
if navigation[1].button(
    "Próxima →",
    disabled=index >= len(filtered) - 1,
    use_container_width=True,
):
    st.session_state[index_key] = index + 1
    st.rerun()
