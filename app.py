import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from download_model import get_model_path
from inference import run_inference

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="ARI-S: Lumbar Stenosis Classifier", layout="wide")
st.title("ARI-S: Lumbar Stenosis Classifier")
st.markdown(
    "3-view lateral X-ray로 **요추 협착증 확률**을 예측합니다. "
    "Multi-Pose ResNet50 모델과 **Grad-CAM** 히트맵으로 AI 판단 근거를 시각화합니다."
)
st.caption("⚠️ 본 도구는 연구 보조 목적으로만 사용됩니다. 임상 진단은 반드시 자격을 갖춘 의료 전문가가 수행해야 합니다.")

st.divider()

# ── Layout: Desktop vs Web info ───────────────────────────────────────────────

_c1, _c2 = st.columns(2, gap="large")

with _c1:
    with st.container(border=True):
        st.markdown("### 💻 Desktop Version")
        st.markdown("""
- ✅ 인터넷 연결 불필요
- ✅ 설치 없이 실행 (독립 실행형)
- ✅ **DICOM · JPG · PNG · BMP** 지원
- ✅ 고해상도 이미지를 손실없이 분석가능
- ✅ 확률(%)과 의심부위를 색으로 표시
- ✅ 결과 이미지 저장 가능
""")
        st.markdown("📧 구매 문의: [imspinesurgeon@gmail.com](mailto:imspinesurgeon@gmail.com)")

with _c2:
    with st.container(border=True):
        st.markdown("### 🌐 Web Version")
        st.markdown("""
- 🌐 브라우저에서 바로 사용 
- 📁 **JPG · PNG · BMP** 지원
- ⚡ 이미지를 압축하여 분석하여 정확도 일부 저하  
- 🧠 결과 이미지 다운로드 불가
""")

st.divider()

# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="모델 로딩 중... (최초 1회, 약 1~2분 소요)")
def load_engine():
    path = get_model_path()
    # Trigger model load (warm up)
    from inference import load_model
    return load_model(path), path

try:
    _, MODEL_PATH = load_engine()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"모델 로딩 실패: {e}")
    st.stop()

# ── Image upload ──────────────────────────────────────────────────────────────

st.subheader("X-ray 이미지 업로드")
st.markdown("Extension(신전위), Flexion(굴곡위), Neutral(중립위) 순서로 업로드해 주세요.")

col_ext, col_flx, col_neu = st.columns(3)

with col_ext:
    st.markdown("**Extension (신전위)**")
    ext_file = st.file_uploader("EXT", type=["jpg", "jpeg", "png", "bmp"], key="ext", label_visibility="collapsed")
    if ext_file:
        st.image(ext_file, width='stretch')

with col_flx:
    st.markdown("**Flexion (굴곡위)**")
    flx_file = st.file_uploader("FLX", type=["jpg", "jpeg", "png", "bmp"], key="flx", label_visibility="collapsed")
    if flx_file:
        st.image(flx_file, width='stretch')

with col_neu:
    st.markdown("**Neutral (중립위)**")
    neu_file = st.file_uploader("NEU", type=["jpg", "jpeg", "png", "bmp"], key="neu", label_visibility="collapsed")
    if neu_file:
        st.image(neu_file, width='stretch')

# ── Analyze button ────────────────────────────────────────────────────────────

st.divider()

if ext_file and flx_file and neu_file:
    if st.button("🔍 분석 시작", type="primary", use_container_width=True):
        with st.spinner("AI 분석 중..."):
            try:
                result = run_inference(
                    ext_file.getvalue(),
                    flx_file.getvalue(),
                    neu_file.getvalue(),
                    MODEL_PATH,
                )
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"분석 실패: {e}")
else:
    st.info("3장의 X-ray 이미지를 모두 업로드한 후 분석을 시작하세요.")

# ── Results ───────────────────────────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    prob   = result["stenosis_prob"]

    st.subheader("분석 결과")

    # Probability display
    pct = prob * 100
    if pct >= 70:
        color, label = "#F85149", "고위험 (High Risk)"
    elif pct >= 40:
        color, label = "#D29922", "중간 위험 (Moderate)"
    else:
        color, label = "#3FB950", "저위험 (Low Risk)"

    st.markdown(
        f"""
        <div style="
            background: #161B22;
            border: 2px solid {color};
            border-radius: 16px;
            padding: 28px 36px;
            text-align: center;
            margin-bottom: 24px;
        ">
            <div style="font-size: 48px; font-weight: 700; color: {color};">
                {pct:.1f}%
            </div>
            <div style="font-size: 20px; color: {color}; margin-top: 4px;">
                {label}
            </div>
            <div style="font-size: 14px; color: #8B949E; margin-top: 8px;">
                요추 협착증 예측 확률 (Stenosis Probability)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Grad-CAM heatmaps
    st.subheader("Grad-CAM 히트맵")
    st.caption("붉은 영역은 모델이 협착증이라고 판단한 부위입니다.")

    view_labels = {"ext": "Extension (신전위)", "flx": "Flexion (굴곡위)", "neu": "Neutral (중립위)"}
    hc1, hc2, hc3 = st.columns(3)

    def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    for col, view in zip((hc1, hc2, hc3), ("ext", "flx", "neu")):
        with col:
            st.markdown(f"**{view_labels[view]}**")
            st.image(_bgr_to_pil(result["heatmaps"][f"{view}_sten"]), width='stretch')

    # Download heatmap images
    st.subheader("결과 이미지 다운로드")
    dl_cols = st.columns(3)
    clicked = False
    for col, view in zip(dl_cols, ("ext", "flx", "neu")):
        with col:
            if st.button(f"📥 {view_labels[view]}", key=f"dl_{view}", use_container_width=True):
                clicked = True

    if clicked:
        st.warning(
            "**결과 이미지 저장은 Desktop 버전 전용 기능입니다.**\n\n"
            "Desktop 버전은 DICOM을 포함한 모든 이미지 형식을 지원하며, "
            "고해상도 결과 이미지 저장, 인터넷 없이 사용 등 더 많은 기능을 제공합니다.\n\n"
            "📧 구매 문의: **imspinesurgeon@gmail.com**"
        )
