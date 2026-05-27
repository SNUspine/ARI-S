"""
Model loader for ARI-S Streamlit web app.
Priority: local file → Google Drive (via STREAMLIT secret or env var)
"""
import os
import sys

MODEL_FILENAME = "Multi_Pose_ResNet50_v221223.pth"


def get_model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(here, MODEL_FILENAME),
        os.path.join(here, "weights", MODEL_FILENAME),
        os.path.join(here, "..", "desktop", "weights", MODEL_FILENAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    # Try Google Drive download
    gdrive_id = _get_gdrive_id()
    if gdrive_id:
        dest = os.path.join(here, MODEL_FILENAME)
        return _download_gdrive(gdrive_id, dest)

    raise FileNotFoundError(
        f"모델 파일({MODEL_FILENAME})을 찾을 수 없습니다.\n\n"
        "해결 방법:\n"
        "1. Streamlit Cloud → Settings → Secrets 에 아래 항목 추가:\n"
        "   MODEL_GDRIVE_ID = \"your_google_drive_file_id\"\n\n"
        "2. 또는 모델 파일을 앱과 같은 폴더에 직접 복사"
    )


def _get_gdrive_id() -> str | None:
    # Env var first
    val = os.getenv("MODEL_GDRIVE_ID")
    if val:
        return val
    # Streamlit secrets
    try:
        import streamlit as st
        return st.secrets.get("MODEL_GDRIVE_ID")
    except Exception:
        return None


def _download_gdrive(file_id: str, dest: str) -> str:
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown 패키지가 필요합니다: pip install gdown")

    import streamlit as st
    with st.spinner(f"모델 파일 다운로드 중... (약 285MB, 최초 1회)"):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest, quiet=False)

    if not os.path.exists(dest):
        raise RuntimeError("모델 다운로드 실패. Google Drive 파일 ID를 확인해 주세요.")
    return dest
