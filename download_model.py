"""
Model loader for ARI-S Streamlit web app.
Priority: local file → download from private HuggingFace repo (token auth)
"""
import os

MODEL_FILENAME = "Multi_Pose_ResNet50_v221223.pth"
MODEL_URL = "https://huggingface.co/cheasther/ARI-S-weights/resolve/main/Multi_Pose_ResNet50_v221223.pth"


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

    dest = os.path.join(here, MODEL_FILENAME)
    return _download(MODEL_URL, dest)


def _hf_token() -> str:
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN", "")
    except Exception:
        return os.environ.get("HF_TOKEN", "")


def _download(url: str, dest: str) -> str:
    import requests
    import streamlit as st

    headers = {}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with st.spinner("모델 파일 다운로드 중... (약 285MB, 최초 1회)"):
        resp = requests.get(url, stream=True, allow_redirects=True,
                            timeout=300, headers=headers)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        bar   = st.progress(0, text="다운로드 중...")
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        bar.progress(min(downloaded / total, 1.0),
                                     text=f"다운로드 중... {downloaded/1e6:.0f}/{total/1e6:.0f} MB")

        bar.empty()

    if not os.path.exists(dest) or os.path.getsize(dest) < 1e6:
        raise RuntimeError("모델 다운로드 실패. URL 또는 HF_TOKEN을 확인해 주세요.")

    return dest
