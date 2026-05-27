"""
ARI-S inference engine — Streamlit web version
Multi-Pose ResNet50 with Grad-CAM
"""
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image


# ── Model Definition ──────────────────────────────────────────────────────────

class _SpineMPModel(nn.Module):
    def __init__(self):
        super().__init__()

        def _make_encoder():
            enc = models.resnet50(weights=None)
            enc.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            enc.fc    = nn.Linear(2048, 512)
            return enc

        self.encoder1 = _make_encoder()
        self.encoder2 = _make_encoder()
        self.encoder3 = _make_encoder()
        self.linear1  = nn.Linear(512 * 3, 512)
        self.linear2  = nn.Linear(512, 2)

    def forward(self, x1, x2, x3):
        y1, y2, y3 = self.encoder1(x1), self.encoder2(x2), self.encoder3(x3)
        d = torch.cat((F.relu(y1), F.relu(y2), F.relu(y3)), dim=1)
        d = F.relu(self.linear1(d))
        return F.softmax(self.linear2(d), dim=1)


class _TuningGradCAM(nn.Module):
    def __init__(self, model: _SpineMPModel):
        super().__init__()
        self.model = model
        self._fwd, self._bwd = {}, {}
        for name, enc in (('ext', model.encoder1), ('flx', model.encoder2), ('neu', model.encoder3)):
            enc.layer4[2].register_forward_hook(self._fwd_hook(name))
            enc.layer4[2].register_full_backward_hook(self._bwd_hook(name))

    def _fwd_hook(self, n): return lambda _, _i, o: self._fwd.__setitem__(n, o)
    def _bwd_hook(self, n): return lambda _, _i, o: self._bwd.__setitem__(n, o[0])

    def _gcam(self, fwd, bwd, shape):
        w   = F.adaptive_avg_pool2d(bwd, 1)
        cam = torch.mul(fwd, w).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, shape, mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().detach().numpy()

    def forward(self, x1, x2, x3):
        shape = x1.shape[2:]
        pred  = self.model(x1, x2, x3)
        out   = {}
        pred[0][1].backward(retain_graph=True)
        for key in ('ext', 'flx', 'neu'):
            out[f'{key}_sten'] = self._gcam(self._fwd[key], self._bwd[key], shape)
        return out


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _get_transform():
    return A.Compose(
        [A.Normalize(0.5, 0.5), ToTensorV2()],
        additional_targets={'image2': 'image', 'image3': 'image'},
    )


def _crop_clahe(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    ch, cw = h // 2, w // 2
    if w < 1200:
        hc = int((w / 600) * 700)
        crop = bgr[max(0, ch - hc//2): ch + hc//2]
    else:
        crop = bgr[ch - 700: ch + 700, cw - 600: cw + 600]
    res = cv2.resize(crop, (600, 700))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(res[:, :, 0])


def _pil_to_bgr(img_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _overlay_heatmap(gray: np.ndarray, cam: np.ndarray) -> np.ndarray:
    rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cam_resized = cv2.resize(cam, (gray.shape[1], gray.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return np.uint8(heat * 0.5 + rgb * 0.5)


# ── Model singleton ───────────────────────────────────────────────────────────

_model: _SpineMPModel | None = None


def load_model(weight_path: str) -> _SpineMPModel:
    global _model
    if _model is None:
        m = _SpineMPModel()
        m.load_state_dict(torch.load(weight_path, map_location='cpu'))
        m.eval()
        _model = m
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def run_inference(ext_bytes: bytes, flx_bytes: bytes, neu_bytes: bytes,
                  weight_path: str) -> dict:
    """
    Run stenosis classification on 3 X-ray images (bytes).

    Returns:
        {
            "stenosis_prob": float,          # 0–1
            "originals":  [gray, gray, gray], # numpy arrays (EXT/FLX/NEU)
            "heatmaps":   { "ext_sten": ndarray, "flx_sten": ..., "neu_sten": ... }
        }
    """
    model = load_model(weight_path)
    tf    = _get_transform()

    originals, grays, tensors = [], [], []
    for raw in (ext_bytes, flx_bytes, neu_bytes):
        bgr  = _pil_to_bgr(raw)
        orig_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        originals.append(orig_gray)
        gray = _crop_clahe(bgr)
        grays.append(gray)
        t = tf(image=gray, image2=gray, image3=gray)['image']
        tensors.append(t.unsqueeze(0))

    t1, t2, t3 = tensors

    with torch.no_grad():
        prob = model(t1, t2, t3)[0][1].item()

    gcam     = _TuningGradCAM(model)
    cams     = gcam(t1, t2, t3)
    heatmaps = {}
    for i, view in enumerate(('ext', 'flx', 'neu')):
        heatmaps[f'{view}_sten'] = _overlay_heatmap(originals[i], cams[f'{view}_sten'])

    return {
        'stenosis_prob': round(prob, 4),
        'originals':     originals,
        'heatmaps':      heatmaps,
    }
