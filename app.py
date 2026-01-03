# -*- coding: utf-8 -*-
import os
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import timm
from flask import Flask, render_template, request, redirect, url_for, flash

from ultralytics import YOLO


# =========================
# Config
# =========================
APP_TITLE = "甲狀腺結節輔助診斷系統"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 你的權重路徑（可自行改）
DET_WEIGHT = os.path.join(WEIGHT_DIR, "det_yolo11l_best.pt")   # 偵測 best.pt
SEG_WEIGHT = os.path.join(WEIGHT_DIR, "seg_yolox_best.pt")     # 分割 best.pt（請確認是 Ultralytics seg 權重）
CLS_WEIGHT = os.path.join(WEIGHT_DIR, "resnet50_best.pth")     # ResNeSt50d 權重


# =========================
# Helpers
# =========================
def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT


def safe_imread(path: str) -> np.ndarray:
    """Read image as BGR uint8 (supports unicode path on Windows/macOS)."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("無法讀取影像，請確認檔案格式是否正確。")
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_bgr(path: str, img_bgr: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == "":
        ext = ".png"
        path = path + ext
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise ValueError("影像編碼失敗")
    buf.tofile(path)


def crop_with_padding(
    img_bgr: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    pad_ratio: float = 0.12
) -> np.ndarray:
    """Crop ROI with padding ratio, clamp to image boundaries."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w - 1, x2 + pad_w)
    ny2 = min(h - 1, y2 + pad_h)

    roi = img_bgr[ny1:ny2, nx1:nx2].copy()
    return roi


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a binary mask onto image with red color.
    mask: HxW, values 0/1 or 0-255
    """
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask.copy()
        if mask_u8.max() <= 1:
            mask_u8 = mask_u8 * 255

    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255  # BGR red channel

    m = mask_u8 == 255
    overlay[m] = cv2.addWeighted(img_bgr[m], 1 - alpha, red[m], alpha, 0)
    return overlay


def draw_bbox(img_bgr: np.ndarray, xyxy: Tuple[int, int, int, int], text: str = "") -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    out = img_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if text:
        cv2.putText(out, text, (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return out


# =========================
# Models
# =========================
@dataclass
class DetResult:
    found: bool
    best_xyxy: Optional[Tuple[int, int, int, int]] = None
    best_conf: float = 0.0
    all_boxes: Optional[List[Tuple[int, int, int, int, float]]] = None  # (x1,y1,x2,y2,conf)


class ThyroidSystem:
    def __init__(self, det_w: str, seg_w: str, cls_w: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- check weights
        if not os.path.exists(det_w):
            raise FileNotFoundError(f"找不到偵測權重: {det_w}")
        if not os.path.exists(seg_w):
            raise FileNotFoundError(f"找不到分割權重: {seg_w}")
        if not os.path.exists(cls_w):
            raise FileNotFoundError(f"找不到分類權重: {cls_w}")

        # --- YOLO models
        self.det_model = YOLO(det_w)
        self.seg_model = YOLO(seg_w)

        # --- Classifier (resnet50)
        self.cls_model = timm.create_model("resnet50", pretrained=False, num_classes=2)
        ckpt = torch.load(cls_w, map_location="cpu")

        state_dict = ckpt
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]

        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v
        self.cls_model.load_state_dict(cleaned, strict=False)

        self.cls_model.to(self.device).eval()

        # input
        self.input_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # label mapping (請依你訓練 label 對應確認)
        self.label_map = {0: "良性", 1: "惡性"}

    @torch.inference_mode()
    def detect_nodule(self, img_bgr: np.ndarray, conf_thres: float = 0.25) -> DetResult:
        preds = self.det_model.predict(img_bgr, conf=conf_thres, verbose=False)
        r = preds[0]

        if r.boxes is None or len(r.boxes) == 0:
            return DetResult(found=False, all_boxes=[])

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(float)

        best_i = int(np.argmax(confs))
        best = boxes[best_i]
        best_conf = float(confs[best_i])

        all_boxes = []
        for b, c in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, b.tolist())
            all_boxes.append((x1, y1, x2, y2, float(c)))

        return DetResult(
            found=True,
            best_xyxy=(int(best[0]), int(best[1]), int(best[2]), int(best[3])),
            best_conf=best_conf,
            all_boxes=all_boxes
        )

    @torch.inference_mode()
    def classify_roi(self, roi_bgr: np.ndarray) -> Dict[str, Any]:
        roi_rgb = bgr_to_rgb(roi_bgr)
        pil = Image.fromarray(roi_rgb).resize((self.input_size, self.input_size), Image.BILINEAR)

        x = np.asarray(pil).astype(np.float32) / 255.0  # HWC
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)  # 1CHW
        x = (x - self.mean) / self.std

        logits = self.cls_model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))

        return {
            "pred_id": pred,
            "pred_label": self.label_map.get(pred, str(pred)),
            "prob_benign": float(probs[0]),
            "prob_malignant": float(probs[1]),
            "confidence": float(probs[pred]),
        }

    @torch.inference_mode()
    def segment_nodule(self, img_bgr: np.ndarray, conf_thres: float = 0.25) -> np.ndarray:
        """
        Full image segmentation -> returns binary mask (H,W) uint8 {0,255}.
        If model outputs multiple instances, take highest-confidence mask.
        """
        preds = self.seg_model.predict(img_bgr, conf=conf_thres, verbose=False)
        r = preds[0]

        h, w = img_bgr.shape[:2]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        masks = r.masks.data.detach().cpu().numpy()  # N x Hm x Wm

        if r.boxes is not None and len(r.boxes) == masks.shape[0]:
            confs = r.boxes.conf.detach().cpu().numpy()
            best_i = int(np.argmax(confs))
            m = masks[best_i]
        else:
            m = masks.max(axis=0)

        m = (m > 0.5).astype(np.uint8) * 255

        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m

    def run_pipeline(self, img_path: str) -> Dict[str, Any]:
        img_bgr = safe_imread(img_path)

        det = self.detect_nodule(img_bgr, conf_thres=0.25)
        if not det.found or det.best_xyxy is None:
            return {
                "status": "normal",
                "message": "偵測階段未發現結節，系統判定為「正常甲狀腺」。",
                "det_found": False,
            }

        # ROI -> classification
        roi = crop_with_padding(img_bgr, det.best_xyxy, pad_ratio=0.12)
        cls = self.classify_roi(roi)

        # full image -> segmentation
        mask = self.segment_nodule(img_bgr, conf_thres=0.25)

        # visuals
        vis_left = draw_bbox(img_bgr, det.best_xyxy, text=f"nodule {det.best_conf:.3f}")
        vis_right = overlay_mask(img_bgr, mask, alpha=0.45)

        return {
            "status": "nodule",
            "message": "偵測到結節，已完成 ROI 分類與整圖分割推論。",
            "det_found": True,
            "det_best_xyxy": det.best_xyxy,
            "det_best_conf": det.best_conf,
            "cls": cls,
            "vis_left": vis_left,
            "vis_right": vis_right,
        }


# =========================
# Flask App
# =========================
app = Flask(__name__)
app.secret_key = "thyroid_gui_secret_key"

SYSTEM: Optional[ThyroidSystem] = None


def load_models():
    """
    Flask 3.x 建議在啟動時初始化（取代 before_first_request）
    """
    global SYSTEM
    SYSTEM = ThyroidSystem(DET_WEIGHT, SEG_WEIGHT, CLS_WEIGHT)
    print("✅ Models loaded:")
    print(f"  - DET: {DET_WEIGHT}")
    print(f"  - SEG: {SEG_WEIGHT}")
    print(f"  - CLS: {CLS_WEIGHT}")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


@app.get("/")
def index():
    return render_template("index.html", title=APP_TITLE)


@app.post("/upload")
def upload():
    if "file" not in request.files:
        flash("未選擇檔案")
        return redirect(url_for("index"))

    f = request.files["file"]
    if f.filename == "":
        flash("未選擇檔案")
        return redirect(url_for("index"))

    if not allowed_file(f.filename):
        flash("檔案格式不支援（請使用 jpg/png/bmp/webp）")
        return redirect(url_for("index"))

    ext = os.path.splitext(f.filename)[1].lower()
    uid = uuid.uuid4().hex
    save_name = f"{uid}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    f.save(save_path)

    return render_template(
        "index.html",
        title=APP_TITLE,
        uploaded_image=url_for("static", filename=f"uploads/{save_name}"),
        uploaded_filename=save_name
    )


@app.post("/infer")
def infer():
    filename = request.form.get("filename", "").strip()
    if not filename:
        flash("找不到已上傳影像，請重新上傳。")
        return redirect(url_for("index"))

    img_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(img_path):
        flash("找不到已上傳影像，請重新上傳。")
        return redirect(url_for("index"))

    if SYSTEM is None:
        flash("系統尚未完成模型初始化，請重新啟動伺服器。")
        return redirect(url_for("index"))

    try:
        out = SYSTEM.run_pipeline(img_path)
    except Exception as e:
        flash(f"推論失敗：{e}")
        return redirect(url_for("index"))

    if out["status"] == "normal":
        return render_template(
            "result.html",
            title=APP_TITLE,
            status="normal",
            message=out["message"],
            original=url_for("static", filename=f"uploads/{filename}"),
            left=url_for("static", filename=f"uploads/{filename}"),
            right=None,
            cls=None,
            det=None
        )

    uid = uuid.uuid4().hex
    left_name = f"{uid}_det.png"
    right_name = f"{uid}_seg.png"

    left_path = os.path.join(RESULT_DIR, left_name)
    right_path = os.path.join(RESULT_DIR, right_name)

    save_bgr(left_path, out["vis_left"])
    save_bgr(right_path, out["vis_right"])

    det_info = {"xyxy": out["det_best_xyxy"], "conf": out["det_best_conf"]}

    return render_template(
        "result.html",
        title=APP_TITLE,
        status="nodule",
        message=out["message"],
        original=url_for("static", filename=f"uploads/{filename}"),
        left=url_for("static", filename=f"results/{left_name}"),
        right=url_for("static", filename=f"results/{right_name}"),
        cls=out["cls"],
        det=det_info
    )


@app.get("/reset")
def reset():
    return redirect(url_for("index"))


if __name__ == "__main__":
    load_models()  # ✅ 啟動時載入模型（Flask 3.x 相容）
    app.run(host="0.0.0.0", port=5001, debug=True)
