# ThyroidCAD-System
ThyroidCAD-System 是一套以深度學習為核心的 甲狀腺超音波影像輔助診斷系統，整合 結節偵測、良惡性分類與結節分割 三個模型模組，並以 Flask 架構實作圖形化使用者介面（GUI），提供清楚、直觀且具互動性的操作流程。

本系統設計目的在於協助使用者以最少的操作步驟，完成甲狀腺結節之 定位、性質判斷與區域分割視覺化，提升影像判讀效率與臨床輔助應用的可行性。


## 系統特色
* 🔍 結節偵測（Detection）
使用 YOLOv11 模型自動定位甲狀腺結節區域
* 🧠 結節分類（Classification）
針對偵測到的 ROI 區域，判斷結節為良性或惡性
* ✂️ 結節分割（Segmentation）
對整張影像進行像素級結節區域分割
* 🖥️ 圖形化使用者介面（GUI）
單頁面整合原始影像、偵測結果、分割遮罩與分類結果
* ⚙️ 臨床流程導向設計
模擬實際醫師診斷流程，保留人工確認步驟以提升系統可控性

## 系統流程說明
1. 使用者上傳單張甲狀腺超音波影像
1. 系統顯示影像預覽，供使用者人工確認
1. 啟動推論流程後，系統首先執行 結節偵測模型
    * 若未偵測到任何結節，直接回傳「正常甲狀腺」結果
1. 若成功偵測結節：
    * 將偵測到的 ROI 區域裁切後送入分類模型
    * 同時將 原始整張影像送入分割模型
1. 推論完成後：
    * 左側顯示含偵測框的原始影像
    * 右側顯示以半透明遮罩標示之分割結果
    * 下方顯示結節分類結果與信心分數

## 使用模型
### 偵測與分割模型
* 架構：YOLOv11
* 框架：Ultralytics YOLO
* 權重檔：
    * det_yolo11l_best.pt
    * seg_yolox_best.pt

### 分類模型
* 架構：ResNet50
* 套件：timm
* 任務：良性 / 惡性二分類
* 權重檔：
    * resnet50_best.pth
* 分類僅針對 偵測後裁切之 ROI 區域 進行推論

```bash
ThyroidCAD-System/
├─ app.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ weights/
│  ├─ det_yolo11l_best.pt
│  ├─ seg_yolox_best.pt
│  └─ resnet50_best.pth
├─ static/
│  ├─ uploads/
│  └─ results/
└─ templates/
   ├─ index.html
   └─ result.html
```

### 安裝與執行方式
## 1. 建立虛擬環境（建議）
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```
## 2. 安裝套件
```bash
pip install -r requirements.txt
```
## 3. 確認權重檔案
請將以下模型權重放置於 weights/ 目錄中：
* det_yolo11l_best.pt
* seg_yolox_best.pt
* resnet50_best.pth

## 4. 啟動系統
```bash
python app.py
```
## 啟動後，於瀏覽器開啟：
```bash
http://127.0.0.1:5001
```

## 注意事項
本系統為 研究與學術展示用途
尚未經醫療器材相關法規認證
不可作為實際臨床診斷依據
建議使用 GPU 以獲得較佳推論效能（CPU 亦可運行）

## 未來發展方向
* 導入 TI-RADS 分類標準
* 自動計算結節大小（長軸、短軸）
* 判斷結節是否具備：
    * 垂直生長（Taller-than-wide）
    * 邊緣模糊或不規則
    * 低回音、微鈣化等影像特徵
* 強化臨床決策輔助能力，提升系統實用性

## 授權
本專案採用 MIT License，詳情請參閱 `LICENSE` 檔案。