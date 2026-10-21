# SOP 工序監控

以攝影機與 AI 辨識協助檢查作業是否依照 SOP 順序完成的 Windows 桌面程式。
適用於單機工站，可在介面管理工站、編輯工序與作業區域，並查閱作業結果及歷史截圖。

[Windows 安裝版 v2026.10.19.1](https://github.com/kongbai0123/SOP/releases/tag/v2026.10.19.1) · [操作與維護指南](docs/user-guide.md) · [版本紀錄](CHANGELOG.md)

## 功能

- **工站管理**：每個工站各自保存 SOP、工序、設定、紀錄、截圖、備份與日誌。
- **流程編輯**：設定工序順序、完成與違規條件、觀察時間、成立比例及超時規則；每筆工序在設定側欄編輯，可選自動判定或人工確認。
- **影像判定**：支援 TorchVision、YOLO 與 RT-DETR 模型包，匯入共用模型庫供各工站使用，並支援固定／跟隨工件的作業區域。
- **試跑驗證**：以正式判定引擎試跑單步或完整流程，違規時在影像與試跑區顯示紅色警示並標出區域，不寫入正式紀錄。
- **執行監控**：依序判定工序，提供人工確認、警報與 OK／NG 結果；可切換逐一／全開顯示，並限制辨識頻率。
- **作業追溯**：保存作業員、工單、序號、工序耗時、實際 SOP 快照及事件截圖，支援 CSV 匯出。
- **資料維護**：持久保存待寫入事件、自動重試，以及工站備份、校驗與還原。

## 執行環境

- Windows x64；攝影機功能使用 Windows DirectShow。
- Python 與套件版本見 [requirements.txt](requirements.txt)。
- GPU 推論需相容的 NVIDIA 驅動；實際需求依模型與影像尺寸而定。
- 模型權重與測試影片需自行提供，儲存庫不包含作業資料。

## 原始碼啟動

在專案目錄建立環境並安裝依賴：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps ultralytics==8.4.150 ultralytics-thop ultralytics-platform
```

執行 `sop.bat` 啟動程式。

Windows 安裝版可從 [GitHub Release](https://github.com/kongbai0123/SOP/releases/tag/v2026.10.19.1) 下載 Setup、四個資料包分片及 `Assemble.cmd`，放在同一資料夾，先執行 `Assemble.cmd` 校驗並重組，再執行 Setup。
建置方式與部署說明見 [安裝文件](installer/README.md)。

## 使用流程

1. 選擇或建立工站。
2. 連線攝影機或影片，在「工序編輯 → 選擇模型…」匯入辨識模型。
3. 在「工序編輯」設定作業區域、工序與判定條件，儲存並試跑。
4. 在「執行監控」填寫作業資訊，開始作業。
5. 在「作業紀錄」查詢結果、查看歷史截圖、匯出或備份資料。

詳細步驟見 [操作與維護指南](docs/user-guide.md)。工件定位的圖解操作見 [操作圖解](docs/operation-guide/README.md)。

## 工站資料

開發版位於專案的 `data/stations/`；安裝版位於
`%LOCALAPPDATA%\SOP.WorkInstruction\data\stations\`。

```text
data/
├─ stations.json
└─ stations/
   ├─ ST001/
   │  ├─ station.json
   │  ├─ app_settings.json
   │  ├─ sops/                # SOP JSON，工序保存在各檔案的 steps 清單
   │  ├─ records/             # 作業紀錄、待寫入事件與 snapshots 截圖
   │  ├─ backups/
   │  └─ logs/
   └─ ST002/
```

首次啟動會將既有共用資料遷入預設工站，原始目錄保留為遷移前副本。

模型集中放在資料根目錄的 `models/`，所有工站共用：開發版為專案的 `models/`
（目前為 `C:\workspace\program\sop\models`），安裝版為 `%LOCALAPPDATA%\SOP.WorkInstruction\models`。
選擇模型 ZIP 或 `model.json` 時，程式會將模型複製至 `models/<名稱>-<SHA256>/`，
並在 SOP 保存 `models/...` 相對路徑；相同內容可重用，不同內容分開保存。
舊模型路徑仍可讀取，重新選擇即可匯入共用目錄。

各站資料與模型請另行備份，不納入 Git。工站備份不含共用模型；搬機時需另複製 `models/`，
其中 `.cache/` 可略過，已匯入模型不必再搬原始來源檔。詳細方式見 [資料維護說明](docs/user-guide.md#備份與還原)。

## 測試與開發

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

主要程式位於 `sop_app/`，介面位於 `sop_app/ui/`，測試位於 `tests/`，開發工具位於 `tools/`。
實體攝影機、模型精度與現場設備仍需依實際工站驗證。
