# SOP 工序監控

以攝影機與 AI 辨識協助檢查作業是否依照 SOP 順序完成的 Windows 桌面程式。
適用於單機工站，可在介面管理工站、編輯工序與作業區域，並查閱作業結果及歷史截圖。

[版本 v2026.10.19](https://github.com/kongbai0123/SOP/releases/tag/v2026.10.19) · [操作與維護指南](docs/user-guide.md) · [版本紀錄](CHANGELOG.md)

## 功能

- **工站管理**：每個工站各自保存 SOP、工序、設定、紀錄、截圖、備份與日誌。
- **流程編輯**：設定工序順序、完成與違規條件、觀察時間、成立比例及超時規則。
- **影像判定**：支援 TorchVision、YOLO 與 RT-DETR 模型包，以及固定／跟隨工件的作業區域。
- **執行監控**：依序判定工序，提供人工確認、警報與 OK／NG 結果。
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

執行 `sop.bat` 啟動程式。Windows 安裝包的建置方式與部署說明見 [安裝文件](installer/README.md)；
GitHub 原始碼發行不代表已附帶安裝檔。

## 使用流程

1. 選擇或建立工站。
2. 連線攝影機或影片，選擇辨識模型。
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
各站資料與模型請另行備份，不納入 Git。搬機、還原及模型路徑的處理方式見 [資料維護說明](docs/user-guide.md#備份與還原)。

## 測試與開發

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

主要程式位於 `sop_app/`，介面位於 `sop_app/ui/`，測試位於 `tests/`，開發工具位於 `tools/`。
實體攝影機、模型精度與現場設備仍需依實際工站驗證。
