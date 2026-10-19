# 單機工站效能優化與驗證（2026-10-19）

比較基準為 `cd24b2abcc3fc84210a94d044179da7d4a9ee65d`。正式工站資料未用作寫入測試；紀錄、故障注入與介面量測均使用暫存資料庫。原始 JSON、日誌與基準 checkout 保存在本機 `.report-build/performance/`，不納入 Git；本頁保存可重現方法與摘要。

## 實作

### 分段觀測

`FramePacket.stage_ms` 提供讀取、推論 RPC、追蹤、引擎、疊圖、耐久接受及封包準備耗時。追蹤另拆特徵抽取、配對、幾何估計、雙向光流及模板搜尋。AI 子程序回傳前處理、推論、後處理、結果轉換與打包；客戶端拆分影像拷貝、準備／鎖等待、往返及解包。

`VideoPipeline.performance_snapshot()` 提供最近最多 300 個已處理幀的 count/P50/P95/P99/max，包含 GUI 忙碌而未繪製的幀。沒有寫入事件的幀不會虛增 JPEG 編碼／pending commit 樣本。切換來源、工站或安裝新模型時清除舊統計。

將滑鼠停在視窗下方 FPS 狀態，可查看實際 backend、device、dtype、FP16、輸入形狀與分段時間；提示最多每秒更新一次。計時只觀測，不參與 SOP 時間窗。缺少或異常的選用 telemetry 不應中斷推論／判定。

時間單位均為 ms。巢狀階段不可相加，背景寫入時間亦不可與前景相加。Ultralytics `result.speed` 是該 runtime 已同步的階段時間，不是單獨 kernel 時間；TorchVision 額外回傳 CUDA event 時間與 CPU enqueue 時間。`capture_to_*` 從主機 `read()` 完成起算，不含相機曝光、驅動緩衝與螢幕呈現。影片時間戳不與系統單調時鐘相減。

### 追蹤恢復

正式管線啟用既有 adaptive budget；直接建立 `WorkpieceLocator` 預設仍保留固定策略，以供對照。

- 正常 LK、前後向檢查、幾何接受門檻、每 15 幀絕對核對與最多 60 幀光流上限保留。
- 失去有效光流且有偵測候選時，先嘗試局部 ORB，再保留全域 ORB。
- SIFT、局部 SIFT、模板公平輪替，每幀最多執行一個昂貴恢復階段；強制重試也占用同一名額。
- 全域搜尋長邊最多 960，局部最多 640；ORB/SIFT 分別最多 1200/800 特徵；光流最多 240 點，以 4×4 空間分布及極值維持覆蓋；RANSAC 最多 1000 次。
- 模板搜尋長邊最多 480，每次搜尋 3 或 6 個角度／尺度候選，跨幀輪替，不重用前幀搜尋分數當本幀量測。

這是軟預算。單次 OpenCV 呼叫無法中途取消；33.3 ms 是初始排程目標，250 ms 是重試機會，兩者都不是硬即時或恢復完成保證。無當前有效定位時，維持原有跟隨區域條件的阻擋／暫停機制。

### GUI 與 JPEG

紀錄頁及今日統計通知以 500 ms 計時器合併；持續通知不重啟計時器，避免刷新飢餓。隱藏紀錄頁僅標記 dirty，顯示、手動刷新、篩選、切站與維護完成可立即刷新。表格依資料 ID 重用項目，保留選取、捲動與未變動的證據圖片。SQL 仍同步讀取，本輪先減少重建及刷新密度。

編輯頁文字預覽最多 5 Hz；試跑依然處理 `all_results()` 每個結果。解除凍結或返回即時影像後，首幀立即更新。執行頁移除重複區域刷新。

pending 中第一次編碼的 JPEG bytes 直接寫入正式截圖；仍解碼驗證可讀性，兼容既有 pending。保留暫存檔、flush/fsync、原子替換、FULL commit、applied_jobs 去重，以及正式提交成功後才清除 pending 的順序。本輪未改精度、模型、事件內容、SOP 時間窗或交易批次策略。

## 同機比較

Windows 11、Intel i9-13900H、RTX 4050 Laptop 6 GB；Python 3.13.15、OpenCV 4.13.0、NumPy 2.5.3。追蹤使用兩版相同的 OpenCV 20 threads、1280×720 固定 seed 影像，6 種情境各重播 3 次，共 522 幀。完整 `locate()` 由工具外層計時，確保新舊量測邊界一致。

| 追蹤情境 | 原版 P50 / P95 | 新版 P50 / P95 |
| --- | ---: | ---: |
| 穩定小幅移動 | 6.46 / 16.29 | 4.01 / 11.59 |
| 遮擋及重新出現 | 7.08 / 109.86 | 3.81 / 10.15 |
| 無關雜訊及重新出現 | 7.56 / 221.92 | 5.37 / 68.00 |
| 大幅移動 | 6.58 / 23.11 | 4.28 / 14.87 |
| 錯誤偵測候選／干擾 | 6.39 / 24.50 | 4.48 / 17.81 |
| 正確候選引導恢復 | 7.48 / 108.22 | 4.10 / 12.43 |

兩版測試中的可見工件漏定位、無工件卻輸出有效定位、超過 8 px 容許誤差的定位、無定位時錯誤成立的消失條件，均為 0；三種重新出現情境皆在首個重新出現的取樣幀恢復。這是固定回放的取樣結果，不代表現場恢復時間為零。

定位座標不是逐位元相同：穩定移動的角點誤差 P95 為 3.14→3.56 px，其餘情境新版 P95 小於 1 px。特徵／解析度限制仍需使用現場遮擋、反光及 ROI 邊界影片驗收，不能把合成測試的零錯誤外推為所有工況保證。

紀錄使用相同 seed 的漸層加雜訊影像、JPEG 品質 85、每解析度 30 筆、一筆完成後才送下一筆。保留 WAL/FULL 與截圖 fsync，以 on_written 時戳量測，不使用輪詢推估完成時間。

| 紀錄階段 | 原版 P50 / P95 | 新版 P50 / P95 |
| --- | ---: | ---: |
| 720p 前景接受（含耐久 pending） | 3.09 / 3.98 | 2.95 / 3.90 |
| 720p 呼叫至正式寫入及清 pending | 10.59 / 13.68 | 8.55 / 11.09 |
| 1080p 前景接受（含耐久 pending） | 6.28 / 8.49 | 6.29 / 9.11 |
| 1080p 呼叫至正式寫入及清 pending | 20.92 / 24.76 | 16.35 / 21.06 |

前景仍需第一次編碼與耐久提交，因此未宣稱其明顯加速；1080p 前景 P95 此次略高，磁碟／排程也會影響結果。正式 JPEG 雜湊在新版與第一次編碼完全一致；兩種解析度均無遺失紀錄，pending 最後為零。

介面使用 offscreen Qt、500 列暫存紀錄、每次新增一輪後發送通知，共 40 次（QTimer 目標間隔 25 ms，忙碌時實際通知會延遲）：

| 項目 | 原版 | 新版 |
| --- | ---: | ---: |
| 紀錄完整刷新次數 | 40 | 3 |
| 今日統計查詢次數 | 40 | 3 |
| 紀錄刷新占用 GUI wall time 總和 | 1646.88 ms | 90.18 ms |
| 最後可見最新紀錄／今日統計 | 正確／540 | 正確／540 |
| 原選取證據及預覽圖片 | 被重設 | 保留 |

持續通知期間已有刷新，沒有等到停止寫入才更新。這些數字不含實體顯示器繪製，亦不是單次刷新時間或整條管線 FPS。

## CUDA 實際確認

PyTorch 2.7.1+cu128、CUDA runtime 12.8、NVIDIA driver 610.88；實際載入 M008 YOLO26s Detect（image_size=800）後，執行後端回報 `device=cuda:0`、`dtype=torch.float32`、`fp16=false`，實際網路輸入為 `[1,3,480,800]`。因此 AI 推論已使用 CUDA，追蹤／條件／GUI／SQLite 則仍主要在 CPU。

用 demo 影片相同的 60 幀各跑兩輪，新舊類別、分數、框與遮罩輸出完全一致。本輪沒有修改 GPU 模型或精度；RPC P50 兩輪原版為 17.22/14.12 ms，新版為 14.41/10.21 ms，受暖機、GPU 時脈與系統排程影響，不能將差值歸因為 GPU 演算法加速，也不能以此宣稱固定 FPS。

## 正確性與重現

完整測試套件 381 項：380 通過，1 項為未明確啟用的 NSIS 安裝／卸載整合測試。故障測試涵蓋：損毀 JPEG 保留 pending、fsync 失敗後重啟補寫、正式 commit 前後突然退出、重播不重複、耐久提交前不回報成功、舊 pending 相容。另有真實影片回放比較工具的 5 項測試通過。

`check_sop_replay.py` 的 8 種固定輸入情境，經真實 pipeline→engine→AsyncRecorder，50 個事件、76 幀狀態及資料庫業務內容與基準完全一致；涵蓋正常、違規、順序、數量與消失、下一輪重置、短／長定位遺失及異位恢復。SOP 快照雜湊相同，截圖存在且可解碼。此處固定定位輸入，是判定及紀錄路徑回歸測試，不能替代追蹤品質測試。

`check_video_tracking_replay.py` 另外以現有 demo、目前工站 SOP 與凍結的 60 幀 GPU 預測，比較真實定位器及引擎。影像雜湊吻合，影片時間為 0–11.8 秒；條件、引擎狀態與事件沒有差異。但兩版有效定位都為 0/60，僅停在第一道工序，未完成整輪，不能以這段片段宣稱現場定位驗收成功。工具會將「沒有共同有效定位」的角點差標為 null，而不是錯報為零誤差。

另以目前 SOP 的實際參考照片及工件框，產生已知仿射、遮擋及干擾的相同 174 幀，baseline fixed／新版 adaptive 各跑一次。兩版在 148 個可見幀及 26 個缺失幀中，上述四項錯誤計數皆為零；重新出現均於首個取樣幀恢復。穩定移動角點誤差 P95 為 4.90→4.70 px；其他情境新版 P95 為 0.81–1.24 px，部分較原版增加約 0.3–0.8 px，皆在此次 8 px 測試容許值內。這個值是回歸測試門檻，不是使用者確認的現場精度規格。照片及背景一同人工變換，不能取代實際反光、模糊、遮擋及 ROI 邊界驗收。原始結果為 `tracking_reference_before.json`／`tracking_reference_after.json`。

同一工具檔可指定不同 checkout，避免比較時連 fixture 也跟著改變。例如：

```powershell
.venv/Scripts/python.exe tools/benchmark_tracking_performance.py --project-root <baseline-checkout> --strategy fixed --repeat 3 --opencv-threads 20 --output <before.json>
.venv/Scripts/python.exe tools/benchmark_tracking_performance.py --strategy adaptive --repeat 3 --opencv-threads 20 --output <after.json>
.venv/Scripts/python.exe tools/benchmark_tracking_performance.py --reference-sop <station-sop.json> --strategy adaptive --opencv-threads 20 --output <reference-after.json>
.venv/Scripts/python.exe tools/benchmark_recorder_performance.py --project-root <baseline-checkout> --output <before.json>
.venv/Scripts/python.exe tools/benchmark_records_refresh.py --project-root <baseline-checkout> --output <before.json>
.venv/Scripts/python.exe tools/check_sop_replay.py --project-root <baseline-checkout> --output <before.json>
.venv/Scripts/python.exe tools/check_sop_replay.py --compare <before.json> --output <after.json>
.venv/Scripts/python.exe tools/benchmark_inference_performance.py --model <model.zip> --source <video.mp4> --frames 60 --output <inference.json>
.venv/Scripts/python.exe -m unittest discover -s tests -q
```

依序執行，勿同時啟動模型、相機與其他效能測試。省略 `--project-root` 使用目前 checkout。其他工具同樣支援 baseline/current 比較；所有輸出檔請使用不同名稱。
