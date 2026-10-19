# Windows NSIS 安裝精靈

此流程為現有 Python／PySide6 程式建立 Windows x64 自包含安裝包。版本讀取 `sop_app/__init__.py`，不另外維護一份應用程式版本。產品識別固定為 `SOP.WorkInstruction`。

版本可為三段日期或加上第四段同日修訂號，例如 `2026.10.19.1`。三段版本的 PE／NSIS 檔案版本補零，四段版本直接沿用。GitHub 原始碼版本更新不代表已重新建置安裝成品；本次 `2026.10.19.1` 發行未附 Windows 安裝包。

## 使用者安裝

在本機交付資料夾內，將 `SOP-版本-windows-x64-Setup.exe` 與同名的 `.payload.zip` 放在一起，雙擊 **Setup.exe**。資料包供安裝器讀取，不必自行解壓縮。

精靈依序為「歡迎 → 安裝位置 → 捷徑 → 安裝進度 → 完成」，提供繁體中文及英文資源，預設採系統語言；繁中系統顯示繁中。

- 安裝位置預設為 `%LOCALAPPDATA%\Programs\SOP.WorkInstruction`，可改為其他專用、可寫入的本機資料夾。
- 可分別勾選桌面、開始功能表捷徑；完成頁可選擇啟動程式，預設不勾選。
- 只安裝給目前使用者，使用 HKCU 登錄，不要求管理員權限。
- 內含 Python、Qt、OpenCV 及目前 CUDA 版推論執行環境；使用者不必安裝 Python 或 pip。GPU 使用仍需適合的 NVIDIA 驅動；未提供辨識模型，請在程式內選取自己的模型包。
- CUDA 執行環境體積較大，因此採 Setup 與離線資料包配對交付。安裝時不下載依賴、模型或更新。
- 本次離線資料包約 3.44 GB，安裝後程式約 5.39 GB；安裝時還需 staging 空間，升級時可能另需舊檔備份空間，請先確認目標磁碟容量。

沒有提供本產品正式授權條款，因此沒有虛構 EULA 同意頁。第三方 notices 與授權文字位於安裝目錄的 `_internal\notices`；本機試裝不代表已完成商業散布授權審查，特別是 Ultralytics 的 AGPL 條款。

## 個人資料、升級與移除

安裝版的各工站資料位於 `%LOCALAPPDATA%\SOP.WorkInstruction\data\stations\<工站編號>`，各自保存設定、SOP、紀錄、截圖、備份與執行日誌；程式檔與資料分離。啟動診斷仍可寫入資料根目錄的 `logs/`。

安裝器不會自動搬動原始碼工作區的資料。更新版程式首次正常啟動時，會將所在資料根目錄的舊資料複製至預設工站 `ST001`，保留原始檔。外部 SOP 可透過「開啟」匯入目前工站，模型路徑需另行確認；完整工站操作見 [操作與維護指南](../docs/user-guide.md)。

重新執行安裝器可同版重裝或更新。既有安裝沿用原位置，不支援安裝器內搬移；若要改位置，先解除安裝再安裝，個人資料會保留。較舊版本不得覆蓋較新版本。

安裝前先檢查整包 SHA-256，再解壓至 staging，逐檔檢查內容；校驗完成才替換程式檔。替換失敗會嘗試還原已改動檔案，無法還原時保留 recovery 副本及日誌。這不是支援斷電恢復的交易式更新系統。

解除安裝可從 Windows「已安裝的應用程式」或安裝目錄的 `Uninstall.exe` 執行。只移除 ownership 清單中的產品檔案及仍指向本產品的捷徑，不遞迴清空自選資料夾，也不移除個人資料或共用 runtime。檔案被鎖定時回報失敗並保留可重試入口。

新安裝只接受空的專用目錄；已有同產品 marker 的安裝目錄可保留未知檔案。拒絕磁碟根目錄、系統目錄、使用者資料根目錄及 junction／符號連結路徑。

## 建置（原始碼專案）

建置機使用 Windows x64、目前專案的 `.venv` 與既有 runtime 套件。僅建置工具可由下列命令安裝；不會升級應用程式依賴。

```powershell
powershell.exe -NoProfile -File .\tools\build_installer.ps1 -InstallBuildTools
```

腳本以自身位置解析專案路徑，不依賴呼叫時的工作目錄。原生 PowerShell 腳本保持 ASCII，支援 Windows PowerShell 5.1。

建置流程：

1. 驗證固定版本的 PyInstaller 等建置工具；需要時由官方 PyPI 下載並核對 wheel SHA-256。
2. PyInstaller onedir 產生 `installer/.build/payload/SOP`；使用乾淨的 DLL 搜尋路徑、完整重建與依來源內容隔離的位元組碼快取，避免誤收其他開發工具的原生 DLL 或舊程式。只收集執行內容，不複製開發 venv 或模型。封裝範例的模型絕對路徑會清空。
3. 收集第三方 notices，產生產品圖示與 PE 版本。
4. 建立離線 ZIP、逐檔 ownership manifest；由官方 SourceForge 下載固定 NSIS 3.13 ZIP，核對 `release-config.json` 的官方 SHA-256，解壓到專案 `.tools`。
5. 以 Unicode、`/INPUTCHARSET UTF8`、`/WX` 編譯 Modern UI 2。
6. 寫出 `release-manifest.json`、`SHA256SUMS.txt`、安裝說明。

輸出位於 `installer/releases/test/應用程式版本/windows-x64/`。工具、staging、交付二進位皆已加入 `.gitignore`。既有同版本輸出預設不覆寫；僅重做未公開的本機測試包時明確加 `-Rebuild`。

已完成 payload、只需重新編譯安裝器時：

```powershell
powershell.exe -NoProfile -File .\tools\build_installer.ps1 -SkipPayload -Rebuild
```

設定來源為 `release-config.json`；應用程式版本來源維持 `sop_app/__init__.py`。目前通道為 test、只交付本機，簽章策略 optional；未提供憑證時產物標示 `NotSigned`。腳本不購買憑證、不改信任設定、不自動公開發布，也不建立自動更新服務。

## 未簽章與其他 PC

使用者目前選擇不購買簽章服務，因此交付為未簽章測試版。NSIS 本身不強制購買憑證；允許未簽章應用程式的 PC 可以使用。Smart App Control、SmartScreen 與企業應用程式控制是不同機制，目標政策可能拒絕安裝器、主程式、子程序、DLL／PYD 或 PowerShell 腳本，不能承諾每台 PC 都能執行。管理者權限、換路徑或 `-ExecutionPolicy Bypass` 也不代表政策放行。

本次 2026-10-19 產生的 `2026.10.16` 版最終 `SOP.exe` 曾以 WinError 4551 被拒絕；CodeIntegrity 3077 的平面 SHA-256 與該程式相符，事件政策名稱為 `VerifiedAndReputableDesktop`。因此這份成品還不能宣稱已完成最終啟動驗收。完整資料包的直接部署／移除測試已通過，範圍見 `verification.json`；這不等於完整 NSIS 一般解除安裝或跨 PC 驗收通過。後續版本依各自的報告判定，不沿用本次結果。

轉交廠商時，將整個交付資料夾複製過去，保留 Setup 與 payload ZIP 同一目錄。先由廠商在其核准的測試環境檢查完整性與實際安裝、啟動、功能、捷徑及一般解除安裝。如被政策拒絕，交由有權限的 IT 依流程核准具體成品；沒有憑證時也可由企業建立允許規則，但是否准許及規則形式由該企業決定。每版成品改動後應重新核對，交付用整檔 SHA-256 不可直接當作 App Control 的 PE／Authenticode hash。

若日後使用公開受信任簽章，簽章不會綁定安裝 PC，接收者不需要私鑰；自簽憑證不會自動被其他 PC 信任。即使有有效簽章，新版本仍可能出現 SmartScreen 提示或不符合企業政策；本專案不會自行產生／匯入自簽信任或關閉防護。參考：[Microsoft SmartScreen](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation)、[App Control 規則](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/design/select-types-of-rules-to-create)、[PowerShell 與 App Control](https://learn.microsoft.com/en-us/powershell/scripting/security/app-control/how-app-control-works)。

## 靜默安裝

```text
SOP-版本-windows-x64-Setup.exe /S /DESKTOP=1 /STARTMENU=0 /D=C:\SOP 安裝
Uninstall.exe /S
```

`/S` 必須大寫。`/DESKTOP` 與 `/STARTMENU` 只接受 `0` 或 `1`，可省略；首次預設皆為 `1`，重裝沿用前次選擇。`/D=` 必須放最後，其值依 NSIS 規則不加引號；從其他程式啟動時仍需正確處理整條命令列。靜默安裝不啟動主程式。

成功退出碼為 `0`，失敗為非零；NSIS 取消也不算成功。詳細日誌位於 `%TEMP%\SOP.WorkInstruction-install.log` 或 `-uninstall.log`。GUI 與靜默安裝使用相同的寫入前驗證。

## 驗證範圍

`tests/test_installer_contract.py` 在 Windows 執行實際 helper／NSIS 安裝生命週期，以獨立 `SOP.WorkInstruction.Smoke.*` 產品識別、HKCU 鍵、捷徑及暫存路徑隔離，不覆蓋正式安裝。這些測試使用微型 fixture，只驗安裝邏輯。

另以真實 frozen payload 的以下入口驗 Qt／OpenCV、主視窗、spawn 及推論 native imports，不讀取正式作業資料、不連相機、不載入模型：

```text
SOP.exe --installer-smoke-test REPORT.json
SOP.exe --installer-runtime-test REPORT.json
```

後者另驗 PyTorch／TorchVision CPU tensor 與 NMS、Ultralytics import。實際結果另存交付目錄的 `verification.json`；沒有乾淨 VM、實機 GPU／攝影機驗收或 Authenticode 簽章時，不以本機測試冒稱已完成這些項目。

驗證完整資料包的解壓縮、逐檔校驗、換路徑啟動與移除，可執行：

下列版本路徑為建置後的示例，須先產生對應版本的交付資料夾。

```powershell
.venv\Scripts\python.exe tools\verify_installer_payload.py --release installer/releases/test/2026.10.19.1/windows-x64
```

此工具採新建的隔離目錄及產品識別，不建立登錄或捷徑。它驗證真實大型資料包與 deployment helper；不取代 NSIS 自身的 GUI／解除安裝程式自刪測試。

官方參考：[NSIS Modern UI 2](https://nsis.sourceforge.io/Docs/Modern%20UI%202/Readme.html)、[NSIS 指令](https://nsis.sourceforge.io/Docs/Chapter4.html)、[固定編譯器來源與 checksum](https://sourceforge.net/projects/nsis/files/NSIS%203/3.13/)、[PyInstaller 部署](https://pyinstaller.org/en/stable/operating-mode.html)。

## Windows 信任狀態的唯讀檢查

在允許執行專案診斷腳本的 Windows 環境中，可對交付資料夾執行以下命令。此工具不啟動 Setup 或 SOP，不解除封鎖，也不變更登錄、應用程式控制政策或憑證存放區；唯一寫入為明確指定的 JSON 報告。

```powershell
powershell.exe -NoProfile -File .\tools\check_windows_trust.ps1 -ReleaseDirectory .\installer\releases\test\2026.10.19.1\windows-x64 -OutputPath .\installer\releases\test\2026.10.19.1\windows-x64\windows-trust-report.json
```

交付資料夾也附有同一工具；在該資料夾開啟 PowerShell 時可執行 `powershell.exe -NoProfile -File .\check_windows_trust.ps1 -ReleaseDirectory . -OutputPath .\windows-trust-report.json`。原附報告來自建置機；若取得允許重新檢查，工具會更新自己的舊報告，請另指定檔名以保留歷史。退出 `0` 只代表完整性相符，仍需看政策事件並實際驗收。

工具以串流計算 manifest 所列成品的 SHA-256 與大小，檢查 Setup 的 Authenticode 狀態。只有明確傳入 `-PayloadDirectory .\installer\.build\payload\SOP`，才另檢查該目錄內 `SOP.exe` 的簽章；不解壓縮、不掃描或啟動整包 DLL，也不認定該目錄與 ZIP 完全相同。退出碼 `0` 只表示 manifest 的檔案大小與雜湊相符，`2` 表示校驗失敗或不可用；簽章、政策資訊另逐項列於報告。

報告盡可能讀取 `Win32_DeviceGuard` 原始欄位，以及系統提供的 `CiTool.exe -lp -json` 政策列表；不提權、不部署或重新整理政策。另分別檢查最近 24 小時各最多 100 筆 CodeIntegrity 3076／3077／3089，以及 AppLocker MSI and Script 8028／8029／8036／8038 事件，保留檔名或 ActivityID 關聯項目的完整 XML。可用 `-EventLookbackHours` 與 `-MaximumEvents` 調整有限範圍。缺少介面、權限不足與找不到事件會如實標示。這些資料無法完整涵蓋 Smart App Control、企業 App Control、AppLocker、防毒與其他限制；沒有事件或顯示有效簽章，均不能證明此機或其他 PC 允許啟動。分享報告前請檢查其中的本機路徑及事件資料。

如果 Windows 連診斷腳本本身也拒絕執行，保留錯誤並交由目標 PC 的管理者依核准流程處理；不要將無法執行改記為通過。參考：[Get-AuthenticodeSignature](https://learn.microsoft.com/powershell/module/microsoft.powershell.security/get-authenticodesignature)、[Win32_DeviceGuard 欄位](https://learn.microsoft.com/windows/security/hardware-security/enable-virtualization-based-protection-of-code-integrity)、[App Control 事件診斷](https://learn.microsoft.com/windows/security/application-security/application-control/app-control-for-business/operations/appcontrol-debugging-and-troubleshooting)。
