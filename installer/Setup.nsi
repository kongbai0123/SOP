; NSIS 3 Unicode. Product.nsh is generated from release-config.json and app version.
Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"
!include "nsDialogs.nsh"
!include "x64.nsh"
!include "WinVer.nsh"
!ifndef PRODUCT_ID
!include "Product.nsh"
!endif

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "${OUTPUT_EXE}"
InstallDir "$LOCALAPPDATA\Programs\${PRODUCT_ID}"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 16
ShowInstDetails show
ShowUninstDetails show
ManifestDPIAware true
VIProductVersion "${FILE_VERSION}"
VIAddVersionKey /LANG=1033 "ProductName" "${PRODUCT_NAME}"
VIAddVersionKey /LANG=1033 "ProductVersion" "${PRODUCT_VERSION}"
VIAddVersionKey /LANG=1033 "FileVersion" "${FILE_VERSION}"
VIAddVersionKey /LANG=1033 "FileDescription" "${PRODUCT_NAME} Setup"
VIAddVersionKey /LANG=1033 "CompanyName" "${PUBLISHER}"
VIAddVersionKey /LANG=1033 "LegalCopyright" "${PUBLISHER}"

!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_ID}"
!define MUI_ICON "${ICON_FILE}"
!define MUI_UNICON "${ICON_FILE}"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "$(WelcomeTitle)"
!define MUI_WELCOMEPAGE_TEXT "$(WelcomeText)"
!insertmacro MUI_PAGE_WELCOME
!define MUI_PAGE_CUSTOMFUNCTION_PRE DirectoryPre
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE DirectoryLeave
!insertmacro MUI_PAGE_DIRECTORY
Page custom ShortcutPage ShortcutLeave
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\${ENTRY_EXE}"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!define MUI_FINISHPAGE_RUN_TEXT "$(RunApplication)"
!define MUI_FINISHPAGE_TEXT "$(FinishText)"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH
!insertmacro MUI_LANGUAGE "TradChinese"
!insertmacro MUI_LANGUAGE "English"

LangString WelcomeTitle ${LANG_TRADCHINESE} "歡迎安裝 ${PRODUCT_NAME}"
LangString WelcomeTitle ${LANG_ENGLISH} "Welcome to ${PRODUCT_NAME} Setup"
LangString WelcomeText ${LANG_TRADCHINESE} "此精靈將為目前的 Windows 使用者安裝 ${PRODUCT_NAME} ${PRODUCT_VERSION}。$\r$\n$\r$\n接下來可選擇安裝路徑，以及桌面和開始功能表捷徑。$\r$\n$\r$\n已包含 Python 與推論執行環境；辨識模型請在程式內另外選取。安裝時請將資料包與 Setup 放在同一資料夾。$\r$\n$\r$\n更新或解除安裝前，請先關閉程式。"
LangString WelcomeText ${LANG_ENGLISH} "Install ${PRODUCT_NAME} ${PRODUCT_VERSION} for the current Windows user.$\r$\n$\r$\nChoose the installation folder and desktop / Start menu shortcuts on the following pages.$\r$\n$\r$\nPython and the inference runtime are included. Select your own model inside the application. Keep the payload beside Setup.$\r$\n$\r$\nClose the application before updating or uninstalling."
LangString ShortcutsTitle ${LANG_TRADCHINESE} "選擇捷徑"
LangString ShortcutsTitle ${LANG_ENGLISH} "Choose shortcuts"
LangString ShortcutsSubtitle ${LANG_TRADCHINESE} "勾選希望建立的捷徑；僅影響目前使用者。"
LangString ShortcutsSubtitle ${LANG_ENGLISH} "Select shortcuts for the current user."
LangString DesktopText ${LANG_TRADCHINESE} "建立桌面捷徑"
LangString DesktopText ${LANG_ENGLISH} "Create a desktop shortcut"
LangString StartMenuText ${LANG_TRADCHINESE} "建立開始功能表捷徑"
LangString StartMenuText ${LANG_ENGLISH} "Create a Start menu shortcut"
LangString DataText ${LANG_TRADCHINESE} "設定、作業紀錄與日誌另存於使用者資料夾；更新及解除安裝時保留。"
LangString DataText ${LANG_ENGLISH} "Settings, records and logs are stored separately in your user profile and are preserved."
LangString RunApplication ${LANG_TRADCHINESE} "啟動 ${PRODUCT_NAME}"
LangString RunApplication ${LANG_ENGLISH} "Launch ${PRODUCT_NAME}"
LangString FinishText ${LANG_TRADCHINESE} "安裝完成。可使用所選捷徑啟動程式，並在工序編輯內選擇自己的辨識模型。"
LangString FinishText ${LANG_ENGLISH} "Setup is complete. Use your selected shortcuts to launch the application and select your own model in the editor."
LangString Unsupported ${LANG_TRADCHINESE} "此安裝包需要 Windows 10 或更新的 x64 系統。"
LangString Unsupported ${LANG_ENGLISH} "This package requires Windows 10 or later on x64."
LangString Failed ${LANG_TRADCHINESE} "操作未完成。請先關閉程式，並確認路徑及資料包正確。詳細原因：$\r$\n$LogPath"
LangString Failed ${LANG_ENGLISH} "The operation did not complete. Close the application and check the path and payload. Details:$\r$\n$LogPath"
LangString ExistingLocation ${LANG_TRADCHINESE} "更新會沿用原安裝位置。若需搬移，請先解除安裝；個人資料會保留。"
LangString ExistingLocation ${LANG_ENGLISH} "Updates keep the existing location. Uninstall first to move the program; personal data is preserved."
LangString Installing ${LANG_TRADCHINESE} "正在校驗、解壓縮並安裝執行環境，可能需要數分鐘…"
LangString Installing ${LANG_ENGLISH} "Verifying and installing the runtime. This can take several minutes..."
LangString Removing ${LANG_TRADCHINESE} "正在移除程式檔案；保留使用者資料…"
LangString Removing ${LANG_ENGLISH} "Removing application files; preserving user data..."

Var DesktopChoice
Var StartMenuChoice
Var DesktopCheckbox
Var StartMenuCheckbox
Var ExistingDir
Var LogPath
Var HelperResult
Var UninstallValidated
Var OperationStep

Function Fail
  FileOpen $0 "$LogPath" a
  FileSeek $0 0 END
  FileWrite $0 "NSIS ${PRODUCT_ID} failed at $OperationStep (helper=$HelperResult).$\r$\n"
  FileClose $0
  SetErrorLevel 1
  IfSilent +2
    MessageBox MB_OK|MB_ICONSTOP "$(Failed)"
  Abort
FunctionEnd

Function .onInit
  SetShellVarContext current
  SetRegView 64
  StrCpy $LogPath "$TEMP\${PRODUCT_ID}-install.log"
  StrCpy $OperationStep "initialize"
  ${IfNot} ${RunningX64}
    IfSilent +2
      MessageBox MB_OK|MB_ICONSTOP "$(Unsupported)"
    SetErrorLevel 1
    Quit
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    IfSilent +2
      MessageBox MB_OK|MB_ICONSTOP "$(Unsupported)"
    SetErrorLevel 1
    Quit
  ${EndIf}
  InitPluginsDir
  ClearErrors
  File /oname=$PLUGINSDIR\deploy.ps1 "${DEPLOY_SCRIPT}"
  File /oname=$PLUGINSDIR\shortcuts.ps1 "${SHORTCUT_SCRIPT}"
  File /oname=$PLUGINSDIR\payload-manifest.json "${PAYLOAD_MANIFEST}"
  ${If} ${Errors}
    Call Fail
    Quit
  ${EndIf}
  StrCpy $DesktopChoice 1
  StrCpy $StartMenuChoice 1
  ReadRegStr $ExistingDir HKCU "${UNINSTALL_KEY}" "InstallLocation"
  ${If} $ExistingDir != ""
    ; Only use the saved directory when no explicit /D= was provided.
    ; NSIS strips /D= from $CMDLINE; inspect the original Windows command line.
    System::Call 'kernel32::GetCommandLineW() t .r0'
    ClearErrors
    ${GetOptions} "$0" "/D=" $1
    ${If} ${Errors}
      StrCpy $INSTDIR $ExistingDir
    ${EndIf}
    ReadRegDWORD $DesktopChoice HKCU "${UNINSTALL_KEY}" "DesktopShortcut"
    ReadRegDWORD $StartMenuChoice HKCU "${UNINSTALL_KEY}" "StartMenuShortcut"
  ${EndIf}
  ${GetParameters} $0
  ClearErrors
  ${GetOptions} "$0" "/DESKTOP=" $1
  ${IfNot} ${Errors}
    StrCpy $DesktopChoice $1
  ${EndIf}
  ClearErrors
  ${GetOptions} "$0" "/STARTMENU=" $1
  ${IfNot} ${Errors}
    StrCpy $StartMenuChoice $1
  ${EndIf}
  ${If} $DesktopChoice != 0
  ${AndIf} $DesktopChoice != 1
    Call Fail
    Quit
  ${EndIf}
  ${If} $StartMenuChoice != 0
  ${AndIf} $StartMenuChoice != 1
    Call Fail
    Quit
  ${EndIf}
FunctionEnd

Function DirectoryPre
  ${If} $ExistingDir != ""
    !insertmacro MUI_HEADER_TEXT "$(MUI_TEXT_DIRECTORY_TITLE)" "$(ExistingLocation)"
  ${EndIf}
FunctionEnd

Function ValidateDirectory
  StrCpy $OperationStep "validate-directory"
  ClearErrors
  ExecWait '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "$PLUGINSDIR\deploy.ps1" -Action Validate -InstallDir "$INSTDIR" -ProductId "${PRODUCT_ID}" -Version "${PRODUCT_VERSION}" -LogPath "$LogPath"' $HelperResult
  IfErrors invalid
  StrCmp $HelperResult 0 valid
  invalid:
    Call Fail
  valid:
FunctionEnd

Function DirectoryLeave
  Call ValidateDirectory
FunctionEnd

Function ShortcutPage
  !insertmacro MUI_HEADER_TEXT "$(ShortcutsTitle)" "$(ShortcutsSubtitle)"
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateCheckbox} 0 8u 100% 14u "$(DesktopText)"
  Pop $DesktopCheckbox
  ${NSD_SetState} $DesktopCheckbox $DesktopChoice
  ${NSD_CreateCheckbox} 0 34u 100% 14u "$(StartMenuText)"
  Pop $StartMenuCheckbox
  ${NSD_SetState} $StartMenuCheckbox $StartMenuChoice
  ${NSD_CreateLabel} 0 70u 100% 40u "$(DataText)"
  Pop $0
  nsDialogs::Show
FunctionEnd

Function ShortcutLeave
  ${NSD_GetState} $DesktopCheckbox $DesktopChoice
  ${NSD_GetState} $StartMenuCheckbox $StartMenuChoice
FunctionEnd

Section "Application"
  AddSize ${ESTIMATED_KB}
  Call ValidateDirectory
  DetailPrint "$(Installing)"
  StrCpy $OperationStep "install-payload"
  ClearErrors
  ExecWait '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "$PLUGINSDIR\deploy.ps1" -Action Install -InstallDir "$INSTDIR" -ProductId "${PRODUCT_ID}" -Version "${PRODUCT_VERSION}" -PayloadPath "$EXEDIR\${PAYLOAD_NAME}" -PayloadSha256 "${PAYLOAD_HASH}" -ManifestPath "$PLUGINSDIR\payload-manifest.json" -LogPath "$LogPath"' $HelperResult
  IfErrors failed
  StrCmp $HelperResult 0 +2
    Goto failed
  SetOutPath "$INSTDIR"
  StrCpy $OperationStep "write-uninstaller"
  ClearErrors
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  IfErrors failed
  StrCpy $OperationStep "write-registry"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "Publisher" "${PUBLISHER}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\${ENTRY_EXE},0"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
  WriteRegStr HKCU "${UNINSTALL_KEY}" "QuietUninstallString" '$\"$INSTDIR\Uninstall.exe$\" /S'
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "EstimatedSize" ${ESTIMATED_KB}
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "DesktopShortcut" $DesktopChoice
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "StartMenuShortcut" $StartMenuChoice
  IfErrors failed
  StrCpy $OperationStep "write-shortcuts"
  ClearErrors
  ExecWait '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "$PLUGINSDIR\shortcuts.ps1" -Action Install -InstallDir "$INSTDIR" -ProductName "${PRODUCT_NAME}" -Desktop "$DesktopChoice" -StartMenu "$StartMenuChoice" -LogPath "$LogPath"' $HelperResult
  IfErrors failed
  StrCmp $HelperResult 0 done
  failed:
    Call Fail
  done:
SectionEnd

Function un.onInit
  SetShellVarContext current
  SetRegView 64
  StrCpy $LogPath "$TEMP\${PRODUCT_ID}-uninstall.log"
  InitPluginsDir
  File /oname=$PLUGINSDIR\deploy.ps1 "${DEPLOY_SCRIPT}"
  File /oname=$PLUGINSDIR\shortcuts.ps1 "${SHORTCUT_SCRIPT}"
FunctionEnd

Section "Uninstall"
  SetOutPath "$TEMP"
  StrCpy $UninstallValidated 0
  ClearErrors
  CopyFiles /SILENT "$INSTDIR\.sop-install.json" "$PLUGINSDIR\previous-marker.json"
  IfErrors unfailed
  DetailPrint "$(Removing)"
  ClearErrors
  ExecWait '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "$PLUGINSDIR\deploy.ps1" -Action Uninstall -InstallDir "$INSTDIR" -ProductId "${PRODUCT_ID}" -Version "${PRODUCT_VERSION}" -LogPath "$LogPath"' $HelperResult
  IfErrors unfailed
  StrCmp $HelperResult 0 +2
    Goto unfailed
  StrCpy $UninstallValidated 1
  ClearErrors
  ExecWait '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "$PLUGINSDIR\shortcuts.ps1" -Action Uninstall -InstallDir "$INSTDIR" -ProductName "${PRODUCT_NAME}" -LogPath "$LogPath"' $HelperResult
  IfErrors unfailed
  StrCmp $HelperResult 0 +2
    Goto unfailed
  ClearErrors
  Delete "$INSTDIR\Uninstall.exe"
  IfErrors unfailed
  Delete "$INSTDIR\.sop-install.json"
  IfErrors unfailed
  DeleteRegKey HKCU "${UNINSTALL_KEY}"
  IfErrors unfailed
  ; Unknown files are intentionally preserved; remove only an empty root.
  RMDir "$INSTDIR"
  SetErrorLevel 0
  Goto undoned
  unfailed:
    FileOpen $0 "$LogPath" a
    FileSeek $0 0 END
    FileWrite $0 "NSIS ${PRODUCT_ID} uninstall failed (helper=$HelperResult).$\r$\n"
    FileClose $0
    ; Keep a retry entry if metadata cleanup failed after removing owned files.
    ${If} $UninstallValidated == 1
      IfFileExists "$INSTDIR\Uninstall.exe" +2
        CopyFiles /SILENT "$EXEPATH" "$INSTDIR\Uninstall.exe"
      IfFileExists "$INSTDIR\.sop-install.json" +2
        CopyFiles /SILENT "$PLUGINSDIR\previous-marker.json" "$INSTDIR\.sop-install.json"
      WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "${PRODUCT_NAME}"
      WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
      WriteRegStr HKCU "${UNINSTALL_KEY}" "Publisher" "${PUBLISHER}"
      WriteRegStr HKCU "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
      WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
      WriteRegStr HKCU "${UNINSTALL_KEY}" "QuietUninstallString" '$\"$INSTDIR\Uninstall.exe$\" /S'
      WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
      WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
    ${EndIf}
    SetErrorLevel 1
    IfSilent +2
      MessageBox MB_OK|MB_ICONSTOP "$(Failed)"
    Abort
  undoned:
SectionEnd
