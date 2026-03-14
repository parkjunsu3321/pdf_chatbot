@echo off
chcp 65001 > nul
echo [PDFChatbot] exe 빌드 중...
echo.

REM App 폴더로 이동
cd /d "%~dp0"

REM 가상환경 사용 시 활성화 (필요시 아래 주석 해제)
REM call venv\Scripts\activate
REM call .venv\Scripts\activate

REM PyInstaller 설치 확인
python -c "import PyInstaller" 2>nul || (
    echo PyInstaller가 없습니다. 설치 중...
    pip install pyinstaller
)

echo.
echo 빌드 시작 (몇 분 소요될 수 있습니다)...
python -m PyInstaller PDFChatbot.spec

if %errorlevel% equ 0 (
    echo.
    echo [완료] dist\PDFChatbot\PDFChatbot.exe 가 생성되었습니다.
    echo .env 파일을 exe와 같은 폴더에 복사해주세요.
) else (
    echo.
    echo [오류] 빌드에 실패했습니다.
)

pause
