@echo off
REM 배치 파일 시작 시 코드 페이지를 UTF-8로 설정
chcp 65001 >nul
REM setup_and_run.bat - 패키지 설치 후 Python 스크립트 실행

REM Python이 설치되어 있는지 확인
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python이 설치되어 있지 않습니다. Python을 설치하고 PATH에 추가하세요.
    pause
    exit /b 1
)

REM requirements.txt 파일이 있는지 확인
IF NOT EXIST "requirements.txt" (
    echo requirements.txt 파일이 존재하지 않습니다. 파일을 생성하세요.
    pause
    exit /b 1
)

REM 패키지 설치
echo 패키지를 설치 중입니다...
pip install --no-cache-dir -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo 패키지 설치 중 오류가 발생했습니다.
    pause
    exit /b 1
)

REM spaCy 언어 모델 다운로드
echo spaCy 언어 모델을 다운로드 중입니다...
python -m spacy download en_core_web_sm
IF %ERRORLEVEL% NEQ 0 (
    echo spaCy 언어 모델 다운로드 중 오류가 발생했습니다.
    pause
    exit /b 1
)

REM Python 스크립트 실행
echo Python 스크립트를 실행 중입니다...
python real-time_trends_Global.py
IF %ERRORLEVEL% NEQ 0 (
    echo Python 스크립트 실행 중 오류가 발생했습니다.
    pause
    exit /b 1
)

echo 모든 작업이 완료되었습니다.