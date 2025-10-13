@Echo Off
Pushd "%~dp0"
..\.venv\Scripts\activate.bat
uv run --python 3.13 .\main.py %*
Popd