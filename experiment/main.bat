@Echo Off
Pushd "%~dp0"
uv run --python 3.13 .\main.py %*
Popd