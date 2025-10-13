pushd "$(dirname "$0")"
source ../.venv/bin/activate
uv run --python 3.13 ./main.py "$@"
popd