pushd "$(dirname "$0")"
uv run --python 3.13 ./main.py "$@"
popd