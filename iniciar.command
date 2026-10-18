#!/bin/zsh
cd -- "$(dirname -- "$0")"
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
