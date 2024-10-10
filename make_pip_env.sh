#!/bin/bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
deactivate
