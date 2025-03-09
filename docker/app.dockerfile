FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 qpdf poppler-utils libmagic-dev tesseract-ocr libreoffice pandoc -y
