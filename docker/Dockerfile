FROM python:3.11-slim

WORKDIR /workspace

COPY qtext /workspace/qtext
COPY pyproject.toml README.md /workspace/
COPY docker/config.json /root/.config/qtext/config.json

RUN python -m pip install .

EXPOSE 8000

ENTRYPOINT [ "qtext" ]
