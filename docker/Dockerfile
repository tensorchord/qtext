FROM python:3.11-slim

WORKDIR /workspace

COPY qtext/ pyproject.toml /workspace/

RUN python -m pip install .

EXPOSE 8000

ENTRYPOINT [ "qtext" ]
