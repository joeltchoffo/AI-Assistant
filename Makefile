.PHONY: install ingest query ui clean

install:
\tpython -m pip install --upgrade pip
\tpip install -r requirements.txt

ingest:
\tpython -m apps.cli ingest "data/*.pdf"

query:
\tpython -m apps.cli query "$(q)" -k $(k)

ui:
\tstreamlit run apps/streamlit_app.py

clean:
\trm -rf indices/faiss/*
