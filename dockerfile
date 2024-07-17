FROM python:3.11
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENV AWS_ACCESS_KEY_ID=AKIA4MTWLPBHJBPD3VWJ
ENV AWS_SECRET_ACCESS_KEY=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
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0" ]
