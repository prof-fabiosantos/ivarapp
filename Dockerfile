FROM python:3.11-slim

# Instalar dependências de sistema necessárias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copiar arquivos
COPY app.py app.py
COPY requirements.txt requirements.txt

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Expor porta do Streamlit
EXPOSE 8501

# Rodar o app Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

