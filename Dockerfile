# Use uma imagem leve do Python
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos necessários
COPY app.py app.py
COPY requirements.txt requirements.txt

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta padrão do Streamlit
EXPOSE 8501

# Define comando de inicialização
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
