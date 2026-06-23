# Previsão de Preços do Bitcoin com LSTM

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub stars](https://img.shields.io/github/stars/andreluizpedroso/forecast_btc?style=flat&logo=github)](https://github.com/andreluizpedroso/forecast_btc/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/andreluizpedroso/forecast_btc?logo=github)](https://github.com/andreluizpedroso/forecast_btc/commits/main)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://forecastbtc-jisdg7mfjdwzjngbr6suwq.streamlit.app/)

Este projeto utiliza Machine Learning para prever preços futuros do **Bitcoin** (`BTC-USD`) com base em dados históricos, utilizando redes neurais LSTM.  
Além disso, o projeto gera uma recomendação automática de **Compra** ou **Aguardar** baseada em médias móveis.

---

## 📚 Tecnologias Utilizadas

- Python 3.10+
- Streamlit
- FastAPI
- TensorFlow/Keras
- Scikit-learn
- yFinance
- Pandas
- Plotly
- SQLite3
- Docker

---

## 🎯 Objetivo do Projeto

- Coletar dados históricos do Bitcoin via Yahoo Finance e armazenar em um banco de dados SQLite.
- Treinar um modelo LSTM para previsão de preços.
- Exibir previsões e recomendações de compra ou espera com base em médias móveis.
- Disponibilizar o modelo por meio de uma API REST com FastAPI.
- Empacotar e rodar a API em Docker para facilitar o deploy.

---

## 💪 Processo Completo

### 1. Configuração Inicial (Ambiente Local)

```bash
git clone https://github.com/andreluizpedroso/forecast_btc.git
cd forecast_btc

python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

---

### 2. Estrutura de Arquivos

```
forecast_btc/
├── forecast_btc/
│   ├── api.py           # Rotas e contratos da API
│   ├── charts.py        # Gráficos Plotly
│   ├── config.py        # Configurações e caminhos
│   ├── data.py          # SQLite e Yahoo Finance
│   ├── forecasting.py   # Previsão e regras de negócio
│   └── model.py         # Carregamento do modelo LSTM
├── tests/               # Testes automatizados
├── app.py               # Entrada da interface Streamlit
├── main.py              # Entrada ASGI da API
├── finance.db           # Base de dados SQLite
├── modelo_lstm.h5       # Modelo LSTM treinado
├── Forecast.ipynb       # Experimentos e treinamento
└── Dockerfile           # Imagem da API
```

A interface e a API usam a mesma implementação de previsão. O tamanho da janela
é obtido diretamente do modelo salvo, evitando configurações divergentes.

---

## 🖥️ Interface com Streamlit

```bash
streamlit run app.py
```

Abra no navegador: [http://localhost:8501](http://localhost:8501)

---

## 🔢 Funcionalidades

- [x] Coleta e armazenamento de dados históricos em SQLite
- [x] Treinamento e salvamento de modelo LSTM
- [x] Interface Streamlit para previsão de até 30 dias
- [x] Recomendação de Compra ou Aguardar (MM20 e MM80)
- [x] Gráficos interativos com Plotly
- [x] API REST com FastAPI
- [x] Endpoint automático de previsão via yFinance
- [x] Empacotamento com Docker

---

## 📊 Exemplo de Uso (API FastAPI)

### Endpoints:
```bash
uvicorn main:app --reload
```
- `GET /` → Verifica status
- `POST /prever` → Envia JSON com pelo menos 10 valores históricos
- `POST /prever_auto` → Faz previsão automática com dados do yFinance

### Exemplo:

```json
POST /prever
{
  "historico": [61234.1, 61300.2, ..., 61150.2]
}
```

```json
Resposta:
{
  "preco_previsto": 61380.45
}
```

---

## 🐳 Docker

### 1. Build da imagem:
```bash
docker build -t forecast-btc-api .
```

### 2. Executar:
```bash
docker run -d -p 8000:8000 forecast-btc-api
```

Acesse: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧠 Modelo de Machine Learning

- Tipo: LSTM (Long Short-Term Memory)
- Arquitetura: 1 camada LSTM (64 unidades) + 1 densa
- Perda: MSE (Mean Squared Error)
- Otimizador: Adam
- Treinamento com 70% dos dados
- Previsão do próximo fechamento (1 dia à frente)

> **Aviso:** as previsões e o sinal por médias móveis são experimentais e não
> constituem recomendação financeira.

---

## ✅ Testes

```bash
python -m unittest discover -v
```

---

## 📅 Limitações

- A previsão é feita com base apenas no fechamento (univariada)
- Não considera variáveis externas (ex: volume, notícias)
- Previsão limitada ao próximo dia (em `/prever_auto`)
- Interface Streamlit limitada a 30 dias por design

---

## 🌐 Fonte de Dados

- [Yahoo Finance - Bitcoin (BTC-USD)](https://finance.yahoo.com/quote/BTC-USD)

---

## 👤 Autor

**Andre Luiz Pedroso**  
Projeto desenvolvido para a **Pós-Tech - Tech Challenge Fase 4 - Engenharia de Machine Learning**.

---

# 🚀 Vamos prever o futuro! 📈  
🔗 [Acesse o app online (Streamlit)](https://forecastbtc-jisdg7mfjdwzjngbr6suwq.streamlit.app/)
