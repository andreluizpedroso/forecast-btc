import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==========================================
# Funções Auxiliares
# ==========================================

def carregar_dados_historicos(ativo):
    conn = sqlite3.connect('finance.db')
    df = pd.read_sql_query(
        "SELECT data, fechamento FROM historico WHERE ativo = ? ORDER BY data",
        conn, params=(ativo,)
    )
    conn.close()
    df['data'] = pd.to_datetime(df['data'])
    return df

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def predict_to_date(target_date, model, sequence_length, data_scaled, scaler, data):
    target_date = pd.to_datetime(target_date)
    last_known_date = pd.to_datetime(data['Date'].iloc[-1])
    num_days = (target_date - last_known_date).days
    if num_days <= 0:
        raise ValueError("A data deve ser futura em relação aos dados atuais.")
    prediction_list = data_scaled[-sequence_length:]
    for _ in range(num_days):
        x = prediction_list[-sequence_length:]
        x = x.reshape((1, sequence_length, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = scaler.inverse_transform(prediction_list.reshape(-1, 1))
    return prediction_list[-1][0]

def gerar_dados_completos(num_prediction, model, sequence_length, data, data_scaled, scaler):
    prediction_list = data_scaled[-sequence_length:]
    for _ in range(num_prediction):
        x = prediction_list[-sequence_length:]
        x = x.reshape((1, sequence_length, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    forecast_full = scaler.inverse_transform(prediction_list.reshape(-1, 1))
    forecast = forecast_full[-num_prediction:]
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()[1:]
    future_df = pd.DataFrame({ 'Date': future_dates, 'Close': forecast.flatten() })
    full_data = pd.concat([data[['Date', 'Close']], future_df], ignore_index=True)
    full_data['MM20'] = full_data['Close'].rolling(window=20).mean()
    full_data['MM80'] = full_data['Close'].rolling(window=80).mean()
    return full_data

def tomar_decisao(full_data, target_date):
    target_date = pd.to_datetime(target_date)
    full_data['Date_only'] = full_data['Date'].dt.date
    target_row = full_data[full_data['Date_only'] == target_date.date()]
    if target_row.empty:
        return "Data não encontrada."
    mm20 = target_row['MM20'].values[0]
    mm80 = target_row['MM80'].values[0]
    if np.isnan(mm20) or np.isnan(mm80):
        return "Não há dados suficientes para decisão."
    if mm20 > mm80:
        return "✅ Recomendacao: Comprar"
    else:
        return "⏳ Recomendacao: Aguardar"

def plotar_forecast(full_data, target_date):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full_data['Date'], y=full_data['Close'], mode='lines', name='Fechamento'))
    fig.add_trace(go.Scatter(x=full_data['Date'], y=full_data['MM20'], mode='lines', name='MM20'))
    fig.add_trace(go.Scatter(x=full_data['Date'], y=full_data['MM80'], mode='lines', name='MM80'))
    fig.add_shape(type="line", x0=pd.to_datetime(target_date), y0=full_data['Close'].min(),
                  x1=pd.to_datetime(target_date), y1=full_data['Close'].max(),
                  line=dict(color="Red", width=2, dash="dash"))
    fig.add_annotation(x=pd.to_datetime(target_date), y=full_data['Close'].max(),
                       text="Data escolhida", showarrow=True, arrowhead=1)
    fig.update_layout(title="Histórico + Forecast + Médias Móveis", xaxis_title="Data", yaxis_title="Preço",
                      template="plotly_white", width=1000, height=600, xaxis=dict(type='date'))
    return fig

# ==========================================
# Streamlit App
# ==========================================

st.set_page_config(page_title="Previsão Financeira Bitcoin", layout="wide")
st.title("📊 Previsão do Preço Futuro do Bitcoin")

# Imagem responsiva e centralizada
from PIL import Image

# Criar layout com 3 colunas
col1, col2, col3 = st.columns([1, 2, 1])

# Centralizar imagem na coluna do meio
with col2:
    image = Image.open("bitcoin_img.png")
    st.image(image, use_container_width=True)
    # st.markdown("### 📈 Escolha uma data à esquerda para iniciar a previsão!")

# Seletor de ativo (só um por enquanto)
ticker = 'BTC-USD'

# Carregar dados
data = carregar_dados_historicos(ticker)
data.rename(columns={'data': 'Date', 'fechamento': 'Close'}, inplace=True)
data.reset_index(drop=True, inplace=True)

# Preparar dados
train_size = int(len(data) * 0.7)
scaler = MinMaxScaler()
scaler.fit(data[['Close']].iloc[:train_size])
data_scaled = scaler.transform(data[['Close']])
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]
sequence_length = 10
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Carregar ou treinar modelo
if os.path.exists('modelo_lstm.h5'):
    model_lstm = load_model('modelo_lstm.h5', compile=False)
else:
    model_lstm = Sequential([
        LSTM(64, input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)
    model_lstm.save('modelo_lstm.h5')

# Input do usuario
st.sidebar.header("Configuração de Previsão")

# Definir limites de datas permitidas
last_known_date = pd.to_datetime(data['Date'].iloc[-1])
start_prediction_date = last_known_date + pd.Timedelta(days=1)
end_prediction_date = last_known_date + pd.Timedelta(days=30)

# Criar lista de datas disponíveis - apenas datas (sem hora)
date_range = pd.date_range(start=start_prediction_date, end=end_prediction_date).date

# Barra de seleção de data
target_date = st.sidebar.select_slider(
    "Escolha uma data futura para previsão:",
    options=date_range,
    value=date_range[0]
)

if st.sidebar.button("Gerar Previsão"):
    try:
        preco_previsto = predict_to_date(target_date, model_lstm, sequence_length, data_scaled, scaler, data)
        full_data = gerar_dados_completos(30, model_lstm, sequence_length, data, data_scaled, scaler)
        fig = plotar_forecast(full_data, target_date)
        recomendacao = tomar_decisao(full_data, target_date)

        st.subheader(f"Resultado para {target_date}")
        st.metric(label="Preço Previsto (USD)", value=f"$ {preco_previsto:,.2f}")
        st.success(recomendacao)
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Erro: {e}")
