
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Portofolio Ajik mengenai Prediksi Stok Harga dengan algoritma prophet')

st.write("""Facebook Prophet Model adalah algoritma yang digunakan untuk
melakukan prediksi dari data yang berbentuk time-series yang berdasar dari model
aditif dimana trend yang bersifat non-linear akan dicocokan dalam deret waktu secara tahunan, mingguan dan harian, dengan efek liburan (Taylor dan Letham, 2018).
Algoritma ini berfungsi dengan sangat baik dengan data time-series yang memiliki
efek musiman yang kuat dan dataset yang punya banyak data. Prophet bersifat robust terhadap data yang hilang dan pergerakan trend, dan juga dapat menangani
outliers dengan baik.""")

stocks = ('ABMM.JK', 'BBCA', 'BBRI')
selected_stock = st.selectbox('Saya Menggunakan 3 dataset yaitu (BBCA, BBRI, ABMM.JK)', stocks)

n_years = st.slider('Prediksi Per Tahun:', 1, 2)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Sebentar Ya, Data Baru dimuat...')
data = load_data(selected_stock)
data_load_state.text('Data Selesai Dimuat!')

st.subheader('Tampilan Data Pada Urutan Awal')
st.write(data.head())

st.subheader('Tampilan Data Pada Urutan Terakhir')
st.write(data.tail())


st.subheader('Descriptive Statistics')
st.write(data.describe())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data dengan Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
