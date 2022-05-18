import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

st.write("# Predição de câncer de mama.")
st.write("## Exemplo com raio, perímetro, área, concavidade, junta côncava")

st.sidebar.write("## Parâmetros")
raio = st.sidebar.slider("raio", 2.0, 4.8, 3.0, 0.1)
perimetro = st.sidebar.slider("perímetro", 1.0, 4.3, 2.8, 0.1)
area = st.sidebar.slider("área", 1.0, 6.0, 3.5, 0.1)
concavidade = st.sidebar.slider("Concavidade", 1.0, 5.0, 3.1, 0.1)
junta_concava = st.sidebar.slider("Junta côncava", 1.0, 2.5, 2.0, 0.1)

with open("objetos.pkl", "rb") as arquivo:
  ss, dtc = pickle.load(arquivo)

estrutura = { 'raio': raio,
              'perímetro': perimetro,
              'area': area,
              'concavidade': concavidade,
              'junta côncava': junta_concava }

df= pd.DataFrame(estrutura, index=[0])

st.write("### Parâmetros de entrada")
st.write(df)

df = ss.transform(df)
st.write(df)

predicao = dtc.predict(df)
st.write(f"O diagnóstico dessa pessoa é: **{predicao[0]}**")
#st.write(predicao)

predicao = dtc.predict_proba(df)
predicao = pd.DataFrame(predicao)
predicao.rename({
    0: 'B',
    1: 'M'
}, axis=1, inplace=True)

st.write("Probabilidades")
st.write(predicao)
