import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

st.write("# Predição de câncer de mama.")
st.write("## Exemplo com perímetro, área")

st.sidebar.write("## Parâmetros")
perimetro = st.sidebar.slider("perimetro", -1.4, 5.2, 2.3, 0.1)
area = st.sidebar.slider("area", -1.5, 4.2, 2.5, 0.1)

with open("objetos.pkl", "rb") as arquivo:
  ss, dtc = pickle.load(arquivo)

estrutura = { 'perimetro': perimetro,
              'area': area }

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
