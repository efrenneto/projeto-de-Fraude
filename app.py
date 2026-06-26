import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Carregando o modelo
with open('modelo_fraude.pkl', 'rb') as file:
    fraud_model = pickle.load(file)

# Interface gráfica
st.title("Detecção de Fraude")
st.write("Este aplicativo detecta transações fraudulentas com base em dados históricos.")

# Campos para entrada de dados
st.header("Detalhe Transação")

valor_transacao = st.number_input("Valor da Transação", min_value=0, max_value=10000, value=50)
tempo_conta = st.number_input("Tempo da Conta (meses)", min_value=0, max_value=120, value=6)
num_transacoes = st.number_input("Número de Transações", min_value=0, max_value=1000, value=3)

pais_origem_opcoes = {
    "Brasil":0, 
    "EUA": 1,
    "Outros": 2
}

pais_origem_escolhido = st.selectbox("Pais de Origem", list(pais_origem_opcoes.keys()))
pais_origem = pais_origem_opcoes[pais_origem_escolhido]

# Botão para previsão
if st.button("Analisar se é Fraude"):

    input_array = np.array([[valor_transacao, tempo_conta, num_transacoes, pais_origem]])

    pred = fraud_model.predict(input_array)    
    proba = fraud_model.predict_proba(input_array)
    st.write("Resultado Analise")
    if pred[0] == 1:
        st.error("Transação suspeita! Possível fraude detectada.")
    else:
        st.success("Transação parece legítima.")

    st.write(f"** Probalidade não fraude: {proba[0][0]:.2f}")
    st.write(f"** Probalidade fraude: {proba[0][1]:.2f}")
else:
    st.write("Informe os detalhes e clique verificar")


