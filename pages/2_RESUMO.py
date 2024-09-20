import warnings
from datetime import datetime
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as m
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import scipy
from matplotlib.pyplot import figure
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import streamlit as st
import plotly.express as px
from modulos.estacionaridade import *
from modulos.testes_estacionarios import *
from modulos.Credito import *
##########################################################################

st.title('SUPERINTENDÊNCIA DE ORÇAMENTO PÚBLICO')
st.write('---')
st.button('Atualizar')
st.subheader('RESUMO ANUAL DA DOTAÇÃO ORÇAMENTÁRIA - 2024')
st.write('##')
dados = pd.read_csv('data/SALDO_PESSOAL.csv')

#  Adicionando controle de radio para seleção de formato
formato_valores = st.sidebar.radio(
    "Selecione o formato dos valores:",
    ('Inteiro', 'Milhões (M)', 'Bilhões (B)')
)

# Função para formatar valores
def formatar_valor(valor, formato):
    if formato == 'Inteiro':
        return f'R$ {valor:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.') # Substituindo pontos por vírgulas e vírgulas por pontos
    elif formato == 'Milhões (M)':
        return f'R$ {valor/1e6:.1f}M'
    elif formato == 'Bilhões (B)':
        return f'R$ {valor/1e9:.1f}B'





# Metŕicas importantes
# Criação e APlicação das métricas
dotacao_inicial = dados['VALOR_DOTACAO_INICIAL'].sum() # Valor da dotação inicial
dotacao_atualizada = dados['VALOR_ATUALIZADO'].sum() # Valor da dotação inicial
diferencial = dotacao_atualizada - dotacao_inicial # Diferencial entre a dotação inicial e a dotação atualizada
empenhado_08 = dados['VALOR_EMPENHADO'].sum() # Valor empenhado até o Mês 2024.8
valor_previsto = dados['PREVISAO'].sum() # Valor previsto até o final do ano
somatorio_empenhado_previsto = empenhado_08 + valor_previsto # Somatório do valor empenhado até o mês 08 e o valor previsto até o final do ano
# Saldo é valor atualizado menos o total
saldo = dotacao_atualizada - somatorio_empenhado_previsto


# Exibindo métricas
# Exibindo métricas
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Dotação Inicial", value=formatar_valor(dotacao_inicial, formato_valores))
    col2.metric(label="Dotação Atualizada", value=formatar_valor(dotacao_atualizada, formato_valores))
    col3.metric(label="Suplementado", value=formatar_valor(diferencial, formato_valores), delta=f'{formatar_valor(dotacao_atualizada - dotacao_inicial, formato_valores)}', delta_color="normal")
st.write('---')

with st.container():
    st.subheader('Previsão Orçamentária:')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Valor Empenhado até mês 08", value=formatar_valor(empenhado_08, formato_valores))
    col2.metric(label="Previsão até o final do exercício", value=formatar_valor(valor_previsto, formato_valores))
    col3.metric(label="Total Empenhado + Previsão", value=formatar_valor(somatorio_empenhado_previsto, formato_valores))
    col4.metric(label="Saldo", value=formatar_valor(saldo, formato_valores), delta=f'-{formatar_valor(saldo, formato_valores)}', delta_color="normal")






# Fazer um for de tudo isso para cada UO do dataframe
# Converter UO e DESCRICAO_UO para str
dados['UO'] = dados['UO'].astype(str)
dados['DESCRICAO_UO'] = dados['DESCRICAO_UO'].astype(str)

st.write('##')
st.write('---')
st.write('##')

st.title('Por Unidade Orçamentária')
st.write('##')

# Adicionando campo de busca
busca_uo = st.text_input("Digite a Unidade Orçamentária que deseja encontrar:")
st.write('##')

# Agrupar dados por UO
grupos_uo = dados.groupby(['UO', 'DESCRICAO_UO'])
uos_saldo_negativo = []
uos_saldo_positivo = []
# Iterar sobre cada UO e calcular métricas

# Só mostrará no streamlit se o usuario clicar no checkbox ou digitar algo no campo de busca



for (uo, descricao_uo), grupo in grupos_uo:
    # Verificar se a busca corresponde à UO ou à descrição da UO
    if busca_uo.lower() in uo.lower() or busca_uo.lower() in descricao_uo.lower():
        # Criação e Aplicação das métricas
        dotacao_inicial = grupo['VALOR_DOTACAO_INICIAL'].sum() # Valor da dotação inicial
        dotacao_atualizada = grupo['VALOR_ATUALIZADO'].sum() # Valor da dotação inicial
        
        diferencial = dotacao_inicial - dotacao_atualizada # Diferencial entre a dotação inicial e a dotação atualizada
        if diferencial > 0:
            diferencial = 0
        elif diferencial < 0:
            diferencial = abs(diferencial)
        empenhado_08 = grupo['VALOR_EMPENHADO'].sum() # Valor empenhado até o Mês 2024.8
        valor_previsto = grupo['PREVISAO'].sum() # Valor previsto até o final do ano
        somatorio_empenhado_previsto = empenhado_08 + valor_previsto # Somatório do valor empenhado até o mês 08 e o valor previsto até o final do ano
        saldo = dotacao_atualizada - somatorio_empenhado_previsto # Saldo é valor atualizado menos o total

        if saldo < 0:
            uos_saldo_negativo.append((descricao_uo, saldo))
        elif saldo > 0:
            uos_saldo_positivo.append((descricao_uo, saldo))


        # COLOCAR TUDO ISSO DENTRO DE UM EXPANDER

        st.expander(f'Unidade Orçamentária: {descricao_uo}')
        # Exibindo métricas para cada UO
        st.write(f'### **UNIDADE ORÇAMENTÁRIA**: *{descricao_uo.lower().title()}*')
        # Determine the delta color based on the value of diferencial
        delta_color = "off" if diferencial == 0 else "normal"

        with st.container():
            st.subheader('Dotação Orçamentária:')
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="Dotação Inicial", value=formatar_valor(dotacao_inicial, formato_valores))
            col2.metric(label="Dotação Atualizada", value=formatar_valor(dotacao_atualizada, formato_valores))
            col3.metric(label="Suplementado", value=formatar_valor(diferencial, formato_valores), delta=f'{formatar_valor(diferencial, formato_valores)}', delta_color=delta_color)

        with st.container():
            st.subheader('Previsão Orçamentária:')
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="Valor Empenhado até mês 08", value=formatar_valor(empenhado_08, formato_valores))
            col2.metric(label="Previsão até o final do exercício", value=formatar_valor(valor_previsto, formato_valores))
            col3.metric(label="Total Empenhado + Previsão", value=formatar_valor(somatorio_empenhado_previsto, formato_valores))
            col4.metric(label="Saldo", value=formatar_valor(saldo, formato_valores), delta=f'{saldo:.2f}', delta_color="normal")
        st.write('---')
        # encerrar o primeiro for para começar o outro
        

# Só será executado quando o primeiro for acabar
# Resumo do Saldo das Unidades Orçamentárias
with st.container():
    st.subheader('Resumo do Saldo das Unidades Orçamentárias:')
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total de Unidades Orçamentárias", value=len(grupos_uo))
    col2.metric(label="Unidades Orçamentárias com Saldo Positivo", value=len(grupos_uo) - len(uos_saldo_negativo))
    col3.metric(label="Unidades Orçamentárias com Saldo Negativo", value=len(uos_saldo_negativo))

with st.expander("Visualizar Unidades Orçamentárias com Saldo Negativo"):
    col1, col2 = st.columns(2)
    for descricao_uo, saldo in uos_saldo_negativo:
        st.write(f'**{descricao_uo}**: {formatar_valor(saldo, formato_valores)}')
st.write('---')

with st.expander("Visualizar Unidades Orçament-aárias com Saldo Positivo"):
    for descricao_uo, saldo in uos_saldo_positivo:
        st.write(f'**{descricao_uo}**: {formatar_valor(saldo, formato_valores)}')





display_credits()


