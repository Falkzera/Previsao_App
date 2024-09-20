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
# Configurações Página
st.set_page_config(layout="wide", page_icon="🤖", page_title="Previsão do Gasto Empenhado")
# Configurações Sidebar
st.sidebar.title('Configurações')
st.sidebar.write('Selecione as opções gráficas.')

###############################################################################
# Carregando e filtrando a Base
@st.cache_data
def load_data():
    df = pd.read_csv('data/desepesa_2018_2024.csv', encoding='latin1', sep=';')
    df1 = pd.read_excel('data/despesa_2014_2017.xlsx')
    df_test = df[['ANO', 'MES', 'PODER', 'UO', 'UG', 'FONTE_MAE', 'NATUREZA3', 'VALOR_EMPENHADO']]
    df_test1 = df1[['ANO', 'MES', 'PODER', 'UO', 'UG', 'FONTE_MAE', 'NATUREZA3', 'VALOR_EMPENHADO']]
    df_test = pd.concat([df_test, df_test1])
    return df_test

# Carregar os dados
dados = load_data()

########################################################################################
# Pré-processamento
sigla = pd.read_csv('data/sigla.csv')
sigla['UO'] = sigla['UO'].astype('object')
sigla_nat3 = pd.read_csv('data/sigla_nat3.csv')

# em sigla contem a coluna UO que é a Unidade Orçamentária e contem a coluna PODER que é a sigla de UO. Quero que essa coluna de PODER va para o df dados e crie uma coluna UO_sigla
dados = dados.merge(sigla[['UO', 'SIGLA']], on='UO', how='left')
dados = dados.merge(sigla_nat3[['NATUREZA3', 'NATUREZA3_DESC']], on='NATUREZA3', how='left')

# Renomear a coluna 'PODER' para 'UO_sigla'
dados.rename(columns={'SIGLA': 'UO_sigla'}, inplace=True)

dados['ANO_MES'] = dados['ANO'].astype(str) + '-' + dados['MES'].astype(str)
dados['ANO_MES'] = pd.to_datetime(dados['ANO_MES'], format='%Y-%m')
dados = dados.sort_values(by=['ANO', 'MES']).reset_index(drop=True)
convertendo_obj = ['ANO', 'MES', 'PODER', 'UO', 'UG', 'FONTE_MAE', 'NATUREZA3']
for column in convertendo_obj:
    dados[column] = dados[column].astype('object')

# dados = dados.dropna()
dados = dados[dados['VALOR_EMPENHADO'] > 1]
dados.set_index('ANO_MES', inplace=True)  # Setando o index e removendo a coluna ANO_MES
# Filtrar por poder
dados = dados[dados['PODER'] == 'EXE']
dados = dados[dados['NATUREZA3'] == 331000000]

# Remover de dadaos 2024.09
dados = dados.loc[dados.index != '2024-09']

# Função para criar lista de DataFrames por UO e UO_sigla
def criar_lista_dataframes_por_uo(dados):
    lista_dataframes = []
    for uo in dados['UO'].unique():
        df_uo = dados[dados['UO'] == uo].copy()
        df_uo = df_uo.groupby('ANO_MES')['VALOR_EMPENHADO'].sum().reset_index()
        df_uo['UO'] = uo
        df_uo['UO_sigla'] = dados[dados['UO'] == uo]['UO_sigla'].iloc[0]
        lista_dataframes.append(df_uo)
    return lista_dataframes

# Criar lista de DataFrames
lista_dataframes = criar_lista_dataframes_por_uo(dados)


###############################################################################################


st.title('SUPERINTENDÊNCIA DE ORÇAMENTO PÚBLICO')
st.button('Atualizar')
st.subheader('Previsão automática do Gasto Empenhado')

# Exibir a quantidade de DataFrames criados
st.subheader(f'Quantidade de previsões: {len(lista_dataframes)}.')



###############################################################################################
def prever_e_plotar(lista_dataframes, escolha_periodo=5):
    resultados_combinados = pd.DataFrame()
    
    for df in lista_dataframes:
        try:
            # Garantir que DATA_MES esteja no índice
            df.set_index('ANO_MES', inplace=True)
            # Logaritimo
            df['VALOR_EMPENHADO_log'] = np.log1p(df['VALOR_EMPENHADO'])
    
            modelo_v4 = ExponentialSmoothing(np.asarray(df['VALOR_EMPENHADO_log']), 
                                             trend='multiplicative', 
                                             seasonal='additive', 
                                             seasonal_periods=12).fit()
            
            # Realizar previsões
            previsoes = modelo_v4.forecast(escolha_periodo)
            
            # Criar série de datas para as previsões
            # previsoes = np.expm1(previsoes)
            ultima_data = df.index[-1] # Última data do dataset
            datas_previsoes = pd.date_range(start=ultima_data, periods=escolha_periodo, freq='M')
            serie_previsoes = pd.Series(previsoes, index=datas_previsoes)
            
            # Adicionar previsões ao DataFrame combinado
            resultados_combinados[df['UO_sigla'].iloc[0]] = serie_previsoes
            
            # Calcular MAE
            y_true = df['VALOR_EMPENHADO_log'][-escolha_periodo:]
            y_pred = serie_previsoes[:len(y_true)]
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = sqrt(mean_squared_error(y_true, y_pred))




            fig = go.Figure()
            fig.add_trace(go.Scatter (x=df.index, y=df['VALOR_EMPENHADO_log'], mode='lines', name='Valor Empenhado'))
            fig.add_trace(go.Scatter (x=serie_previsoes.index, y=serie_previsoes, fillcolor='Red', mode='lines', name='Previsão', line=dict(dash='dash', color='red')))
            fig.update_layout(title=f'Previsão de Gasto Empenhado para {df["UO_sigla"].iloc[0]}', xaxis_title='Data', yaxis_title='Valor Empenhado')
            st.plotly_chart(fig)



            st.write(f'[{df["UO"].iloc[0]} : {df["UO_sigla"].iloc[0]}] - MAE: {mae:.4f} - MAPE: {mape:.4f} - RMSE: {rmse:.4f}')
        
        except Exception as e:
            st.write(f"Não foi possível realizar a previsão para [{df['UO'].unique()[0]} : {df['UO_sigla'].unique()[0]}] : Erro: {e}")
            pass
    
    # Formatar o índice de data para mostrar apenas ANO e MES
    resultados_combinados.index = resultados_combinados.index.to_period('M').strftime('%Y-%m')
    
    return resultados_combinados

# Exemplo de uso
resultados_combinados = prever_e_plotar(lista_dataframes)
st.write('Resultados Combinados')
# inverter o logaritimo de resultados_combinados
resultados_combinados = np.expm1(resultados_combinados)
st.write(resultados_combinados)

# if st.button('Salvar previsões'):
#     resultados_combinados.to_excel('data/previsoes_2.xlsx', index=True)

display_credits()