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
from modulos.filtros import *
from modulos.carregamento import *
##########################################################################
# Configurações Página
st.set_page_config(layout="wide", page_icon=":bar_chart:", page_title="Análise do Gasto Empenhado")
# Configurações Sidebar
st.sidebar.title('Configurações')
st.sidebar.write('Selecione as opções gráficas.')
# Botão de atualizar

###############################################################################

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

sigla = pd.read_csv('data/sigla.csv')
sigla['UO'] = sigla['UO'].astype('object')
sigla_nat3 = pd.read_csv('data/sigla_nat3.csv')
dados = dados.merge(sigla[['UO', 'SIGLA']], on='UO', how='left')
dados = dados.merge(sigla_nat3[['NATUREZA3', 'NATUREZA3_DESC']], on='NATUREZA3', how='left')
dados.rename(columns={'SIGLA': 'UO_sigla'}, inplace=True)
dados['ANO_MES'] = dados['ANO'].astype(str) + '-' + dados['MES'].astype(str)
dados['ANO_MES'] = pd.to_datetime(dados['ANO_MES'], format='%Y-%m')
dados = dados.sort_values(by=['ANO', 'MES']).reset_index(drop=True)
convertendo_obj = ['ANO', 'MES', 'PODER', 'UO', 'UG', 'FONTE_MAE', 'NATUREZA3']
for column in convertendo_obj:
    dados[column] = dados[column].astype('object')
dados = dados[dados['VALOR_EMPENHADO'] > 1]
dados.set_index('ANO_MES', inplace=True)  # Setando o index e removendo a coluna
dados = dados.loc[dados.index != '2024-09']
############################################################################################
if st.sidebar.checkbox('Ativar Siglas', value=True):
    dados['UO'] = dados['UO_sigla']
    dados['NATUREZA3'] = dados['NATUREZA3_DESC']
else:
    dados['UO'] = dados['UO']
    dados['NATUREZA3'] = dados['NATUREZA3']


dados = filtros_usuario(dados)

#####################################################################################################
# Metrícas
total_gasto = dados['VALOR_EMPENHADO'].sum() # Total gasto no período
gastos_por_natureza = dados.groupby('NATUREZA3')['VALOR_EMPENHADO'].sum()  # Gasto total por natureza
porcentagem_gastos = (gastos_por_natureza / total_gasto * 100)  # Porcentagem dos gastos por natureza

# Criar um DataFrame para exibir o ranking por natureza
ranking_gastos_natureza = pd.DataFrame({
    'Natureza': gastos_por_natureza.index.astype(str).str.replace(',', ''),  # Converter para string e remover vírgulas
    'Gasto (R$)': gastos_por_natureza.values,
    'Porcentagem (%)': porcentagem_gastos.values
}).sort_values(by='Gasto (R$)', ascending=False)  # Ordenar por gasto em ordem decrescente

# Formatação dos valores de gasto e porcentagem para exibição
ranking_gastos_natureza['Gasto (R$)'] = ranking_gastos_natureza['Gasto (R$)'].apply(lambda x: f'R$ {x/1e9:.1f}B' if x >= 1e9 else f'R$ {x/1e6:.1f}M')
ranking_gastos_natureza['Porcentagem (%)'] = ranking_gastos_natureza['Porcentagem (%)'].apply(lambda x: f'{x:.2f}%')

# Calcular a porcentagem dos gastos por unidade orçamentária
gastos_por_uo = dados.groupby('UO')['VALOR_EMPENHADO'].sum()  # Gasto total por unidade orçamentária
porcentagem_gastos_uo = (gastos_por_uo / total_gasto * 100)  # Porcentagem dos gastos por unidade orçamentária

# Criar um DataFrame para exibir o ranking por unidade orçamentária
ranking_gastos_uo = pd.DataFrame({
    'Unidade Orçamentária': gastos_por_uo.index.astype(str).str.replace(',', ''),  # Converter para string e remover vírgulas
    'Gasto (R$)': gastos_por_uo.values,
    'Porcentagem (%)': porcentagem_gastos_uo.values
}).sort_values(by='Gasto (R$)', ascending=False)  # Ordenar por gasto em ordem decrescente

# Formatar os valores de gasto para exibição
ranking_gastos_uo['Gasto (R$)'] = ranking_gastos_uo['Gasto (R$)'].apply(lambda x: f'R$ {x/1e9:.1f}B' if x >= 1e9 else f'R$ {x/1e6:.1f}M')
ranking_gastos_uo['Porcentagem (%)'] = ranking_gastos_uo['Porcentagem (%)'].apply(lambda x: f'{x:.2f}%')

# Calcular a porcentagem dos gastos por unidade orçamentária e tipo de natureza
gastos_por_uo_natureza = dados.groupby(['UO', 'NATUREZA3'])['VALOR_EMPENHADO'].sum().unstack().fillna(0)  # Gasto total por unidade orçamentária e tipo de natureza
porcentagem_gastos_uo_natureza = (gastos_por_uo_natureza / total_gasto * 100)  # Porcentagem dos gastos por unidade orçamentária e tipo de natureza

# Criar um DataFrame para exibir o ranking por unidade orçamentária e tipo de natureza
ranking_gastos_uo_natureza = gastos_por_uo_natureza.copy()
ranking_gastos_uo_natureza['Total Gasto (R$)'] = gastos_por_uo_natureza.sum(axis=1)
ranking_gastos_uo_natureza = ranking_gastos_uo_natureza.sort_values(by='Total Gasto (R$)', ascending=False)  # Ordenar por gasto total em ordem decrescente

# Formatar os valores de gasto para exibição
def formatar_valor(valor):
    return f'R$ {valor/1e9:.1f}B' if valor >= 1e9 else f'R$ {valor/1e6:.1f}M'

ranking_gastos_uo_natureza = ranking_gastos_uo_natureza.applymap(formatar_valor)
ranking_gastos_uo_natureza.index = ranking_gastos_uo_natureza.index.astype(str).str.replace(',', '')
#######################################################################################################################################################################
# Agrupando os dados
dados = dados.drop(columns=['ANO', 'MES'])
dados = dados.groupby('ANO_MES')['VALOR_EMPENHADO'].sum().reset_index()
dados.set_index('ANO_MES', inplace=True)  # Setando o index novamente após o agrupamento
############################################################################################
# Layout Página
st.title('SUPERINTENDÊNCIA DE ORÇAMENTO PÚBLICO')
st.button('Atualizar')
st.subheader('Estudos e Projeções do Gasto Empenhado')
st.write('Análise da série temporal: 2014 - 2024.')
st.title('Gráfico da Série Temporal')
st.write('Selecione o Périodo de Análise.')

# Filtro temporal
min_date = dados.index.min().to_pydatetime()
max_date = dados.index.max().to_pydatetime()
data_inicio, data_fim = st.slider(
    'Selecione o Período',
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM")
# Filtrar os dados com base no período selecionado
dados = dados.loc[data_inicio:data_fim]
#######################################################################################################################################################################
# Criação e APlicação das métricas
total_gasto = dados['VALOR_EMPENHADO'].sum() # Total gasto no período
minimo_historico = dados['VALOR_EMPENHADO'].min() # Mínimo histórico
maximo_historico = dados['VALOR_EMPENHADO'].max() # Máximo histórico
dados['media_movel'] = dados['VALOR_EMPENHADO'].rolling(window=12).mean() # Média Móvel de 12 meses significa que estamos considerando o gasto médio dos últimos 12 meses
# Calcular Desvio padrão móvel (Rolling Standard Deviation) para 12 meses, dando o resultado em CV, coeficiente de variação
dados['desvio_padrao'] = dados['VALOR_EMPENHADO'].rolling(window=12).std() / dados['media_movel'] * 100
# Exibir as métricas
with st.container():
    st.write('Métricas da Série Temporal:')
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label="Mínimo Histórico", value=f'R$ {minimo_historico/1e9:.1f}B' if minimo_historico >= 1e9 else f'R$ {minimo_historico/1e6:.1f}M')
    col2.metric(label="Máximo Histórico", value=f'R$ {maximo_historico/1e9:.1f}B' if maximo_historico >= 1e9 else f'R$ {maximo_historico/1e6:.1f}M')
    col3.metric(label="Total Gasto", value=f'R$ {total_gasto/1e9:.1f}B' if total_gasto >= 1e9 else f'R$ {total_gasto/1e6:.1f}M')
    col4.metric(label="Média Móvel", value=f'R$ {dados["media_movel"].iloc[-1]/1e9:.1f}B' if dados["media_movel"].iloc[-1] >= 1e9 else f'R$ {dados["media_movel"].iloc[-1]/1e6:.1f}M')
    col5.metric(label="Desvio Padrão (%)", value=f'{dados["desvio_padrao"].iloc[-1]:.2f}%')
#######################################################################################################################################################################
# Adicionar checkboxes para ativar ou desativar a visualização dos pontos de mínimo e máximo e as legendas dos governadores
show_min_max = st.sidebar.checkbox('Mostrar Pontos de Mínimo e Máximo', value=True) # Mostrar pontos de mínimo e máximo
show_governors = st.sidebar.checkbox('Mostrar Legendas dos Governadores', value=False) # Mostrar legendas dos governadores
frequencia_pontos = st.sidebar.selectbox('Frequência dos Pontos no Gráfico (Meses)', ['Nenhum Ponto'] + list(range(1, 13)), index=0) # Frequência dos pontos no gráfico

# Encontrar os mínimos e máximos para cada ano
dados['Ano'] = dados.index.year
min_max_per_year = dados.groupby('Ano')['VALOR_EMPENHADO'].agg(['min', 'max'])

# Criar o gráfico com plotly
fig = go.Figure()
# Adicionar a linha do gráfico para cada período de governador
if show_governors:
    # Período Teotônio Vilela
    dados_teotonio = dados['2014-01':'2014-12']
    fig.add_trace(go.Scatter(x=dados_teotonio.index, y=dados_teotonio['VALOR_EMPENHADO'], mode='lines', name='Teotônio Vilela', line=dict(color='blue')))

    # Período Renan Filho
    dados_renan = dados['2015-01':'2022-03']
    fig.add_trace(go.Scatter(x=dados_renan.index, y=dados_renan['VALOR_EMPENHADO'], mode='lines', name='Renan Filho', line=dict(color='orange')))

    # Período Paulo Dantas
    dados_paulo = dados['2022-04':]
    fig.add_trace(go.Scatter(x=dados_paulo.index, y=dados_paulo['VALOR_EMPENHADO'], mode='lines', name='Paulo Dantas', line=dict(color='green')))
else:
    # Adicionar a linha do gráfico sem distinção de governador
    fig.add_trace(go.Scatter(x=dados.index, y=dados['VALOR_EMPENHADO'], mode='lines', name='Valor Empenhado'))

# Adicionar pontos de destaque nos mínimos e máximos para cada ano, se selecionado
if show_min_max:
    for ano, row in min_max_per_year.iterrows():
        min_val = row['min']
        max_val = row['max']
        min_date = dados[(dados['Ano'] == ano) & (dados['VALOR_EMPENHADO'] == min_val)].index[0]
        max_date = dados[(dados['Ano'] == ano) & (dados['VALOR_EMPENHADO'] == max_val)].index[0]

        # Adicionar ponto mínimo
        fig.add_trace(go.Scatter(
            x=[min_date], y=[min_val], mode='markers+text', name='Mínimo',
            marker=dict(color='red', size=10),
            text=[f'R$ {min_val/1e9:.1f}B' if min_val >= 1e9 else f'R$ {min_val/1e6:.1f}M'],
            textposition='bottom center',
            showlegend=False  # Não mostrar na legenda lateral
        ))

        # Adicionar ponto máximo
        fig.add_trace(go.Scatter(
            x=[max_date], y=[max_val], mode='markers+text', name='Máximo',
            marker=dict(color='green', size=10),
            text=[f'<b>R$ {max_val/1e9:.1f}B</b>' if max_val >= 1e9 else f'<b>R$ {max_val/1e6:.1f}M</b>'],
            textposition='top center',
            showlegend=False  # Não mostrar na legenda lateral
        ))

    # Adicionar legendas genéricas para mínimo e máximo
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', name='Mínimo',
        marker=dict(color='red', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', name='Máximo',
        marker=dict(color='green', size=10)
    ))

# Adicionar pontos no gráfico com a frequência selecionada, se não for "Nenhum Ponto"
if frequencia_pontos != 'Nenhum Ponto':
    frequencia_pontos = int(frequencia_pontos)
    pontos_frequencia = dados[dados.index.month % frequencia_pontos == 0]
    fig.add_trace(go.Scatter(
        x=pontos_frequencia.index, y=pontos_frequencia['VALOR_EMPENHADO'], mode='markers+text', name=f'Pontos a Cada {frequencia_pontos} Meses',
        marker=dict(color='purple', size=8),
        text=[f'R$ {val/1e9:.1f}B' if val >= 1e9 else f'R$ {val/1e6:.1f}M' for val in pontos_frequencia['VALOR_EMPENHADO']],
        textposition='top center',
        showlegend=False  # Não mostrar na legenda lateral
    ))

# Atualizar layout do gráfico
fig.update_layout(title='Valor Empenhado com Destaques nos Mínimos e Máximos por Ano',
                  xaxis_title='Data',
                  yaxis_title='Valor Empenhado',
                  legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)))

st.plotly_chart(fig)
#####################################################################################################
# st.expander para mostrar os detalhes das despesas
with st.expander("Detalhamento de Despesa"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Ranking de Gastos por Natureza:")
        st.dataframe(ranking_gastos_natureza)
    
    with col2:
        st.write("Ranking de Gastos por Unidade Orçamentária:")
        st.dataframe(ranking_gastos_uo)
    
    with col3:
        st.write("Ranking de Gastos por Unidade Orçamentária e Tipo de Natureza:")
        st.dataframe(ranking_gastos_uo_natureza)
#####################################################################################################
#####################################################################################################
st.write('---')
#####################################################################################################
# Previsão do modelo
# Transformnado em logaritmo

st.title('Previsão do Valor Empenhado')

# Teste da estacionaridade da serie temporal
st.subheader('Testando Estacionaridade da série temporal')
teste_estacionaridade = dsa_testa_estacionaridade(dados['VALOR_EMPENHADO'])
if teste_estacionaridade:
    st.success('A série temporal é estacionária.')
else:
    st.error('A série temporal não é estacionária por padrão.')

# Transformando a série temporal em logaritmo - Estacionaridade Manual
dados['VALOR_EMPENHADO_log'] = np.log(dados['VALOR_EMPENHADO'])

####################################################################################################
# S.O.L.I.D - PREVISÃO, GŔAFICO, LEGENDAS E METRICAS
# Função para calcular métricas
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from io import BytesIO

# Função para calcular métricas
def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, mse, rmse, mape

# Função para plotar gráficos na escala original e retornar dataframes
def plotar_grafico(df_treino, df_valid, previsoes, titulo, previsoes_futuras=None):
    fig = go.Figure()

    # Adicionar a linha dos dados de treino (escala original)
    fig.add_trace(go.Scatter(
        x=df_treino.index, 
        y=np.exp(df_treino['VALOR_EMPENHADO_log']), 
        mode='lines', 
        name='Dados de Treino'
    ))

    # Adicionar a linha dos dados de validação (escala original)
    fig.add_trace(go.Scatter(
        x=df_valid.index, 
        y=np.exp(df_valid['VALOR_EMPENHADO_log']), 
        mode='lines', 
        name='Dados de Validação'
    ))

    # Adicionar a linha das previsões (escala original)
    fig.add_trace(go.Scatter(
        x=df_valid.index, 
        y=np.exp(previsoes), 
        mode='lines', 
        name=titulo
    ))

    # Adicionar a linha das previsões futuras, se fornecida (escala original)
    if previsoes_futuras is not None:
        fig.add_trace(go.Scatter(
            x=previsoes_futuras.index, 
            y=np.exp(previsoes_futuras), 
            mode='lines', 
            name='Previsões Futuras', 
            line=dict(color='blue', dash='dash')
        ))

    # Atualizar o layout do gráfico
    fig.update_layout(
        title=titulo,
        xaxis_title='Data',
        yaxis_title='Valor Empenhado',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)),
    )

    # Exibir o gráfico com Streamlit
    st.plotly_chart(fig)

    # Criar dataframes para download
    previsoes_df = pd.DataFrame({
        'Data': pd.concat([pd.Series(df_valid.index), pd.Series(previsoes_futuras.index)]).reset_index(drop=True) if previsoes_futuras is not None else pd.Series(df_valid.index).reset_index(drop=True),
        'Previsoes': pd.concat([np.exp(previsoes), np.exp(previsoes_futuras)]).reset_index(drop=True) if previsoes_futuras is not None else np.exp(previsoes).reset_index(drop=True)
    })

    serie_historica_completa_df = pd.DataFrame({
        'Data': pd.concat([pd.Series(df_treino.index), pd.Series(df_valid.index), pd.Series(previsoes_futuras.index)]).reset_index(drop=True) if previsoes_futuras is not None else pd.concat([pd.Series(df_treino.index), pd.Series(df_valid.index)]).reset_index(drop=True),
        'Valores': pd.concat([np.exp(df_treino['VALOR_EMPENHADO_log']), np.exp(df_valid['VALOR_EMPENHADO_log']), np.exp(previsoes_futuras)]).reset_index(drop=True) if previsoes_futuras is not None else pd.concat([np.exp(df_treino['VALOR_EMPENHADO_log']), np.exp(df_valid['VALOR_EMPENHADO_log'])]).reset_index(drop=True)
    })

    return previsoes_df, serie_historica_completa_df

# Função para baixar dataframes como Excel
def baixar_excel(df, nome_arquivo):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()  # Use close() instead of save()
    processed_data = output.getvalue()
    return processed_data

# Classe base para modelos de previsão
class ModeloPrevisao:
    def __init__(self, df_treino, df_valid):
        self.df_treino = df_treino
        self.df_valid = df_valid

    def treinar(self):
        raise NotImplementedError

    def prever(self, mostrar_previsao_futura=False, escolha_periodo=0):
        raise NotImplementedError

    def exibir_metricas(self, y_true, y_pred):
        mae, mse, rmse, mape = calcular_metricas(y_true, y_pred)
        with st.container():
            st.write('Métricas do Modelo Exponential Smoothing:')
            col1, col2, col3, col4 = st.columns(4)
            
            # Exibir métricas com delta
            col1.metric(label="MAE", value=f"{mae:.5f}", delta=f"{mae:.5f}", delta_color="off")  # Erro médio absoluto
            col2.metric(label="MSE", value=f"{mse:.5f}", delta=f"{mse:.5f}", delta_color="off")  # Erro médio quadrático
            col3.metric(label="RMSE", value=f"{rmse:.5f}", delta=f"{rmse:.5f}", delta_color="off")  # Raiz do erro médio quadrático
            col4.metric(label="MAPE", value=f"{mape:.5f}%", delta=f"{mape:.5f}%", delta_color="off")  # Erro médio percentual absoluto

# Modelo Simple Exponential Smoothing
class ModeloSES(ModeloPrevisao):
    def treinar(self):
        array_VALOR_EMPENHADO_log_treino = np.asarray(self.df_treino['VALOR_EMPENHADO_log'])
        self.modelo = SimpleExpSmoothing(array_VALOR_EMPENHADO_log_treino).fit(smoothing_level=0.9, optimized=True)

    def prever(self, mostrar_previsao_futura=False, escolha_periodo=0):
        self.df_valid['previsoes_v1'] = self.modelo.forecast(len(self.df_valid))
        previsoes_futuras = None
        if mostrar_previsao_futura:
            previsoes_futuras = self.modelo.forecast(escolha_periodo)
            previsoes_futuras = pd.Series(previsoes_futuras, index=pd.date_range(start=self.df_valid.index[-1], periods=escolha_periodo, freq='M'))
        previsoes_df, serie_historica_completa_df = plotar_grafico(self.df_treino, self.df_valid, self.df_valid['previsoes_v1'], 'Simple Exponential Smoothing Forecasting', previsoes_futuras)
        self.exibir_metricas(self.df_valid['VALOR_EMPENHADO_log'], self.df_valid['previsoes_v1'])
        if mostrar_previsao_futura:
            mostrar_resultados_previsao(previsoes_futuras)
        return previsoes_df, serie_historica_completa_df

# Modelo Double Exponential Smoothing
class ModeloDES(ModeloPrevisao):
    def treinar(self):
        self.modelo = ExponentialSmoothing(np.asarray(self.df_treino['VALOR_EMPENHADO_log']), trend='additive').fit(smoothing_level=0.2, optimized=True)

    def prever(self, mostrar_previsao_futura=False, escolha_periodo=0):
        self.df_valid['previsoes_v2'] = self.modelo.forecast(len(self.df_valid))
        previsoes_futuras = None
        if mostrar_previsao_futura:
            previsoes_futuras = self.modelo.forecast(escolha_periodo)
            previsoes_futuras = pd.Series(previsoes_futuras, index=pd.date_range(start=self.df_valid.index[-1], periods=escolha_periodo, freq='M'))
        previsoes_df, serie_historica_completa_df = plotar_grafico(self.df_treino, self.df_valid, self.df_valid['previsoes_v2'], 'Double Exponential Smoothing Forecasting', previsoes_futuras)
        self.exibir_metricas(self.df_valid['VALOR_EMPENHADO_log'], self.df_valid['previsoes_v2'])
        if mostrar_previsao_futura:
            mostrar_resultados_previsao(previsoes_futuras)
        return previsoes_df, serie_historica_completa_df

# Modelo Triple Exponential Smoothing
class ModeloTES(ModeloPrevisao):
    def treinar(self):
        self.modelo = ExponentialSmoothing(np.asarray(self.df_treino['VALOR_EMPENHADO_log']), trend='multiplicative', seasonal='additive', seasonal_periods=12).fit()

    def prever(self, mostrar_previsao_futura=False, escolha_periodo=0):
        self.df_valid['previsoes_v3'] = self.modelo.forecast(len(self.df_valid))
        previsoes_futuras = None
        if mostrar_previsao_futura:
            previsoes_futuras = self.modelo.forecast(escolha_periodo)
            previsoes_futuras = pd.Series(previsoes_futuras, index=pd.date_range(start=self.df_valid.index[-1], periods=escolha_periodo, freq='M'))
        previsoes_df, serie_historica_completa_df = plotar_grafico(self.df_treino, self.df_valid, self.df_valid['previsoes_v3'], 'Triple Exponential Smoothing Forecasting', previsoes_futuras)
        self.exibir_metricas(self.df_valid['VALOR_EMPENHADO_log'], self.df_valid['previsoes_v3'])
        if mostrar_previsao_futura:
            mostrar_resultados_previsao(previsoes_futuras)
        return previsoes_df, serie_historica_completa_df

# Função auxiliar para formatar valores
def formatar_valor(valor):
    if valor >= 1e9:
        return f"R${valor / 1e9:.2f}B"
    elif valor >= 1e6:
        return f"R${valor / 1e6:.2f}M"
    else:
        return f"R${valor:.2f}"

# Função auxiliar para formatar o índice de data
def formatar_data(data):
    return data.strftime('%Y-%m')

# Mostrar os resultados da previsão em Métrica
def mostrar_resultados_previsao(previsoes_futuras):
    # exponenciando as previsões
    previsoes_futuras = np.exp(previsoes_futuras)
    somatorio_total = previsoes_futuras.sum()
    media_previsao = previsoes_futuras.mean()
    minimo_previsao = previsoes_futuras.min()
    maximo_previsao = previsoes_futuras.max()

    st.write('Resultados da Previsão Futura:')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Somatório Total", value=formatar_valor(somatorio_total))
    col2.metric(label="Média da Previsão", value=formatar_valor(media_previsao))
    col3.metric(label="Mínimo da Previsão", value=formatar_valor(minimo_previsao))
    col4.metric(label="Máximo da Previsão", value=formatar_valor(maximo_previsao))

    # Formatar previsões para exibição tabular
    previsoes_futuras_formatadas = previsoes_futuras.apply(formatar_valor)
    previsoes_futuras_formatadas.index = previsoes_futuras_formatadas.index.map(formatar_data)

    # Mostrar previsões em formato tabular
    st.write('Previsões Futuras (Formato Tabular):')
    st.dataframe(previsoes_futuras_formatadas)

# Aplicação Streamlit
st.write('Selecione a porcentagem de treino.')
escolha_porcentagem_treino = st.slider('Porcentagem de Treino', 0.1, 0.99, 0.8)
porcentagem_treino = escolha_porcentagem_treino
indice_corte = int(len(dados) * porcentagem_treino)
df_treino = dados.iloc[:indice_corte]
df_valid = dados.iloc[indice_corte:]

try:
    with st.expander("Modelos de Previsão", expanded=True):
        modelo_selecionado = st.radio(
            "Selecione o modelo de previsão",
            ('Simple Exponential Smoothing', 'Double Exponential Smoothing', 'Triple Exponential Smoothing'),
        )

        mostrar_previsao_futura = st.checkbox("Mostrar Previsão Futura")
        escolha_periodo = st.selectbox('Selecione o período futuro para previsão (meses)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36], index=2)

        if modelo_selecionado == 'Simple Exponential Smoothing':
            modelo = ModeloSES(df_treino, df_valid)
        elif modelo_selecionado == 'Double Exponential Smoothing':
            modelo = ModeloDES(df_treino, df_valid)
        elif modelo_selecionado == 'Triple Exponential Smoothing':
            modelo = ModeloTES(df_treino, df_valid)

        modelo.treinar()
        previsoes_df, serie_historica_completa_df = modelo.prever(mostrar_previsao_futura, escolha_periodo)
        
        # Botões para download dos dataframes
        # COolocar em coluna
        col1, col2 = st.columns(2)  
        with col1:
            st.download_button(
                label="Baixar Previsões 2024",
                data=baixar_excel(previsoes_df, 'previsoes.xlsx'),
                file_name='previsoes_2024.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        with col2:
            st.download_button(
                label="Baixar Série Histórica Completa + Previsões 2024",
                data=baixar_excel(serie_historica_completa_df, 'serie_historica_completa.xlsx'),
                file_name='serie_historica_completa.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

except Exception as e:
    st.error(f"Ocorreu um erro: {e}")

display_credits()