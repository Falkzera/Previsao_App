import pandas as pd
import streamlit as st

def preprocess_data(dados):
    sigla = pd.read_csv('data/sigla.csv')
    sigla['UO'] = sigla['UO'].astype('object')
    sigla_nat3 = pd.read_csv('data/sigla_nat3.csv')

    # Merge sigla and sigla_nat3 data
    dados = dados.merge(sigla[['UO', 'SIGLA']], on='UO', how='left')
    dados = dados.merge(sigla_nat3[['NATUREZA3', 'NATUREZA3_DESC']], on='NATUREZA3', how='left')

    # Rename columns
    dados.rename(columns={'SIGLA': 'UO_sigla'}, inplace=True)

    # Create ANO_MES column
    dados['ANO_MES'] = dados['ANO'].astype(str) + '-' + dados['MES'].astype(str)
    dados['ANO_MES'] = pd.to_datetime(dados['ANO_MES'], format='%Y-%m')
    dados = dados.sort_values(by=['ANO', 'MES']).reset_index(drop=True)

    # Convert columns to object type
    convertendo_obj = ['ANO', 'MES', 'PODER', 'UO', 'UG', 'FONTE_MAE', 'NATUREZA3']
    for column in convertendo_obj:
        dados[column] = dados[column].astype('object')

    # Filter data
    dados = dados[dados['VALOR_EMPENHADO'] > 1]
    dados.set_index('ANO_MES', inplace=True)  # Set index and remove the column

    return dados

def handle_sidebar(dados):
    if st.sidebar.checkbox('Ativar Siglas', value=True):
        dados['UO'] = dados['UO_sigla']
        dados['NATUREZA3'] = dados['NATUREZA3_DESC']
    else:
        dados['UO'] = dados['UO']
        dados['NATUREZA3'] = dados['NATUREZA3']
    return dados

# Main script
def carregamento(dados):

    # Preprocess data
    dados = preprocess_data(dados)

    # Handle sidebar options
    dados = handle_sidebar(dados)

    # Further processing or visualization can be added here

if __name__ == "__main__":
    carregamento()

# Exemplo de uso

# from modulos.carregamento import carregamento
# carregamento()

