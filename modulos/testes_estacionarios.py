import numpy as np
from scipy.stats import boxcox
import streamlit as st
from modulos.estacionaridade import dsa_testa_estacionaridade

def transformar_log(dados):
    return np.log(dados)

def transformar_sqrt(dados):
    return np.sqrt(dados)

def transformar_boxcox(dados):
    return boxcox(dados)[0]

def transformar_media_movel_simples(dados):
    dados_transformados = dados.rolling(window=12).mean()
    dados_transformados = dados - dados_transformados
    dados_transformados.dropna(inplace=True)
    return dados_transformados

def transformar_media_movel_exponencial(dados):
    dados_transformados = dados.ewm(alpha=0.2, adjust=True).mean()
    return dados - dados_transformados

def aplicar_transformacao(dados, transformacao):
    if transformacao == 'log':
        return transformar_log(dados)
    elif transformacao == 'sqrt':
        return transformar_sqrt(dados)
    elif transformacao == 'boxcox':
        return transformar_boxcox(dados)
    elif transformacao == 'media_movel_simples':
        return transformar_media_movel_simples(dados)
    elif transformacao == 'media_movel_exponencial':
        return transformar_media_movel_exponencial(dados)
    else:
        raise ValueError("Transformação desconhecida")

def aplicar_transformacao_e_testar(dados, transformacao, descricao):
    dados_transformados = aplicar_transformacao(dados, transformacao)
    teste_estacionaridade = dsa_testa_estacionaridade(dados_transformados)
    if teste_estacionaridade:
        st.success(f'A série é estacionária após a transformação: {descricao}.')
    return teste_estacionaridade, dados_transformados
