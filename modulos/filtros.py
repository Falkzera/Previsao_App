import streamlit as st

def filtros_usuario(dados):
    
    # Opções
    
    natureza_options = ['TODAS'] + list(dados['NATUREZA3'].unique()) # Opções de seleção para Natureza
    poder_options = ['TODAS'] + list(dados['PODER'].unique()) # Opções de seleção para Poder
    fonte_options = ['TODAS'] + list(dados['FONTE_MAE'].unique()) # Opções de seleção para Unidade Orçamentária
    uo_options = ['TODAS'] + list(dados['UO_sigla'].unique()) # Opções de seleção para Unidade Orçamentária
    default_poder = ['EXE']
    default_fonte = ['TODAS']
    default_natureza = ['TODAS']
    default_uo = ['TODAS']
    escolha_poder = st.sidebar.multiselect('Escolha o Poder', poder_options, default_poder) # Seleção de Poder
    escolha_nat3 = st.sidebar.multiselect('Escolha a Natureza - 3', natureza_options, default_natureza) # Seleção de Natureza
    escolha_fonte = st.sidebar.multiselect('Escolha a Fonte',fonte_options, default_fonte) # Seleção de Fonte
    escolha_uo = st.sidebar.multiselect('Escolha a Unidade Orçamentária', uo_options, default_uo) # Seleção de Unidade Orçamentária

    if 'TODAS' not in escolha_nat3:
        dados = dados[dados['NATUREZA3'].isin(escolha_nat3)]
    if 'TODAS' not in escolha_poder:
        dados = dados[dados['PODER'].isin(escolha_poder)]
    if 'TODAS' not in escolha_fonte:
        dados = dados[dados['FONTE_MAE'].isin(escolha_fonte)]
    if 'TODAS' not in escolha_uo:
        dados = dados[dados['UO_sigla'].isin(escolha_uo)]
    return dados

if __name__ == "__main__":
    filtros_usuario()