# estacionaridade.py

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import streamlit as st

# Função para testar a estacionaridade
def dsa_testa_estacionaridade(serie, window=12):
    """
    Função para testar a estacionaridade de uma série temporal.
    
    Parâmetros:
    - serie: pandas.Series. Série temporal a ser testada.
    - window: int. Janela para cálculo das estatísticas móveis.
    
    Retorna:
    - bool. True se a série for estacionária, False caso contrário.
    """
    # Teste Dickey-Fuller
    print('\nResultado do Teste Dickey-Fuller:')
    dfteste = adfuller(serie, autolag='AIC')
    dfsaida = pd.Series(dfteste[0:4], index=['Estatística do Teste', 
                                             'Valor-p', 
                                             'Número de Lags Consideradas', 
                                             'Número de Observações Usadas'])
    for key, value in dfteste[4].items():
        dfsaida['Valor Crítico (%s)' % key] = value
        
    print(dfsaida)
    # st.write(dfsaida)
    
    # Conclusão baseada no valor-p
    if dfsaida['Valor-p'] > 0.05:
        print('\nConclusão:\nO valor-p é maior que 0.05 e, portanto, não temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente não é estacionária.')
        # st.write('\nConclusão:\nO valor-p é maior que 0.05 e, portanto, não temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente não é estacionária.')
        return False
    else:
        print('\nConclusão:\nO valor-p é menor que 0.05 e, portanto, temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente é estacionária.')
        # st.write("\nConclusão:\nO valor-p é menor que 0.05 e, portanto, temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente é estacionária.")
        return True