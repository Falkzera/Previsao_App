import streamlit as st


def display_credits():
    with st.sidebar:
    
        st.image('image/SOP.png', width=300, use_column_width=False)
        


if __name__ == "__main__":
    display_credits()