import streamlit as st


def display_credits():
    with st.sidebar:
        st.write('---')
        st.caption('Develper by:')
        st.image('image/SOP.png', width=300, use_column_width=True)
        


if __name__ == "__main__":
    display_credits()