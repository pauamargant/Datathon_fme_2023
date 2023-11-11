import streamlit as st


# Interfaccia utente Streamlit
st.title("Generatore di Outfit")

# Variabili modificabili dall'utente
cod_color_code = st.text_input("Inserisci il codice colore:")
des_color_specification_esp = st.text_input("Inserisci la specifica del colore:")
des_agrup_color_eng = st.text_input("Inserisci la descrizione del colore:")
des_sex = st.selectbox("Sesso:", ["Male", "Female"])
des_age = st.selectbox("Età:", ["Adult", "Teenager", "Child"])
des_line = st.selectbox("Linea:", ["Casual", "Formal", "Sportswear"])

# Pulsante per generare l'outfit
if st.button("Genera Outfit"):
    st.success('OK')
