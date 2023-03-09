import streamlit as st
import pandas as pd
import plotly.express as plt
st.set_page_config(layout="wide")
data=st.file_uploader("Upload")
if data is not None:
    file=pd.read_csv(data)
    st.write(file.head())
    st.write(plt.box(file['Age']))
    if st.button("press"):
        c1,c2,c3=st.columns(3)
        with c3:
            st.markdown("""<h1>Hallo</h1>""",True)
    col1,col2=st.columns(2)
    with col1:
        st.write(file.head())
    with col2:
        st.write(file.tail())