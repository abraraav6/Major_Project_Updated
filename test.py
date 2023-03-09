import streamlit as st
import pandas as pd
import numpy as np
file=st.file_uploader('Upload file',type=['CSV','XLSX'])
if file is not None:
    data=pd.read_csv(file)
if st.button("Group By"):
    col=['']
    col=col.extend(st.selectbox("Choose Column",options=data.columns))
    num_col=data.select_dtypes([np.number]).columns
    cat_col=data.select_dtypes(['object']).columns
    if col in num_col:
        st.write(data.groupby(col).describe())
    else:
        st.write(data.groupby(col).describe())