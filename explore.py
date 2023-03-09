import pandas as pd
import pandas_profiling as pp
import streamlit as st
import streamlit_pandas_profiling as spp
data=pd.DataFrame()
class understand:
    def exp(data):
        profile=pp.ProfileReport(data)
        st.write(data)
        spp.st_profile_report(profile)