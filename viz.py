import plotly.express as plt
import streamlit as st
class viz_class:
    def visualization(data):
        col1,col2=st.columns(2)
        chart=col1.selectbox('Choose chart', options=['','Line','Bar','Histogram','Scatter'])
        columns=col2.multiselect('Choose column',options=data.columns)
        try:
            if len(columns)==2:
                if chart=='Line':
                    st.write(plt.line(x=data[columns[0]],y=data[columns[1]]))
                if chart=='Bar':
                    st.write(plt.bar(x=data[columns[0]],y=data[columns[1]]))
                if chart=='Histogram':
                    st.write(plt.histogram(x=data[columns[0]],y=data[columns[1]]))
                if chart=='Scatter':
                    st.write(plt.scatter(x=data[columns[0]],y=data[columns[1]]))
                    
            else:
                if chart=='Line':
                    st.write(plt.line(data[columns]))
                if chart=='Bar':
                    st.write(plt.bar(data[columns]))
                if chart=='Histogram':
                    st.write(plt.histogram(data[columns]))
                if chart=='Scatter':
                    st.write(plt.scatter(data[columns]))
                
        except:
            st.write('Choose correct columns')
            