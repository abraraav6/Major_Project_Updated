import pandas as pd
import streamlit as st
import viz
import analysis
import explore
import agri
import stock_market
import ml_algos 
import prepare_data
st.markdown("""<h2 style="font-family: cursive;color: green;"><center><u>Upload your File and start Questioning</u></center></h2>""",unsafe_allow_html=True)
class start_class:
    def fileupload(self):
        self.file=st.file_uploader('Upload file',type=['CSV','XLSX'])
        if self.file is not None:
            try:
                self.data=pd.read_csv(self.file)
                return self.data
            except:
                try:
                    self.data=pd.read_excel(self.file)
                    return self.data
                except:
                    st.warning('Supported CSV, EXCEL files only')
    def opt(self):
        self.option=st.sidebar.selectbox('Choose operation to perform',['','Visualize','Analyze','Explore','Agriculture','Stock Market','Algorithms Comparission','Prepare Data'])
        return self.option
if __name__=='__main__':
    obj=start_class()
    data=obj.fileupload()
    to_do=obj.opt()
    if data is not None:
        if to_do=='Analyze':
            analysis.analyse_class.analyse_fun(data)
        if to_do=='Visualize':
            viz.viz_class.visualization(data)
        if to_do=='Explore':
            explore.understand.exp(data)
        if to_do=='Algorithms Comparission':
            ml_algos.algos.app(data)
        if to_do=='Prepare Data':
            prepare_data.clean_prepare_data.clean_prepare(data)
    if to_do=='Agriculture':
        agri.agri_rec.input_fun()
    if to_do=='Stock Market':
        stock_market.stock_prediction.start()
            