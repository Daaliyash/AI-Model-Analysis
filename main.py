import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from Pages import page_0, page_1

st.set_page_config(page_title="Regression Analysis", layout='wide')

# with open('style.css') as f:
#     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = 0

if 'file_path' not in st.session_state:
    st.session_state['file_path'] = ''

if 'clicked' not in st.session_state:
    st.session_state['clicked'] = None

if 'next_bt' not in st.session_state:
    st.session_state['next_bt'] = False

if 'model' not in st.session_state:
    st.session_state['model'] = 0

if 'file_data' not in st.session_state:
    st.session_state['file_data'] = 0

if st.session_state.page == 0:
    page_0()
elif st.session_state.page == 1:
    page_1()
