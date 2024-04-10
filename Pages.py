import streamlit as st
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import warnings
warnings.filterwarnings('ignore')
from Func import data_cleaning, model_split_build

def nextpage(): st.session_state.page += 1
def back(): 
    st.session_state.page -= 1
    st.session_state.clicked = None

def ind_dep(df):
    col1, col2 = st.columns([1,1])
    with col1.container(border=True):
        ind_var = st.multiselect("Select your independent columns:",df.columns)
    with col2.container(border=True):
        dep_var = st.multiselect("Select your dependent columns:",[i for i in df.columns.values if i not in ind_var],max_selections=1)
    next_bt = st.button("Next")
    return ind_var, dep_var, next_bt

def page_0():
    st.image(
            "https://datalyzer.b-cdn.net/wp-content/uploads/2022/01/logo-3.png.webp",
            width=200, # Manually Adjust the width of the image as per requirement
        )
    st.title('List of Datasets')
    with st.container(border=True):
        cwd = os.getcwd()
        files = os.listdir(cwd)
        documents = [f for f in files if os.path.isfile(os.path.join(cwd, f)) and (f[-3:] == 'csv' or f[-4:] == 'xlsx')]
        data = pd.DataFrame({'Select': [False for i in range(len(documents))],
                'Index': [i+1 for i in range(len(documents))],
                'File Name': documents,
                'Timestamp': [datetime.datetime.fromtimestamp(os.path.getatime(os.path.join(cwd, f))) for f in documents]
        })
        res = st.data_editor(data,
                            column_config={"Select": st.column_config.CheckboxColumn(default=False), 
                                            "File Name": st.column_config.Column(width="large"), 
                                            "Timestamp": st.column_config.Column(width="large")},
                            hide_index=True, 
                            use_container_width=True)
        st.session_state['file_path'] = res.loc[res.Select.idxmax()]['File Name']
        
        uploaded_file = st.file_uploader('Upload a new file:')
        if uploaded_file is not None:
            # d = pd.read_csv(uploaded_file)
            if uploaded_file.name[-3:] == 'csv':
                try:
                    pd.read_csv(uploaded_file).to_csv(uploaded_file.name)
                except:
                    pd.read_csv(uploaded_file, delimiter=';').to_csv(uploaded_file.name)
                #st.session_state['file_data'] = pd.read_csv(uploaded_file)
            else:
                pd.read_excel(uploaded_file).to_excel(uploaded_file.name)
            st.session_state['file_path'] = uploaded_file.name
            # for i in range(1):
            #     st.rerun()
        # except:
        #     st.write()
        col1, col2 = st.columns([3,1])
        if len(res[res.Select == True])==1 or uploaded_file is not None:
            with col1:
                st.button("Proceed to Regression", on_click=nextpage, disabled=(st.session_state.page > 1))
            with col2:
                if st.button('Delete'):
                    if os.path.isfile(res.loc[res.Select.idxmax()]['File Name']):
                        os.remove(res.loc[res.Select.idxmax()]['File Name'])
                        st.rerun()
        else:
            st.write('Please select only 1 option / database')
        # st.session_state['file_data'] = pd.read_csv(res.loc[res.Select.idxmax()]['File Name'])

def page_1():
    st.title('Regression/Classification Analysis')
    # st.dataframe(st.session_state['file_data'])
    try:
        try:
            df = pd.read_csv(st.session_state['file_path'])
        except:
            df = pd.read_csv(st.session_state['file_path'], delimiter = ';')
    except:
        df = pd.read_excel(st.session_state['file_path'])
    df = data_cleaning(df)
    with st.container(border=True):
        col1, col2, col3 = st.columns([2,2,2])
        with col1:
            if st.button('DataInfo'):
                st.session_state['clicked'] = 1
        with col2:
            if st.button('AI Model Analysis'):
                st.session_state['clicked'] = 2
        with col3:
            if st.button('Predictor'):
                st.session_state['clicked'] = 3
    
        if st.session_state.clicked == 1:
            ind_var, dep_var, next_bt = ind_dep(df)
            if next_bt:
                st.write('----')
                pr = ProfileReport(df[ind_var+dep_var], title="Data Report")
                st_profile_report(pr)
            st.info('Dataset info')
            # st.write('----')
        elif st.session_state.clicked == 2:
            ind_var, dep_var, next_bt = ind_dep(df)
            st.session_state.ind_var, st.session_state.dep_var, st.session_state.next_bt = ind_var, dep_var, next_bt
            if next_bt:
                st.write('----')
                model_split_build(df[ind_var+dep_var])
                # st.write('----')
        elif st.session_state.clicked == 3:
            # st.write('----')
            try:
                if st.session_state.next_bt:
                    st.write('----')
                    st.dataframe(df[st.session_state.ind_var+st.session_state.dep_var])
                    col1, col2 = st.columns([1,1])
                    with col1.container(border=True):
                        slider_val = []
                        for i in st.session_state.ind_var:
                            slider_val.append(st.slider(i, float(df[i].min()),float(df[i].max()),float(df[i].mean())))
                        model = st.session_state['model']
                    with col2.container(border=True):
                        st.write(f'Predicted value of {st.session_state.dep_var[0]}')
                        num_steps = 100
                        steps = (df[st.session_state.dep_var[0]].max() - df[st.session_state.dep_var[0]].min()) / 99
                        colors,arr = [],[]
                        for i in range(num_steps):
                            red = 1.0 if i < num_steps / 2 else 1.0 - (2.0 * (i - num_steps / 2) / num_steps)
                            green = 1.0 - abs(i - num_steps / 2) / num_steps
                            blue = 1.0 if i >= num_steps / 2 else 1.0 - (2.0 * (num_steps / 2 - i) / num_steps)
                            colors.append((red, green, blue))
                            if i == 0:
                                arr.append(df[st.session_state.dep_var[0]].min())
                            else:
                                arr.append(arr[i-1]+steps)
                        for i in range(1, len(arr)):
                            if model.predict([slider_val])[0] <= arr[i]:
                                colors[i] = (0,0,0)
                                break
                        plt.figure(figsize=(10, 2))
                        plt.imshow([colors], extent=[0, num_steps, 0, 1])
                        plt.axis('off')
                        st.pyplot(plt, use_container_width=True)
                        st.slider('', float(df[st.session_state.dep_var[0]].min()), float(df[st.session_state.dep_var[0]].max()), float(model.predict([slider_val])[0]), disabled=True)
                else:
                    st.info("Please perform **AI Model Analysis** and then press the Predictor button")
            except:
                st.info("Please perform **AI Model Analysis** and then press the Predictor button")
    st.button('Back to Datasets', on_click=back, disabled=(st.session_state.page > 1))
