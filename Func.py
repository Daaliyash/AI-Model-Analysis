import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# import time

def data_cleaning(data1):
    #Dropping columns with only 1 unique value, having more than 20% null data and containing data other than int or float datatypes
    for i in data1.columns:
        if (len(data1[i].unique()) <= 1) or (len(data1[i].unique()) == 2 and True in np.isnan(data1[i].unique())) or (data1[i].isna().sum() >= 0.2*(len(data1[i]))) or (data1[i].dtype != "int64" and data1[i].dtype != "float64"):
            data1.drop(i,axis=1,inplace=True)

    #Treating Nan values
    for i in data1.columns:
        data1[i].fillna(data1[i].median(), inplace=True)
    return data1

def acc_score(test, pred):
    mse = mean_squared_error(test, pred)
    return mse**0.5
    
def tt_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    return X_train, X_test, y_train, y_test

def plot_chart(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(25)], y=y_test[-25:], mode='markers', name='Testing points', line=dict(color="red")))
    fig.add_trace(go.Scatter(x=[i for i in range(25)], y=y_pred[-25:], mode='lines+markers', name='Regression Line',opacity=0.5, line=dict(color="light blue")))
    fig.update_layout(
    title="Regression Analysis (Actual vs Predicted)",
    xaxis_title="Actual Test Values",
    yaxis_title="Model Predictions",
    legend_title="Legend",
    )
    st.plotly_chart(fig, use_container_width=True)

def model_split_build(data1):
    X = data1[data1.columns[:-1]]
    y = data1[data1.columns[-1]]
    X_train, X_test, y_train, y_test = tt_split(X, y)
    if len(y.unique()) >= 10:
        st.title('Regression Analysis')
        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)
        importances = rfr.feature_importances_
        forest_importances = pd.Series(importances, index=X.columns)
        forest_importances = forest_importances.sort_values(ascending=False)
        with st.container(border=True):
            st.info("**Visualization of how each Independent variable impacts the Dependent variable**")
            st.bar_chart(forest_importances)
        progress_bar = st.progress(0)
        status_text = st.empty()
        # for i in range(1, 101):
        #     status_text.text("%i%% Complete" % i)
        #     progress_bar.progress(i)
        #     time.sleep(0.05)
        st.session_state['model'] = rfr
        # with open('model.pkl','wb') as f:
        #     pickle.dump(rfr, f)
        y_pred = rfr.predict(X_test)
        col1,col2 = st.columns([2,1])
        with col1.container(border=True):
            plot_chart(y_test[:30],y_pred[:30])
        with col2.container(border=True):
            num_steps = 100  
            colors = []
            for i in range(num_steps):
                red = 1.0 if i < num_steps / 2 else 1.0 - (2.0 * (i - num_steps / 2) / num_steps)
                green = 1.0 - abs(i - num_steps / 2) / num_steps
                blue = 1.0 if i >= num_steps / 2 else 1.0 - (2.0 * (num_steps / 2 - i) / num_steps)
                colors.append((red, green, blue))
            # colors[round(accuracy_score(y_test,y_pred)*100)] = (0,0,0)
            plt.figure(figsize=(10, 2))
            plt.imshow([colors], extent=[0, num_steps, 0, 1])
            plt.axis('off')
            st.pyplot(plt, use_container_width=True)
            # st.slider('', 0, 100, round(accuracy_score(y_test,y_pred)*100), disabled=True)
            st.info(f'The Error rate is {acc_score(y_test, y_pred)} for the range of {y.max() - y.min()}')
            st.info(f'Accuracy: {(rfr.score(X_test, y_test)*100).round(2)}%')

    elif len(y.unique()) <= 10:
        st.title('Classification Analysis')
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        importances = rfc.feature_importances_
        forest_importances = pd.Series(importances, index=X.columns)
        forest_importances = forest_importances.sort_values(ascending=False)
        with st.container(border=True):
            st.info("**Visualization of how each Independent variable impacts the Dependent variable**")
            st.bar_chart(forest_importances)
        progress_bar = st.progress(0)
        status_text = st.empty()
        # for i in range(1, 101):
        #     status_text.text("%i%% Complete" % i)
        #     progress_bar.progress(i)
        #     time.sleep(0.05)
        st.session_state['model'] = rfc
        # with open('model.pkl','wb') as f:
        #     pickle.dump(rfc, f)
        y_pred = rfc.predict(X_test)
        col1, col2 = st.columns([1.5,1])
        with col2.container(border=True):
            st.info('**Accuracy of Model**')
            num_steps = 100  
            colors = []
            for i in range(num_steps):
                red = 1.0 if i < num_steps / 2 else 1.0 - (2.0 * (i - num_steps / 2) / num_steps)
                green = 1.0 - abs(i - num_steps / 2) / num_steps
                blue = 1.0 if i >= num_steps / 2 else 1.0 - (2.0 * (num_steps / 2 - i) / num_steps)
                colors.append((red, green, blue))
            colors[round(accuracy_score(y_test,y_pred)*100)] = (0,0,0)
            plt.figure(figsize=(10, 2))
            plt.imshow([colors], extent=[0, num_steps, 0, 1])
            plt.axis('off')
            st.pyplot(plt, use_container_width=True)
            st.slider('', 0, 100, round(accuracy_score(y_test,y_pred)*100), disabled=True)
            st.text(f"Accuracy: {round(accuracy_score(y_test,y_pred)*100, 2)}")
        with col1.container(border=True):
            st.info("**Confusion Matrix**")
            cm = confusion_matrix(y_test,y_pred)
            fig = go.Figure(data=go.Heatmap(
                   z=cm,
                   x=['True', 'False'],
                   y=['True', 'False'],
                   text=[[cm[0][0], cm[0][1]],
                         [cm[1][0], cm[1][1]]],
                   texttemplate='%{text}',
                   hoverongaps = True))
            st.plotly_chart(fig)
            st.text(classification_report(y_test,y_pred))
    # rmse = acc_score(y_test, y_pred)
    # plot_chart(y_test, y_pred)
    # st.write('Error Margin: ', rmse)
    # acc = rfr.score(X_test, y_test)*100
    # st.write('Accuracy: ',acc.round(2) ,'%')

