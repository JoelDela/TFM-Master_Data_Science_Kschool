
import pandas as pd
import altair as alt
import streamlit as st
from pathlib import Path
import numpy as np
import os 
alt.data_transformers.disable_max_rows()

# After importing the modules, we are defining our functions:
def read_markdown_file(markdown_file):
    """
    With this function we will be able to read the README.md
    """
    return Path(markdown_file).read_text()
    
def mean_absolute_percentage_error(y_true, y_pred): 
    """
    This is the MAPE metric
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSE_metric(y_test,y_pred):
    """
    This is the RMSE metric
    """
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_test, y_pred))

def R2_metric(y_test,y_pred):
    """
    This is the R2 metric
    """
    from sklearn.metrics import r2_score
    return r2_score(y_test, y_pred)    
 
def return_metrics(df):
    """
    With this function we will return the metrics for each model
    """
    return f"""
    For the Facebook Prophet model we have achieved a total RMSE of: 
    {round(RMSE_metric(df[df['type']=='test']['value'],
    df[df['type']=='prediccion']['value']),2)}, a R2 of:
    {round(R2_metric(df[df['type']=='test']['value'],
    df[df['type']=='prediccion']['value']),2)} and a MAPE of:
    {round(mean_absolute_percentage_error(df[df['type']=='test']['value'],
    df[df['type']=='prediccion']['value']),2)}
    """

def plot_model(df, plot_title):
    """
    With this function we will plot each model
    """
    # First we create the selector
    sel = alt.selection(type='interval',encodings = ['x'], name='sel')

    # Then we plot our first chart where we'll be able to select the range:'
    df_predictions = alt.Chart(df).mark_line().encode(
        x=alt.X('date:T',
           sort=alt.Sort(field="date",
                              order="descending")),
        y=alt.Y('value', title = 'price €/MWh'),
        color='type',
        tooltip = [alt.Tooltip('date:T'),
                   alt.Tooltip('value:Q')],
    ).properties(
        selection = sel,
        width = 1000,
        height = 500
    )
    # Then we plot the chart where we will see the range's RMSE and we will be abe to interact
    zoom_df = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('date:T',
           sort=alt.Sort(field="date",
                              order="descending")),
        y=alt.Y('value', title = 'price €/MWh'),
        color='type',
        tooltip = [alt.Tooltip('date:T'),
                   alt.Tooltip('value:Q')]
    ).transform_filter(
        sel.ref()
    ).properties(
        width = 1000,
        height = 500
    ).interactive()
    # Finally we create the dynamic RMSE:
    metric = alt.Chart(df).transform_pivot(
        pivot='type',groupby=['date'], value='value',
    ).transform_filter(
        sel.ref()
    ).transform_calculate(
        diff='(datum.test - datum.prediccion)*(datum.test - datum.prediccion)'  
    ).transform_aggregate(
        total2 = 'mean(diff)'
    ).transform_calculate(
        total = "sqrt(datum.total2)",
        date_range="sel.date ? datetime(sel.date[0]) + ' to ' + datetime(sel.date[1]) : 'all'",
        text="""'Total RMSE for ' + datum.date_range + ': ' + format(datum.total, '.2f')""" 
    ).mark_text(
        align='left',
        baseline='top',
        size = 11
    ).encode(
        x=alt.value(5),
        y=alt.value(3),
        text=alt.Text('text:N')
    )
    # We plot one chart on top of the other with the metric:
    st.write((zoom_df+metric & df_predictions).properties(
        title=plot_title
    ))
    #We return the main metrics for the project:
    st.markdown(return_metrics(df))
    
    
# Then we read the dataframes:
dataframes = [] 
for i in os.listdir('./'): 
    if i.endswith('.csv'): 
        info = pd.DataFrame(
            pd.read_csv(i,index_col=0).stack().reset_index()
        )
        info.columns = ['date','type','value']
        dataframes.append(info)
    else:
        continue


# Give a name for each dataframe:

xgb_shuff = dataframes[5]
xgb_sort = dataframes[6]
sarimax = dataframes[4]
fp_mod = dataframes[0]
lstm_multilayer = dataframes[3]
dnn_shuff = dataframes[1]
dnn_sort = dataframes[2]



# _________________________
# Creating a side bar where you can choose the kind of model:
st.sidebar.title('Content')
choice = st.sidebar.radio("Choice your model: ",("Introduction","Time Series","Neural Network","Gradient Boosting"))

# if you check introduction you will read the frontpage from github:
if choice == "Introduction":

    intro_markdown = read_markdown_file("./../README.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

# When you choose the tab for Time Series:
elif choice == "Time Series": 
    # Title
    st.title("Comparison between different regression models for predicting the electricity price")
    # Subtitle
    st.subheader("""
    With this project we want to compare the behaviour of some different models when predicting the electricity price, checking if a time series model could fit for a short period time:
    """)
    
     # If you check this one the fb prophet shows:
    check1 = st.checkbox("FB Prophet")
    
    if check1:
        st.markdown("""The following chart is showing us the results for the Facebook Prophet model. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the 
        bottom chart""")
        
        plot_model(fp_mod, 'Facebook Prophet model predictions')
    
    check2 = st.checkbox("SARIMAX")
    
    if check2:
    
        st.markdown("""The following chart is showing us the results for the SARIMAX model. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the 
        bottom chart""")

        plot_model(sarimax, 'SARIMAX model predictions')

# When you choose the tab for Neural Networks:    
elif choice == "Neural Network":    
    # Title
    st.title("Comparison between different regression models for predicting the electricity price")
    # Subtitle
    st.subheader("""
    With this project we want to compare the behaviour of some different models when predicting the electricity price, checking if a time series model could fit for a short period time:
    """)
    
    # if you check the DNN with shuffled data:
    check3 = st.checkbox("DNN with shuffled data")
    
    if check3:
        # Header for the model
        st.markdown("""The following chart is showing us the results for the DNN model with shuffled data. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging 
        the mouse on the bottom chart""")
        # Plotting the model
        plot_model(dnn_shuff, 'DNN with shuffled data')
        
    #If you check the DNN with sorted data:   
    check5 = st.checkbox("DNN with sorted data")
    
    if check5:
        # Header
        st.markdown("""The following chart is showing us the results for the DNN model with sorted data. This means
        that for the training and test model we divided by date the data, so it hasn't been trained with 2020 data.
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")
        # Plotting the model
        plot_model(dnn_sort, 'DNN with sorted data')
               
    # If you check the LSTM multilayer:
    check6 = st.checkbox("Neural Network LSTM multilayer")
    
    if check6:
        # Header
        st.markdown("""The following chart is showing us the results for the LSTM Neural Network with more than 1 layer. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the 
        bottom chart. Only 7 days are shown on this chart because of limitations.""")
        # Plotting the model
        plot_model(lstm_multilayer, 'LSTM multilayer')
        
# When you choose the tab for Gradient Boosting: 
elif choice == 'Gradient Boosting':
    # Title
    st.title("Comparison between different regression models for predicting the electricity price")
    # Subtitle
    st.subheader("""
    With this project we want to compare the behaviour of some different models when predicting the electricity price, checking if a time series model could fit for a short period time:
    """)
    
    # if you check the XGBoost with shuffled data;
    check7 = st.checkbox("XGBoost with shuffled data")
    
    if check7:
        # Header
        st.markdown("""The following chart is showing us the results for the XGBoost with shuffled data. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")
        # Plotting the model
        plot_model(xgb_shuff, "XGBoost with shuffled data")
        
    # if you check the XGBoost with sorted data
    check8 = st.checkbox("XGBoost with sorted data")
    
    if check8:
        # Header
        st.markdown("""The following chart is showing us the results for the XGBoost with sorted data. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")
        # Plotting the model
        plot_model(xgb_sort, "XGBoost with sorted data")
