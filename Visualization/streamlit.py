
import pandas as pd
import altair as alt
import streamlit as st
from pathlib import Path
import numpy as np
import os 
alt.data_transformers.disable_max_rows()

# After importing the modules, we are defining our functions:
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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

def RMSE_metric(y_test,y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_test, y_pred))

def R2_metric(y_test,y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_test, y_pred)    
    
# _________________________
st.sidebar.title('Content')
choice = st.sidebar.radio("Choice your model: ",("Introduction","Time Series","Neural Network","Gradient Boosting"))


if choice == "Introduction":
    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()

    intro_markdown = read_markdown_file("./../README.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

elif choice == "Time Series": 
    
    st.title("Comparison between different regression models for predicting the electricity price")

    st.subheader("""
    With this project we want to compare the behaviour of some different models when predicting the electricity price, checking if a time series model could fit for a short period time:
    """)
    
    sel = alt.selection(type='interval',encodings = ['x'], name='sel')

    
    check1 = st.checkbox("FB Prophet")
    
    if check1:
        st.markdown("""The following chart is showing us the results for the Facebook Prophet model. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse""")

        fp_predictions = alt.Chart(dataframes[0]).mark_line(point=True).encode(
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

        metric1 = alt.Chart(dataframes[0]).transform_pivot(
            pivot='type',groupby=['date'], value='value',
        ).transform_filter(
            sel.ref()
        ).transform_calculate(
            diff='(datum.y_test - datum.y_pred)*(datum.y_test - datum.y_pred)'  
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

        st.write((fp_predictions+metric1).properties(
            title='Facebook Prophet model predictions'
        ))

        st.markdown(f"""
        For the Facebook Prophet model we have achieved a total RMSE of: 
        {round(RMSE_metric(dataframes[0][dataframes[0]['type']=='y_test']['value'],
        dataframes[0][dataframes[0]['type']=='y_pred']['value']),2)}, a R2 of:
        {round(R2_metric(dataframes[0][dataframes[0]['type']=='y_test']['value'],
        dataframes[0][dataframes[0]['type']=='y_pred']['value']),2)} and a MAPE of:
        {round(mean_absolute_percentage_error(dataframes[0][dataframes[0]['type']=='y_test']['value'],
        dataframes[0][dataframes[0]['type']=='y_pred']['value']),2)}
        """)
    
    check2 = st.checkbox("SARIMAX")
    
    if check2:
    
        st.markdown("""The following chart is showing us the results for the SARIMAX model. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse""")

        sarimax_predictions = alt.Chart(dataframes[5]).mark_line(point=True).encode(
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

        metric6 = alt.Chart(dataframes[5]).transform_pivot(
            pivot='type',groupby=['date'], value='value',
        ).transform_filter(
            sel.ref()
        ).transform_calculate(
            diff='(datum.y_test - datum.y_pred)*(datum.y_test - datum.y_pred)'  
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

        st.write((sarimax_predictions+metric6).properties(
            title='SARIMAX model predictions'
        ))

        st.markdown(f"""
        For the SARIMAX model we have achieved a total RMSE of: 
        {round(RMSE_metric(dataframes[5][dataframes[5]['type']=='y_test']['value'],
        dataframes[5][dataframes[5]['type']=='y_pred']['value']),2)}, a R2 of:
        {round(R2_metric(dataframes[5][dataframes[5]['type']=='y_test']['value'],
        dataframes[5][dataframes[5]['type']=='y_pred']['value']),2)} and a MAPE of:
        {round(mean_absolute_percentage_error(dataframes[5][dataframes[5]['type']=='y_test']['value'],
        dataframes[5][dataframes[5]['type']=='y_pred']['value']),2)}
        """)
    
elif choice == "Neural Network":    

    
    st.title("Comparison between different regression models for predicting the electricity price")

    st.subheader("""
    With this project we want to compare the behaviour of some different models when predicting the electricity price, checking if a time series model could fit for a short period time:
    """)
    
    sel = alt.selection(type='interval',encodings = ['x'], name='sel')

    check3 = st.checkbox("Neural Network multilayer shuffled data")
    
    if check3:

        st.markdown("""The following chart is showing us the results for the Neural Network model with shuffled data. This means
        that for the training and test model we divided randomly the data, so there is no connection between one point and another.
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")

        nn_reg_predictions = alt.Chart(dataframes[1]).mark_line().encode(
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

        zoom_nn_reg_predictions = alt.Chart(dataframes[1]).mark_line().encode(
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

        metric2 = alt.Chart(dataframes[1]).transform_pivot(
            pivot='type',groupby=['date'], value='value',
        ).transform_filter(
            sel.ref()
        ).transform_calculate(
            diff='(datum.y_test - datum.y_pred)*(datum.y_test - datum.y_pred)'  
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

        st.write((zoom_nn_reg_predictions+metric2 & nn_reg_predictions).properties(
            title='Network multilayer model predictions'
        ))
        
        st.markdown(f"""
        For the Neural Network multilayer model with shuffled data we have achieved a total RMSE of: 
        {round(RMSE_metric(dataframes[1][dataframes[1]['type']=='y_test']['value'],
        dataframes[1][dataframes[1]['type']=='y_pred']['value']),2)}, a R2 of:
        {round(R2_metric(dataframes[1][dataframes[1]['type']=='y_test']['value'],
        dataframes[1][dataframes[1]['type']=='y_pred']['value']),2)} and a MAPE of:
        {round(mean_absolute_percentage_error(dataframes[1][dataframes[1]['type']=='y_test']['value'],
        dataframes[1][dataframes[1]['type']=='y_pred']['value']),2)}
        """)

    check4 = st.checkbox("Neural Network LSTM 1 layer")
    
    if check4:

        st.markdown("""The following chart is showing us the results for the LSTM Neural Network with only 1 layer. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")

        lstm_1_predictions = alt.Chart(dataframes[2]).mark_line().encode(
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

        zoom_lstm_1_predictions = alt.Chart(dataframes[2]).mark_line().encode(
            x=alt.X('date:T',
               sort=alt.Sort(field="date",
                                  order="descending")),
            y=alt.Y('value', title = 'price €/MWh'),
            color='type',
            tooltip = [alt.Tooltip('date:T'),
                       alt.Tooltip('value:Q')],
        ).transform_filter(
            sel.ref()
        ).properties(
            width = 1000,
            height = 500
        ).interactive()

        metric3 = alt.Chart(dataframes[2]).transform_pivot(
            pivot='type',groupby=['date'], value='value',
        ).transform_filter(
            sel.ref()
        ).transform_calculate(
            diff='(datum.y_test - datum.y_pred)*(datum.y_test - datum.y_pred)'  
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

        st.write((zoom_lstm_1_predictions+metric3 & lstm_1_predictions).properties(
            title='LSTM Neural Network model with 1 layer predictions'
        ))

        st.markdown(f"""
        For the LSTM Neural Network model with 1 layer we have achieved a total RMSE of: 
        {round(RMSE_metric(dataframes[2][dataframes[2]['type']=='y_test']['value'],
        dataframes[2][dataframes[2]['type']=='y_pred']['value']),2)}, a R2 of:
        {round(R2_metric(dataframes[2][dataframes[2]['type']=='y_test']['value'],
        dataframes[2][dataframes[2]['type']=='y_pred']['value']),2)} and a MAPE of:
        {round(mean_absolute_percentage_error(dataframes[2][dataframes[2]['type']=='y_test']['value'],
        dataframes[2][dataframes[2]['type']=='y_pred']['value']),2)}
        """)
        
    check5 = st.checkbox("Neural Network multilayer sorted data")
    
    if check5:
        
        st.markdown("""The following chart is showing us the results for the Neural Network model with sorted data. This means
        that for the training and test model we divided by date the data, so it hasn't been trained with 2020 data.
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")

        nn_reg_ts_predictions = alt.Chart(dataframes[3]).mark_line().encode(
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

        zoom_nn_reg_ts_predictions = alt.Chart(dataframes[3]).mark_line().encode(
            x=alt.X('date:T',
               sort=alt.Sort(field="date",
                                  order="descending")),
            y=alt.Y('value', title = 'price €/MWh'),
            color='type',
            tooltip = [alt.Tooltip('date:T'),
                       alt.Tooltip('value:Q')],
        ).transform_filter(
            sel.ref()
        ).properties(
            width = 1000,
            height = 500
        ).interactive()

        metric4 = alt.Chart(dataframes[3]).transform_pivot(
            pivot='type',groupby=['date'], value='value',
        ).transform_filter(
            sel.ref()
        ).transform_calculate(
            diff='(datum.y_test - datum.y_pred)*(datum.y_test - datum.y_pred)'  
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

        st.write((zoom_nn_reg_ts_predictions+metric4 & nn_reg_ts_predictions).properties(
            title='Neural Network Multilayer model with sorted data predictions'
        ))
        
        st.markdown(f"""
        For the Neural Network Multilayer model with sorted data we have achieved a total RMSE of: 
        {round(RMSE_metric(dataframes[3][dataframes[3]['type']=='y_test']['value'],
        dataframes[3][dataframes[3]['type']=='y_pred']['value']),2)}, a R2 of:
        {round(R2_metric(dataframes[3][dataframes[3]['type']=='y_test']['value'],
        dataframes[3][dataframes[3]['type']=='y_pred']['value']),2)} and a MAPE of:
        {round(mean_absolute_percentage_error(dataframes[3][dataframes[3]['type']=='y_test']['value'],
        dataframes[3][dataframes[3]['type']=='y_pred']['value']),2)}
        """)

    check6 = st.checkbox("Neural Network LSTM multilayer")
    
    if check6:

        st.markdown("""The following chart is showing us the results for the LSTM Neural Network with more than 1 layer. 
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")
        
        lstm_mlt_predictions = alt.Chart(dataframes[4]).mark_line().encode(
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

        zoom_lstm_mlt_predictions = alt.Chart(dataframes[4]).mark_line().encode(
            x=alt.X('date:T',
               sort=alt.Sort(field="date",
                                  order="descending")),
            y=alt.Y('value', title = 'price €/MWh'),
            color='type',
            tooltip = [alt.Tooltip('date:T'),
                       alt.Tooltip('value:Q')],
        ).transform_filter(
            sel.ref()
        ).properties(
            width = 1000,
            height = 500
        ).interactive()

        metric5 = alt.Chart(dataframes[4]).transform_pivot(
            pivot='type',groupby=['date'], value='value',
        ).transform_filter(
            sel.ref()
        ).transform_calculate(
            diff='(datum.y_test - datum.y_pred)*(datum.y_test - datum.y_pred)'  
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

        st.write((zoom_lstm_mlt_predictions+metric5 & lstm_mlt_predictions).properties(
            title='LSTM Neural Network model with more than 1 layer predictions'
        ))
        
        st.markdown(f"""
        For the LSTM Neural Network model with more than 1 layer we have achieved a total RMSE of: 
        {round(RMSE_metric(dataframes[4][dataframes[4]['type']=='y_test']['value'],
        dataframes[4][dataframes[4]['type']=='y_pred']['value']),2)}, a R2 of:
        {round(R2_metric(dataframes[4][dataframes[4]['type']=='y_test']['value'],
        dataframes[4][dataframes[4]['type']=='y_pred']['value']),2)} and a MAPE of:
        {round(mean_absolute_percentage_error(dataframes[4][dataframes[4]['type']=='y_test']['value'],
        dataframes[4][dataframes[4]['type']=='y_pred']['value']),2)}
        """)

elif choice == 'Gradient Boosting':
    
    st.title("Comparison between different regression models for predicting the electricity price")

    st.subheader("""
    With this project we want to compare the behaviour of some different models when predicting the electricity price, checking if a time series model could fit for a short period time:
    """)
    
    sel = alt.selection(type='interval',encodings = ['x'], name='sel')
    
    check7 = st.checkbox("XGBoost shuffled data")
    
    if check7:

        st.markdown("""The following chart is showing us the results for the XGBoost with shuffled data. This means
        that for the training and test model we divided randomly the data, so there is no connection between one point and another.
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")

        xgb_predictions = alt.Chart(dataframes[6]).mark_line().encode(
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

        zoom_xgb_predictions = alt.Chart(dataframes[6]).mark_line().encode(
            x=alt.X('date:T',
               sort=alt.Sort(field="date",
                                  order="descending")),
            y=alt.Y('value', title = 'price €/MWh'),
            color='type',
            tooltip = [alt.Tooltip('date:T'),
                       alt.Tooltip('value:Q')],
        ).transform_filter(
            sel.ref()
        ).properties(
            width = 1000,
            height = 500
        ).interactive()

        metric7 = alt.Chart(dataframes[6]).transform_pivot(
            pivot='type',groupby=['date'], value='value',
        ).transform_filter(
            sel.ref()
        ).transform_calculate(
            diff='(datum.y_test - datum.y_pred)*(datum.y_test - datum.y_pred)'  
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

        st.write((zoom_xgb_predictions+metric7 & xgb_predictions).properties(
            title='XGBoost model with shuffled data  predictions'
        ))
    
        st.markdown(f"""
        For the XGBoost model with shuffled data we have achieved a total RMSE of: 
        {round(RMSE_metric(dataframes[6][dataframes[6]['type']=='y_test']['value'],
        dataframes[6][dataframes[6]['type']=='y_pred']['value']),2)}, a R2 of:
        {round(R2_metric(dataframes[6][dataframes[6]['type']=='y_test']['value'],
        dataframes[6][dataframes[0]['type']=='y_pred']['value']),2)} and a MAPE of:
        {round(mean_absolute_percentage_error(dataframes[6][dataframes[6]['type']=='y_test']['value'],
        dataframes[6][dataframes[6]['type']=='y_pred']['value']),2)}
        """)

    check8 = st.checkbox("XGBoost sorted data")
    
    if check8:
        
        st.markdown("""The following chart is showing us the results for the XGBoost with sorted data. This means
        that for the training and test model we divided by date the data, so it hasn't been trained with 2020 data.
        In this chart you can select the dates you want to check the RMSE by clicking and dragging the mouse on the bottom chart""")
        
        xgb_ts_predictions = alt.Chart(dataframes[7]).mark_line().encode(
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

        zoom_xgb_ts_predictions = alt.Chart(dataframes[7]).mark_line().encode(
            x=alt.X('date:T',
               sort=alt.Sort(field="date",
                                  order="descending")),
            y=alt.Y('value', title = 'price €/MWh'),
            color='type',
            tooltip = [alt.Tooltip('date:T'),
                       alt.Tooltip('value:Q')],
        ).transform_filter(
            sel.ref()
        ).properties(
            width = 1000,
            height = 500
        ).interactive()

        metric8 = alt.Chart(dataframes[7]).transform_pivot(
            pivot='type',groupby=['date'], value='value',
        ).transform_filter(
            sel.ref()
        ).transform_calculate(
            diff='(datum.y_test - datum.y_pred)*(datum.y_test - datum.y_pred)'  
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

        st.write((zoom_xgb_ts_predictions+metric8 & xgb_ts_predictions).properties(
            title='XGBoost model with sorted data  predictions'
        ))
        
        st.markdown(f"""
        For the XGBoost model with sorted data we have achieved a total RMSE of: 
        {round(RMSE_metric(dataframes[7][dataframes[7]['type']=='y_test']['value'],
        dataframes[7][dataframes[7]['type']=='y_pred']['value']),2)}, a R2 of:
        {round(R2_metric(dataframes[7][dataframes[7]['type']=='y_test']['value'],
        dataframes[7][dataframes[7]['type']=='y_pred']['value']),2)} and a MAPE of:
        {round(mean_absolute_percentage_error(dataframes[7][dataframes[7]['type']=='y_test']['value'],
        dataframes[7][dataframes[7]['type']=='y_pred']['value']),2)}
        """)
