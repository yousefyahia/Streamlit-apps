
import streamlit as st
import pandas as pd 
import numpy as np
from scipy.optimize import curve_fit
pd.options.mode.chained_assignment = None
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.metrics import mean_squared_error
import math

# function to load the data 
def load_data(filepath , well_names_col ,well_name , prod_col_name , date_col_name):

    df=pd.read_excel(filepath)
    # return the dataframe with only the two columns [ data , production ], and they must be with the same name
    df=df[df[well_names_col]==well_name]
    df=df[[date_col_name ,prod_col_name]]
    df.columns=["date","production"]
    return df

# remove the outliers form the production column
def remove_outliers(df,col_name="production"):
    df=df[df[col_name]!= 0]
    return df

# data smoothing 
def smooth(df,col_name="production", window_size=100):
    # your code
    df[col_name+"_smoothed"] = df[col_name].rolling(window=window_size, center=True).mean() #moving average
    df=df.dropna()
    return df
# get the days as int 

def get_days(df,col_name="date"):
    
    # your code
    df["days"] = (df[col_name] - df[col_name].min()).dt.days
    return df 

#Models
def hyperbolic(t, qi, di, b):
      return qi / (np.abs((1 + b * di * t))**(1/b))

def exponential(t, qi, di):
        return qi * np.exp(-di * t)

def harmonic(t, qi, di):
        return qi / (1 + di * t)

#Curve Fitting
# fitting the exponential curve
def exponential_fitting( T_nomalized , Q_normalized):
    # apply curve fit funciton and get the parameters
        # write your code here
    params , _ = curve_fit(exponential , T_nomalized , Q_normalized)
        
    # denormalize the parameter to return it to the normal range
      # write your code here
    qi,di = params
    qi=qi*max(Q)
    di=di/max(T)
    return { "qi":qi,"di":di, "b":0}
    
# fitting the hyperbolic curve
def hyperbolic_fitting( T_nomalized , Q_normalized):

    # apply curve fit funciton and get the parameters
        # write your code here
    params , _ = curve_fit(hyperbolic , T_nomalized , Q_normalized)
        
    # denormalize the parameter to return it to the normal range
      # write your code here
    qi,di,b = params
    qi=qi*max(Q)
    di=di/max(T)
    return { "qi":qi,"di":di, "b":b}
    
# fitting the harmonic curve
def harmonic_fitting( T_nomalized , Q_normalized):

    # apply curve fit funciton and get the parameters
        # write your code here
    params , _ = curve_fit(harmonic , T_nomalized , Q_normalized)
        
    # denormalize the parameter to return it to the normal range
      # write your code here
    qi,di = params
    qi=qi*max(Q)
    di=di/max(T)
    return { "qi":qi,"di":di, "b":1}

st.write("""
         # Decline Curve Analysis
          An app to make decline curve analysis using ARP's models for conventional reservoirs.
         - Upload your data
         - Specify a single well data
         - Choose the right parameters
         - The result will show
""")

with st.sidebar:
    with st.spinner():
        st.header("Data Input :green[Production Data]")
        uploaded_file=st.file_uploader(label="Upload Data File", type=["xls" , "xlsx" , "xlsm"], accept_multiple_files=False)
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            type_of_data=st.selectbox(label="Type of data",options=("One Well","Field Data"))
            if type_of_data=="One Well":
                well_names_col=st.selectbox(label="Well Names Columns", options=df.columns)
                well_name=st.selectbox(label="Well Name", options=df[well_names_col].unique())
            
            prod_col_name=st.selectbox(label="Production Column", options=df.columns)
            date_col_name=st.selectbox(label="Date Column", options=df.columns)
            #datefreq=st.selectbox(label="Data Frequency", options=("Daily","Weekly","Monthly"))

try:
    if uploaded_file is not None:
            st.write("#### A sample of the data")
            st.dataframe(data=df.head())
            cut_df=load_data(uploaded_file, well_names_col, well_name, prod_col_name, date_col_name)
            clean_df=remove_outliers(cut_df)
            
            st.write("### Smoothing the data using moving average")
            window_size=st.slider(label="Window Size",min_value=0,max_value=200,step=10,value=100)
            
            #outlier=st.slider(label="Removing Outliers",min_value=0,max_value=10,step=1,value=1)
            
            smoothed_df=smooth(clean_df,window_size=window_size)
            df_final = get_days(smoothed_df)
            st.line_chart(df_final,x="days",y=["production","production_smoothed"])
            #fig=px.line(smoothed_df,x="date",y=["production","production_smoothed"])
            #st.plotly_chart(fig)
            
                
            # getting only the days as T , production_smoothed as Q then normalize
            T = df_final["days"]
            Q=  df_final["production_smoothed"]
            T_nomalized=T/max(T)
            Q_normalized=Q/max(Q)
            
            
            #applying the three models
            paramsexp = exponential_fitting(T_nomalized,Q_normalized)
            Q_exp = exponential(T , paramsexp["qi"] , paramsexp["di"])
            EXP_MSE = mean_squared_error(Q, Q_exp)
            EXP_RMSE = math.sqrt(EXP_MSE)
            
            paramshyp = hyperbolic_fitting(T_nomalized,Q_normalized)
            Q_hyp = hyperbolic(T , paramshyp["qi"] , paramshyp["di"], paramshyp["b"])
            HYP_MSE = mean_squared_error(Q, Q_hyp)
            HYP_RMSE = math.sqrt(HYP_MSE)
            
            paramshar = harmonic_fitting(T_nomalized,Q_normalized)
            Q_har = harmonic(T , paramshar["qi"] , paramshar["di"])
            HAR_MSE = mean_squared_error(Q, Q_har)
            HAR_RMSE = math.sqrt(HAR_MSE)
            
            st.subheader("ARP's Models Fitted")
            
            Q_df=pd.DataFrame({"Days":T,"Original_Smoothed":Q,"Exponential":Q_exp,"Hyperbolic":Q_hyp,"Harmonic":Q_har})
            #fig=px.line(Q_df,x="Days",y=["Smoothed","Exponential","Hyperbolic","Harmonic"])
            #st.plotly_chart(fig)
            st.line_chart(Q_df,x="Days",y=["Original_Smoothed","Exponential","Hyperbolic","Harmonic"])
            
            dff=pd.DataFrame([paramsexp,paramshyp,paramshar],index=["Exponential","Hyperbolic","Harmonic"])
            dff.columns=["Qi","Di","b"]
            dff["RMSE"]=[EXP_RMSE,HYP_RMSE,HAR_RMSE]
            st.subheader("Models Parameter")
            st.table(data=dff)
            
            st.success("Best fit model is "+dff["RMSE"].idxmin())
            
except:
    st.warning("Complete Columns Choice")
