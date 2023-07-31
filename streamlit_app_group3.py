import streamlit as st
import pandas as pd
#import joblib
#from snowflake.snowpark.session import Session
#import snowflake.snowpark.functions as F
#import snowflake.snowpark.types as T
#from snowflake.snowpark.window import Window
#from sklearn import preprocessing # https://github.com/Snowflake-Labs/snowpark-python-demos/tree/main/sp4py_utilities
#from snowflake.snowpark.functions import col

#import getpass
#import pandas as pd
import matplotlib.pyplot as plt
#from datetime import datetime
#import math
#from datetime import timedelta

#Loading model and data
model=joblib.load('model.joblib')
connection_parameters = { "account": 'hiioykl-ix77996',"user": 'JAVIER',"password": '02B289223r04', "role": "ACCOUNTADMIN","database": "FROSTBYTE_TASTY_BYTES","warehouse": "COMPUTE_WH"}
session = Session.builder.configs(connection_parameters).create()
X_final_scaled=pd.read_csv('x_final_scaled.csv')
unique_location_ids = X_final_scaled['LOCATION_ID'].unique()
# Create a list to store the table data
table_data = []
# Create a DataFrame to store the table data
df_unique_locations_lat_long = pd.DataFrame(columns=["Location ID", "Latitude", "Longitude"])

# Iterate over each unique location ID
for location_id in unique_location_ids:
    location = X_final_scaled[X_final_scaled['LOCATION_ID'] == location_id]
    latitude = location['LAT'].values[0]
    longitude = location['LONG'].values[0]
    df_unique_locations_lat_long = pd.concat([df_unique_locations_lat_long, pd.DataFrame({"Location ID": [location_id],
                                                  "Latitude": [latitude],
                                                  "Longitude": [longitude]})],
                         ignore_index=True)
#import plotly.express as px
st.title('SpeedyBytes 🚚')
st.image('speedybytes_icon2.jpg',  width=600)
# st.image('speedybytes_icon2.jpg',width=600)
tab1,tab2,tab3,tab4,tab5 = st.tabs(["tab1", "One year revenue forecast", "Optimal Shift Timing Recommendation",'tab4','tab5'])
with tab1: #ayrton
    st.header('omg tab1 works yay')


with tab2: #minh
    print('gelo')

with tab3: #javier
    st.header('Optimal Shift Timing Recommendation')
    st.subheader('Want to find out the optimal working hours for your truck?')
    st.subheader('1. Specify your truck details')
    truck_ids = [27,43,28,44,46,47]
    truck_id = st.selectbox("Select your Truck ID", truck_ids)
    if truck_id:
            st.success(f"Your selected Truck ID '{truck_id}' has been saved!")
    st.subheader('2. Specify the number of hours your truck is working for')

    no_of_hours = st.text_input("Enter the number of hours (1-23): ")
    if no_of_hours.isnumeric():
        hours = int(no_of_hours)
        if 1 <= hours <= 23:
            st.success(f"Your input of {hours} hours has been saved!")
        else:
            st.warning("Please enter a number between 1 and 23.")
    elif no_of_hours.strip() != "":
        st.warning("Please enter a valid numeric value.")
  

    st.subheader('3. Specify the date your truck is working on')
    def is_valid_date_format(date_string):
        try:
        # Try to parse the input string as a date
            year, month, day = map(int, date_string.split('-'))
            if 1 <= month <= 12 and 1 <= day <= 31 and year>2000:
                return True
        except ValueError:
            pass
        return False

    date = st.text_input("Enter the date (YYYY-M-D)", key="date_input")
    
    # Validate the user input and display a success message
    if is_valid_date_format(date):
        st.success(f"Your input date '{date}' has been saved !")
    elif date.strip() != "":
        st.warning("Please enter a valid date in the format 'YYYY-M-D'.")
    st.subheader('4. Optimal shift timing will be recommended to you based on the forecasted total average revenue across all locations')
    



with tab4: #natha
    print('nathan')

with tab5: #vibu
    
    df= pd.read_csv('truck_location_df.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df["Date"]=pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Sort the DataFrame by Date
    df.sort_values(by='Date', inplace=True)



# Dashboard title
    st.title("Food Truck Competitor Analysis Dashboard")

# Overview Section
    st.header("Overview")
    df['predicted_earning'] = df['predicted_earning'].apply(lambda x: [int(float(i)) for i in x.strip('[]').split(',')])
    total_sales=df['predicted_earning'].apply(lambda x: sum(x)).sum()

    average_predicted_earnings = df['predicted_earning'].apply(lambda x: sum(x) / len(x)).mean()
    total_locations_visited = df['Num_of_locs'].sum()

    st.write(f"Total Sales (Last 2 weeks): ${round(total_sales, 2)}")
    st.write(f"Average Predicted Earnings: ${round(average_predicted_earnings, 2)}")
    st.write(f"Total Locations Visited: {total_locations_visited}")

# # Sales Performance Section
#     st.header("Sales Performance")
#     fig_sales = px.bar(df, x='Date', y='predicted_earning', title='Predicted Earnings Over Time')
#     st.plotly_chart(fig_sales)

# Efficiency Metrics Section
    st.header("Efficiency Metrics")
    fig_hours = px.pie(df, names='Truck_ID', values='working_hour', title='Distribution of Working Hours')
    fig_shifts = px.bar(df, x='Truck_ID', y='Num_of_locs', title='Number of Locations Visited')
    st.plotly_chart(fig_hours)
    st.plotly_chart(fig_shifts)


# Prioritization Analysis Section
    st.header("Prioritization Analysis")
    df['Priority_Order'] = df['Truck_ID'].rank(method='first')
    fig_priority = px.bar(df, x='Truck_ID', y='predicted_earning', color='Priority_Order',
                      labels={'Truck_ID': 'Truck ID', 'predicted_earning': 'Predicted Earnings'},
                      title='Truck Prioritization Based on Sales Performance')
    st.plotly_chart(fig_priority)

 
    
