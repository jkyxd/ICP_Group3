import streamlit as st
import pandas as pd
import joblib
from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark.window import Window
#from sklearn import preprocessing # https://github.com/Snowflake-Labs/snowpark-python-demos/tree/main/sp4py_utilities
from snowflake.snowpark.functions import col
import plotly.express as px
import getpass
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import math
from datetime import timedelta

import folium
from streamlit_folium import st_folium

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


    def find_optimal_hour(truck_id,date,no_of_hours):
        # user input
        
        datetime_object = datetime.strptime(date, '%Y-%m-%d')
        
        input_df = pd.DataFrame({'TRUCK_ID': [truck_id],'date': [date]})
    
        #seperate date into month, dow, day, public_holiday
        input_df['date'] = pd.to_datetime(input_df['date'])
        input_df['MONTH'] = input_df['date'].dt.month
        input_df['DOW'] = input_df['date'].dt.weekday
        input_df['DAY'] = input_df['date'].dt.day
        input_df['WOM'] = (input_df['DAY'] - 1) // 7 + 1
        input_df['YEAR'] = input_df['date'].dt.year
        
        public_holidays = [
            {'Month': 7, 'Day': 4, 'DOW': None, 'WOM': None},  # 4th of July
            {'Month': 12, 'Day': 24, 'DOW': None, 'WOM': None},  # Christmas Eve
            {'Month': 12, 'Day': 25, 'DOW': None, 'WOM': None},  # Christmas Day
            {'Month': 10, 'Day': None, 'DOW': '0', 'WOM': 2},  # Columbus Day (second Monday in October)
            {'Month': 6, 'Day': 19, 'DOW': None, 'WOM': None},  # Juneteenth
            {'Month': 9, 'Day': None, 'DOW': '0', 'WOM': 1},  # Labor Day (first Monday in September)
            {'Month': 1, 'Day': None, 'DOW': '0', 'WOM': 3},  # Martin Luther King, Jr. Day (third Monday in January)
            {'Month': 5, 'Day': None, 'DOW': '0', 'WOM': -1},  # Memorial Day (last Monday in May)
            {'Month': 1, 'Day': 1, 'DOW': None, 'WOM': None},  # New Year's Day
            {'Month': 12, 'Day': 31, 'DOW': None, 'WOM': None},  # New Year's Eve
            {'Month': 11, 'Day': None, 'DOW': '3', 'WOM': 4},  # Thanksgiving Day (fourth Thursday in November)
            {'Month': 11, 'Day': None, 'DOW': '2', 'WOM': 4},  # Thanksgiving Eve (fourth Wednesday in November)
            {'Month': 2, 'Day': 14, 'DOW': None, 'WOM': None},  # Valentine's Day
            {'Month': 11, 'Day': 11, 'DOW': None, 'WOM': None},  # Veterans Day
            {'Month': 10, 'Day': 31, 'DOW': None, 'WOM': None},  # Halloween
            {'Month': 3, 'Day': 17, 'DOW': None, 'WOM': None},  # St. Patrick's Day
            {'Month': 11, 'Day': 25, 'DOW': '4', 'WOM': None},  # Black Friday
            {'Month': 12, 'Day': 26, 'DOW': None, 'WOM': None},  # Boxing Day
        ]
        
        # Iterate over the public holidays and create the 'public_holiday' column
        input_df['PUBLIC_HOLIDAY'] = 0
        for holiday in public_holidays:
            month_mask = input_df['date'].dt.month == holiday['Month']
            day_mask = input_df['date'].dt.day == holiday['Day']
            dow_mask = input_df['date'].dt.dayofweek == int(holiday['DOW']) if holiday['DOW'] is not None else True
            wom_mask = (input_df['date'].dt.day - 1) // 7 + 1 == holiday['WOM'] if holiday['WOM'] is not None else True
            mask = month_mask & day_mask & dow_mask & wom_mask
            input_df.loc[mask, 'PUBLIC_HOLIDAY'] = 1
    
        wdf=session.sql("Select * from ANALYTICS.WEATHER_DATA_API")
        wdf=wdf.withColumn("H",F.substring(wdf["TIME"], 12, 2).cast("integer"))
        wdf=wdf.withColumn("DATE",F.substring(wdf["TIME"], 0, 10))
        wdf=wdf.select("WEATHERCODE","LOCATION_ID","H","DATE" )
        wdf=wdf.to_pandas()
    
        average_revenue_for_hour=pd.DataFrame(columns=['TRUCK_ID','HOUR','AVERAGE REVENUE PER HOUR'])
        #TODO for loop testing - change hour, sum1,sum2,weathercode
        for x in range(8,24):
            # session.use_schema("RAW_POS")
            # query = "SELECT * FROM TRUCK WHERE TRUCK_ID = '{}'".format(truck_id)
            # truck_df=session.sql(query).toPandas()
            truck_df=pd.read_csv('truck_df.csv')
            truck_df=(truck_df[truck_df['TRUCK_ID']==truck_id])
            
            city = truck_df['PRIMARY_CITY'].iloc[0]
        
#             # query = "SELECT * FROM LOCATION WHERE CITY = '{}'".format(city)
#             # location_df=session.sql(query).toPandas()
#             location_df=pd.read_csv('location_df.csv')
#             #location_df = location_df[location_df['CITY']==city]
#             city_locations = location_df.merge(df_unique_locations_lat_long, left_on='LOCATION_ID', right_on='Location ID', how='inner')
#             city_locations = city_locations[['LOCATION_ID','Latitude','Longitude']]
#             city_locations.rename(columns={"Latitude": "LAT"},inplace=True)
#             city_locations.rename(columns={"Longitude": "LONG"},inplace=True)
        
#             loc_checker = city_locations.copy()
#             loc_checker['DATE'] = date
            
#             loc_checker['DATE']=pd.to_datetime(loc_checker['DATE'],format='%Y-%m-%d')
#             loc_checker['DATE']=loc_checker['DATE'].astype('str')
#             weadf = pd.merge(wdf, loc_checker, on=['LOCATION_ID', 'DATE']).drop_duplicates()
#             input_df['date']=input_df['date'].astype('str')
#             input_df['HOUR']=x
#             new_df = pd.merge(input_df, weadf,  how='left', left_on=['date','HOUR'], right_on = ['DATE','H']).drop_duplicates()
            
#             #sales_pred=session.sql("select * from ANALYTICS.SALES_PREDICTION").to_pandas() #this is the problem.
        
#             #sales_pred.to_csv('sales_pred.csv')
#             sales_pred=pd.read_csv('sales_pred.csv')
#             X_final_scaled=pd.read_csv('x_final_scaled.csv')
#             X_final_scaled=X_final_scaled.merge(sales_pred["l_w5i8_DATE"].astype(str).str[:4].rename('YEAR'), left_index=True, right_index=True)
#             filtered_df = X_final_scaled[(X_final_scaled['TRUCK_ID'] == truck_id) & (X_final_scaled['YEAR'].astype(int) == input_df['YEAR'][0].astype(int))]
#             filtered_df = filtered_df[['TRUCK_ID', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 
#                                     'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 
#                                     'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 
#                                     'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 
#                                     'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 
#                                     'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']]
#             merge_df = new_df.merge(filtered_df, left_on='TRUCK_ID', right_on='TRUCK_ID', how='inner').drop_duplicates()
        
#             filtered_df = X_final_scaled[(X_final_scaled['TRUCK_ID'] == truck_id) & (X_final_scaled['HOUR'] == x) & (X_final_scaled['YEAR'].astype(int) == input_df['YEAR'][0].astype(int))]
        
#             sum_prev_year=filtered_df['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].mean()
#             sum_day_of_week=filtered_df['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].mean()
            
#             filtered_df = X_final_scaled[(X_final_scaled['TRUCK_ID'] == truck_id) & 
#                                         (X_final_scaled['HOUR'] == x) &
#                                         (X_final_scaled['YEAR'].astype(int) == input_df['YEAR'][0].astype(int))]
        
#             filtered_df = filtered_df[['TRUCK_ID', 'MONTH','DAY', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', 'YEAR']]
#             filtered_df['YEAR'] = filtered_df['YEAR'].astype(int)
        
#             #Perform the left merge based on truck_id and date
#             merged_df = pd.merge(merge_df, filtered_df, on=['TRUCK_ID', 'YEAR', 'MONTH', 'DAY'], how='left').drop_duplicates()
#             merged_df = merged_df.sort_values(by=['TRUCK_ID', 'YEAR', 'MONTH', 'DAY'])
        
        
#             filtered_df['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'] = filtered_df['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].astype(float)
#             filtered_df['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'] = filtered_df['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].astype(float)
        
#             merged_df = merged_df.fillna({ 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE':(sum_prev_year)})
#             merged_df = merged_df.fillna({ 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE':(sum_day_of_week)})
        
#             # Reset the index of the merged DataFrame
#             merged_df = merged_df.reset_index(drop=True)
#             merged_df['LOCATION_ID'] = merged_df['LOCATION_ID'].astype(int)
#             initial_df_position = merged_df[['TRUCK_ID', 'MONTH', 'HOUR', 'DOW', 'DAY', 'PUBLIC_HOLIDAY', 'LAT', 'LONG', 'LOCATION_ID', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', 'WEATHERCODE', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']]
        
#             predictions = model.predict(initial_df_position)
#             initial_df_position['Predicted'] = predictions
           
        
            
#             data_for_avg_revenue=[truck_id,x,initial_df_position['Predicted'].mean()]
#             average_revenue_for_hour.loc[len(average_revenue_for_hour)]=data_for_avg_revenue
            
       
        
        
#         # Initialize variables
#         average_revenue_for_hour['rolling_average']=0
#         max_revenue = 0
#         optimal_hours = []
#         for i in range(len(average_revenue_for_hour) - 4):
            
#             total_revenue=0
#             # Calculate the total revenue for the current 5-hour window
#             total_revenue = average_revenue_for_hour.loc[i:i+4, 'AVERAGE REVENUE PER HOUR'].sum()
#             average_revenue_for_hour['rolling_average'].loc[i] = total_revenue
            
#              # Check if the current total revenue is greater than the previous maximum
#             if total_revenue > max_revenue:
#                 max_revenue = total_revenue
#                 optimal_hours = average_revenue_for_hour.loc[i:i+4, 'HOUR'].tolist()
#         values=[1,2]



    
#         return values
#     if st.button("Run Algorithm"):
#         # Display a loading message while the algorithm is running
#         with st.spinner("Running the algorithm..."):
#             output = find_optimal_hour(truck_id,date,no_of_hours)
    
#         # Show the output once the algorithm is done
#         st.success("Algorithm completed!")
#         st.write("Output:")
#         st.text("Optimal Hours: "+str(output[0]))
#         st.text("Maximum Revenue: "+str(output[1]))




    



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

 
    
