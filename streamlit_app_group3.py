import streamlit as st
import pandas as pd
import joblib
from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark.window import Window
from sklearn import preprocessing # https://github.com/Snowflake-Labs/snowpark-python-demos/tree/main/sp4py_utilities
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
import openrouteservice as ors
import operator
from functools import reduce

st.set_page_config(layout="wide")
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
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Routing Map", "One year revenue forecast", "Optimal Shift Timing Recommendation",'tab4','tab5'])

with tab1: #Nathan
    X_final_scaled=pd.read_csv('x_final_scaled.csv')
    truck_location_df=pd.read_csv('truck_manager_merged_df.csv')
    
    truck_location_df["location_visited"] = truck_location_df["location_visited"].apply(eval)
    truck_location_df["predicted_earning"] = truck_location_df["predicted_earning"].apply(eval)
    truck_df_exploded = truck_location_df.explode(["location_visited", "predicted_earning"], ignore_index=True)
    truck_df_exploded["shift"] = truck_df_exploded.groupby("Truck_ID").cumcount() + 1
    all_locations = truck_df_exploded["location_visited"].tolist()
    
    # Find the latitude and longitude for each location
    location_lat_long = {}
    for location in all_locations:
        location_info = X_final_scaled[X_final_scaled["LOCATION_ID"] == location]
        if not location_info.empty:
            lat = location_info["LAT"].values[0]
            long = location_info["LONG"].values[0]
            location_lat_long[location] = (lat, long)
    
    def get_lat_long(location_list):
        lat_list = []
        long_list = []
        for loc in location_list:
            if loc in location_lat_long:
                lat, long = location_lat_long[loc]
                lat_list.append(lat)
                long_list.append(long)
            else:
                lat_list.append(None)
                long_list.append(None)
        return lat_list, long_list
    
    lat, lon = get_lat_long(truck_df_exploded["location_visited"])
    
    truck_df_exploded["Lat"] = lat
    truck_df_exploded["Lon"] = lon
    
    
    #ors client
    ors_client = ors.Client(key='5b3ce3597851110001cf6248d282bc5c5d534216a412fa4e36a497e3')
    
    # Define a function to get the route between two points using ORS
    def get_route(start_point, end_point):
            radius = 500  # 10 kilometers
            profile = 'driving-car'
            try:
                    # Get the route between the start and end points
                    route = ors_client.directions(
                    coordinates=[start_point, end_point],
                    profile=profile,
                    format='geojson',
                    radiuses=[radius, radius]
                    )
                    return route
    
            except ors.exceptions.ApiError as e:
                    print(e)
                    return None
    
    def filter_truck_data(truck_id):
        return truck_df_exploded[truck_df_exploded['Truck_ID'] == truck_id]
    
    def create_map(selected_truck_ids):
        # Check if truck IDs are selected
        if selected_truck_ids:

                # Get the data for the first selected truck
                selected_truck_data = filter_truck_data(selected_truck_ids[0])

                # FOLIUM MAP
                m = folium.Map(location=[selected_truck_data['Lat'].iloc[0], selected_truck_data['Lon'].iloc[0]], zoom_start=13)

                # Iterate through selected truck IDs to display each truck route
                for selected_truck_id in selected_truck_ids:
                        # Filter truck data based on the selected truck ID
                        selected_truck_data = filter_truck_data(selected_truck_id)

                        place_lat = selected_truck_data['Lat'].astype(float).tolist()
                        place_lng = selected_truck_data['Lon'].astype(float).tolist()

                        points = []

                        # read a series of points from coordinates and assign them to points object
                        for i in range(len(place_lat)):
                                points.append([place_lat[i], place_lng[i]])

                        # Choose a different polyline color for each truck route
                        colors = ['darkred', 'black', 'blue', 'orange', 'lightblue', 'lightgreen', 'purple', 'gray', 'white', 'cadetblue', 'darkgreen', 'pink', 'darkblue', 'lightgray', 'beige']
                        color_index = available_trucks.index(selected_truck_id)
                        polyline_color = colors[color_index % len(colors)]

                        folium.PolyLine(points, color=polyline_color, dash_array='5', opacity='0.6',
                                        tooltip=f'Truck Route {selected_truck_id}').add_to(m)


                        # Specify marker color based on polyline color
                        marker_color = 'white' if polyline_color != 'white' else 'black'

                        # Add markers for each truck location
                        for index, lat in enumerate(place_lat):
                                folium.Marker([lat, place_lng[index]],
                                        popup=('Truck Location {} \n '.format(index)),
                                        icon=folium.Icon(color=polyline_color, icon_color=marker_color, prefix='fa', icon='truck')
                                        ).add_to(m)
                        
                        for i in range(len(place_lat) - 1):
                                start_point = [place_lng[i], place_lat[i]]  # Corrected order: [long, lat]
                                end_point = [place_lng[i + 1], place_lat[i + 1]]  # Corrected order: [long, lat]

                                # Check if the start point and end point are the same
                                if start_point != end_point:
                                        # Get the route between two consecutive points
                                        route = get_route(start_point, end_point)

                                        # Check if the route is found
                                        if route is not None:

                                                # print(route_coords)
                                                waypoints = list(dict.fromkeys(reduce(operator.concat, list(map(lambda step: step['way_points'], route['features'][0]['properties']['segments'][0]['steps'])))))

                                                folium.PolyLine(locations=[list(reversed(coord)) for coord in route['features'][0]['geometry']['coordinates']], color=polyline_color).add_to(m)

                                                # folium.PolyLine(locations=[list(reversed(route['features'][0]['geometry']['coordinates'][index])) for index in waypoints], color="red").add_to(m)
                
                                        else:
                                                print(f"No route found between {start_point} and {end_point}")
                                else:
                                        print(f"Start point and end point are the same: {start_point}")

                        # Convert 'predicted_earning' to numeric (float) and round to 0 decimal places
                        truck_df_exploded['predicted_earning'] = truck_df_exploded['predicted_earning'].astype(int)

                        # Calculate total revenue for each truck with 0 decimal places
                        total_revenue_per_truck = truck_df_exploded.groupby('Truck_ID')['predicted_earning'].sum()

                        truck_info = {
                        'Truck Manager Name': [],  # Add the truck manager name as the first column
                        'City': [],
                        'Truck ID🚚': [],  # Updated column name
                        'Number of Shifts': [],
                        'Total Revenue💵': [],
                        'Truck Colour': []  # Updated column name
                        }

                # Populate the truck_info dictionary with data
                for selected_truck_id in selected_truck_ids:
                        # Get the truck manager name for the selected truck ID
                        selected_truck_manager = truck_location_df[truck_location_df['Truck_ID'] == selected_truck_id]['Name'].iloc[0]
                        truck_info['Truck Manager Name'].append(selected_truck_manager)
                        selected_truck_manager = truck_location_df[truck_location_df['Truck_ID'] == selected_truck_id]['City'].iloc[0]
                        truck_info['City'].append(selected_truck_manager)
                        truck_info['Truck ID🚚'].append(selected_truck_id)
                        selected_truck_data = filter_truck_data(selected_truck_id)
                        num_shifts = selected_truck_data['shift'].max()
                        truck_info['Number of Shifts'].append(num_shifts)
                        total_revenue = total_revenue_per_truck.get(selected_truck_id, 0)
                        truck_info['Total Revenue💵'].append(total_revenue)
                        color_index = available_trucks.index(selected_truck_id)
                        truck_color = colors[color_index % len(colors)]
                        truck_info['Truck Colour'].append(truck_color)
                        
                # Convert the truck_info dictionary to a DataFrame
                truck_info_df = pd.DataFrame(truck_info)

                # Create a new column "Colour" with background colors
                truck_info_df['Colour'] = truck_info_df['Truck Colour']

                # Display the truck information table with colored cells and black text for color names
                st.subheader('Truck Information')

                # Custom CSS style to display colored cells with white text and black text for color names
                def style_color_cells(val):
                        return f'background-color: {val}; color: {val}; text-align: center;'

                # Apply the custom style to the 'Truck Color' column
                styled_truck_info_df = truck_info_df.style.applymap(style_color_cells, subset=['Colour'])

                # Display the styled DataFrame using st.dataframe
                st.dataframe(styled_truck_info_df)

                # Display the map with the selected truck routes
        st_folium(m, width=1500)
    
    ## STREAMLIT APP
    
    st.header('Routing Map🗺️')
    st.subheader('Choose a Truck Manager')
    
    # Extract a list of unique names from the "Name" column
    unique_names = truck_location_df['Name'].unique().tolist()
    
    # Create a multiselect widget to choose one or more names
    selected_names = st.multiselect("Select Truck Manager Name", unique_names)
    
    # Filter the available trucks based on the chosen names
    available_trucks = truck_location_df[truck_location_df['Name'].isin(selected_names)]['Truck_ID'].tolist()
    
    # Check if 'prev_selected_truck_ids' exists in session state
    if 'prev_selected_truck_ids' not in st.session_state:
        st.session_state.prev_selected_truck_ids = []
    
    # Display the truck selection section
    st.subheader('Choose a truck 🚚')
    selected_truck_ids = st.multiselect("Select Truck IDs 🌮🍦", available_trucks)
    
    # Add a "Run Map" button
    with st.form("RunMapForm"):
            st.form_submit_button("Run Map")
    
            if selected_truck_ids:
                    if selected_truck_ids != st.session_state.prev_selected_truck_ids:
                            # Save the current selected truck IDs to session state
                            selected_truck_ids_str = ', '.join(str(truck_id) for truck_id in selected_truck_ids)
                            st.success(f"Your selected Truck IDs {selected_truck_ids_str} have been saved!")
                            # Create the map and display truck routes
                            create_map(selected_truck_ids)
                    else:
                            st.info("Selected truck IDs have not changed. The map has not been changed.")
                            create_map(selected_truck_ids)
            else:
                    st.info("No truck IDs have been selected.")
    
    map_placeholder = st.empty()





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
    
        # wdf=session.sql("Select * from ANALYTICS.WEATHER_DATA_API")
        # wdf=wdf.withColumn("H",F.substring(wdf["TIME"], 12, 2).cast("integer"))
        # wdf=wdf.withColumn("DATE",F.substring(wdf["TIME"], 0, 10))
        # wdf=wdf.select("WEATHERCODE","LOCATION_ID","H","DATE" )
        # wdf=wdf.to_pandas()
        wdf=pd.read_csv('wdf.csv')
    
        average_revenue_for_hour=pd.DataFrame(columns=['TRUCK_ID','HOUR','AVERAGE REVENUE PER HOUR'])
        #TODO for loop testing - change hour, sum1,sum2,weathercode
        for x in range(8,24):
            # session.use_schema("RAW_POS")
            # query = "SELECT * FROM TRUCK WHERE TRUCK_ID = '{}'".format(truck_id)
            # truck_df=session.sql(query).toPandas()
            truck_df=pd.read_csv('truck_df.csv')
            truck_df=(truck_df[truck_df['TRUCK_ID']==truck_id])
            
            city = truck_df['PRIMARY_CITY'].iloc[0]
        
            #  query = "SELECT * FROM LOCATION WHERE CITY = '{}'".format(city)
            #  location_df=session.sql(query).toPandas()
            location_df=pd.read_csv('location_df.csv')
            location_df = location_df[location_df['CITY']==city]
            city_locations = location_df.merge(df_unique_locations_lat_long, left_on='LOCATION_ID', right_on='Location ID', how='inner')
            city_locations = city_locations[['LOCATION_ID','Latitude','Longitude']]
            city_locations.rename(columns={"Latitude": "LAT"},inplace=True)
            city_locations.rename(columns={"Longitude": "LONG"},inplace=True)
        
            loc_checker = city_locations.copy()
            loc_checker['DATE'] = date
            
            loc_checker['DATE']=pd.to_datetime(loc_checker['DATE'],format='%Y-%m-%d')
            loc_checker['DATE']=loc_checker['DATE'].astype('str')
            weadf = pd.merge(wdf, loc_checker, on=['LOCATION_ID', 'DATE']).drop_duplicates()
            input_df['date']=input_df['date'].astype('str')
            input_df['HOUR']=x
            new_df = pd.merge(input_df, weadf,  how='left', left_on=['date','HOUR'], right_on = ['DATE','H']).drop_duplicates()
            
        #     sales_pred=session.sql("select * from ANALYTICS.SALES_PREDICTION").to_pandas() #this is the problem.
        
            sales_pred.to_csv('sales_pred.csv')
            #sales_pred=pd.read_csv('sales_pred.csv')
            X_final_scaled=pd.read_csv('x_final_scaled.csv')
            X_final_scaled=X_final_scaled.merge(sales_pred["l_w5i8_DATE"].astype(str).str[:4].rename('YEAR'), left_index=True, right_index=True)
            filtered_df = X_final_scaled[(X_final_scaled['TRUCK_ID'] == truck_id) & (X_final_scaled['YEAR'].astype(int) == input_df['YEAR'][0].astype(int))]
            filtered_df = filtered_df[['TRUCK_ID', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 
                                     'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 
                                     'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 
                                     'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 
                                     'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 
                                     'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']]
            merge_df = new_df.merge(filtered_df, left_on='TRUCK_ID', right_on='TRUCK_ID', how='inner').drop_duplicates()
        
            filtered_df = X_final_scaled[(X_final_scaled['TRUCK_ID'] == truck_id) & (X_final_scaled['HOUR'] == x) & (X_final_scaled['YEAR'].astype(int) == input_df['YEAR'][0].astype(int))]
        
            sum_prev_year=filtered_df['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].mean()
            sum_day_of_week=filtered_df['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].mean()
            
            filtered_df = X_final_scaled[(X_final_scaled['TRUCK_ID'] == truck_id) & 
                                        (X_final_scaled['HOUR'] == x) &
                                        (X_final_scaled['YEAR'].astype(int) == input_df['YEAR'][0].astype(int))]
        
            filtered_df = filtered_df[['TRUCK_ID', 'MONTH','DAY', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', 'YEAR']]
            filtered_df['YEAR'] = filtered_df['YEAR'].astype(int)
        
             #Perform the left merge based on truck_id and date
            merged_df = pd.merge(merge_df, filtered_df, on=['TRUCK_ID', 'YEAR', 'MONTH', 'DAY'], how='left').drop_duplicates()
            merged_df = merged_df.sort_values(by=['TRUCK_ID', 'YEAR', 'MONTH', 'DAY'])
        
        
            filtered_df['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'] = filtered_df['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].astype(float)
            filtered_df['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'] = filtered_df['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].astype(float)
        
            merged_df = merged_df.fillna({ 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE':(sum_prev_year)})
            merged_df = merged_df.fillna({ 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE':(sum_day_of_week)})
        
            # Reset the index of the merged DataFrame
            merged_df = merged_df.reset_index(drop=True)
            merged_df['LOCATION_ID'] = merged_df['LOCATION_ID'].astype(int)
            initial_df_position = merged_df[['TRUCK_ID', 'MONTH', 'HOUR', 'DOW', 'DAY', 'PUBLIC_HOLIDAY', 'LAT', 'LONG', 'LOCATION_ID', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', 'WEATHERCODE', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']]
        
            predictions = model.predict(initial_df_position)
            initial_df_position['Predicted'] = predictions
           
        
            
            data_for_avg_revenue=[truck_id,x,initial_df_position['Predicted'].mean()]
            average_revenue_for_hour.loc[len(average_revenue_for_hour)]=data_for_avg_revenue
            
       
        
        
        #Initialize variables
        average_revenue_for_hour['rolling_average']=0
        max_revenue = 0
        optimal_hours = []
        for i in range(len(average_revenue_for_hour) - 4):
            
            total_revenue=0
            # Calculate the total revenue for the current 5-hour window
            total_revenue = average_revenue_for_hour.loc[i:i+4, 'AVERAGE REVENUE PER HOUR'].sum()
            average_revenue_for_hour['rolling_average'].loc[i] = total_revenue
            
             # Check if the current total revenue is greater than the previous maximum
            if total_revenue > max_revenue:
                max_revenue = total_revenue
                optimal_hours = average_revenue_for_hour.loc[i:i+4, 'HOUR'].tolist()
        values=[1,2]



    
        return values
    if st.button("Run Algorithm"):
        # Display a loading message while the algorithm is running
        with st.spinner("Running the algorithm..."):
            output = find_optimal_hour(truck_id,date,no_of_hours)
    
        # Show the output once the algorithm is done
        st.success("Algorithm completed!")
        st.write("Output:")
        st.text("Optimal Hours: "+str(output[0]))
        st.text("Maximum Revenue: "+str(output[1]))




    



with tab4: #Aryton
    print('Aryton')


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

 
    
