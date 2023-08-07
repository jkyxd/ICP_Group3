import streamlit as st
import joblib
import concurrent.futures
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
from datetime import timedelta
import math

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
# Create a list to store the tables data
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
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Routing Map", "Revenue Forecasting", "Optimal Shift Timing Recommendation",'tab4','tab5'])

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
    ors_client = ors.Client(key='5b3ce3597851110001cf624817eb9bc1474c4917b9dda7114d579034')
    
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
    import calendar
    def get_dates(year, month):
    
        """ Function Documentation
            Generate a list of dates for the given month and year.
    
            Returns:
                list: A list of date strings in the format 'YYYY-MM-DD'.
            """
    
        try:
            # Get the number of days in the given month
            num_days = calendar.monthrange(year, month)[1]
    
            # Generate a list of dates for the given month and year
            dates = [f"{year}-{month:02d}-{day:02d}" for day in range(1, num_days + 1)]
    
            return dates
    
        except Exception as e:
            print(f"An error occurred while retrieving list of dates: {e}")
            return pd.DataFrame()
            
    import datetime
    import holidays
    def add_public_holiday_column(df, date_column): #ONLY USED IN CONJUCTION WITH upload_input_data_to_snowflake() function
    
        """ Function Documentation
            Add a column to the DataFrame indicating whether each date is a public holiday using imported library.
    
            Returns:
                pandas.DataFrame: The DataFrame with an additional column "PUBLIC_HOLIDAY".
            """
    
        try:
            # Create an instance of the holiday class for the appropriate country
            country_code = 'US'  # Replace with the appropriate country code
            holiday_list = holidays.CountryHoliday(country_code)
    
            # Convert the date column to datetime if it's not already in that format
            df[date_column] = pd.to_datetime(df[date_column])
    
            # Create a new column "PUBLIC_HOLIDAY" and set initial values to 0
            df['PUBLIC_HOLIDAY'] = 0
    
            # Iterate over each date in the date column
            for date in df[date_column]:
                # Check if the date is a public holiday
                if date in holiday_list:
                    # Set the value of "PUBLIC_HOLIDAY" to 1 if it is a public holiday
                    df.loc[df[date_column] == date, 'PUBLIC_HOLIDAY'] = 1
    
            return df
    
        except Exception as e:
            print(f"An error occurred while retrieving public holiday information: {e}")
            return pd.DataFrame()
    
    def get_location_id(truck_id): #ONLY USED IN CONJUCTION WITH upload_input_data_to_snowflake() function
        """
        Get location ID, city, and region for a given truck ID.
    
        Returns:
            pandas.DataFrame: A DataFrame containing the location ID, city, and region for the given truck ID.
        """
        try:
            # Set the schema to "RAW_POS"
            session.use_schema("RAW_POS")
    
            # Query the Truck table to get the city for the given truck ID
            query = "SELECT PRIMARY_CITY FROM TRUCK WHERE TRUCK_ID = {}".format(truck_id)
            city_df = session.sql(query).toPandas()
            city = city_df['PRIMARY_CITY'].iloc[0]
    
            # Query the Location table to get the location ID for the city
            query = "SELECT LOCATION_ID FROM LOCATION WHERE CITY = '{}'".format(city)
            location_df = session.sql(query).toPandas()
    
            # Add the truck ID to the DataFrame
            location_df['TRUCK_ID'] = truck_id
    
            return location_df
    
        except Exception as e:
            print(f"An error occurred while retrieving location information: {e}")
            return pd.DataFrame()
    
    def get_hours_df(truck_id):
        """
        Get a DataFrame with hours and the corresponding truck ID.
    
        Returns:
            pandas.DataFrame: A DataFrame containing the truck ID and hours from 0 to 23.
        """
        try:
            # Create a list of hours from 0 to 23
            hours = list(range(24))
    
            # Create a dictionary with column names and corresponding data
            data = {'TRUCK_ID': [truck_id] * 24, 'HOUR': hours}
    
            # Create a new DataFrame from the dictionary
            new_df = pd.DataFrame(data)
    
            return new_df
    
        except Exception as e:
            print(f"An error occurred while generating the hours DataFrame: {e}")
            return pd.DataFrame()
    
    def upload_input_data_to_snowflake():
        """
        Uploads input data to Snowflake for the food truck revenue trend forecast.
    
        This function performs various data preprocessing and joins, and then writes the final DataFrame
        containing input data to the "Trend_Input_Data" table in the "ANALYTICS" schema of the "FROSTBYTE_TASTY_BYTES" database.
    
        Note: This function is intended for one-time use to upload data to Snowflake.
    
        Returns:
            None
        """
        try:
            # Set the schema to "ANALYTICS"
            session.use_schema("ANALYTICS")
    
            # Load data from the "Sales_Forecast_Training_Data" table
            X_final_scaled = session.sql('Select * from "Sales_Forecast_Training_Data"').to_pandas()
            X_final_scaled.rename(columns={"Profit": "Revenue"}, inplace=True)
    
            # Load data from the "ANALYTICS.SALES_PREDICTION" table
            sales_pred = session.sql("select * from ANALYTICS.SALES_PREDICTION").to_pandas()
    
            # Merge the dataframes based on the "l_w5i8_DATE" column
            X_final_scaled = X_final_scaled.merge(sales_pred["l_w5i8_DATE"].astype(str).str[:4].rename('YEAR'), left_index=True, right_index=True)
    
            # Filter data for specific truck IDs and years
            truck_ids = [27, 28, 43, 44, 46, 47]
            years = ['2020', '2021', '2022']
            X_final_scaled = X_final_scaled[(X_final_scaled['TRUCK_ID'].isin(truck_ids)) & (X_final_scaled['YEAR'].isin(years))]
    
            # Set the schema to "ANALYTICS"
            session.use_schema("ANALYTICS")
    
            # Load data from the "weadf_trend" table
            weadf = session.sql('select * from "weadf_trend"').to_pandas()
            weadf['DATE'] = pd.to_datetime(weadf['DATE'])
    
            # Process data for each truck, year, and month
            for truck in truck_ids:
                for year in years:
                    for month in range(1, 13):
                        current_df = X_final_scaled[
                            (X_final_scaled['TRUCK_ID'] == truck) & 
                            (X_final_scaled['MONTH'] == month) & 
                            (X_final_scaled['YEAR'] == year)
                        ]
    
                        # Generate dates for the given month and year
                        current_dates = get_dates(int(year), month)
                        main_df = pd.DataFrame({'TRUCK_ID': [truck] * len(current_dates), 'DATE': current_dates})
    
                        # Add public holiday column to main_df
                        main_df = add_public_holiday_column(main_df, 'DATE')
    
                        # Join location data
                        main_df = pd.merge(main_df, get_location_id(truck), how='left', left_on='TRUCK_ID', right_on='TRUCK_ID').drop_duplicates()
    
                        # Join hours data
                        main_df = pd.merge(main_df, get_hours_df(truck), how='left', left_on='TRUCK_ID', right_on='TRUCK_ID').drop_duplicates()
    
                        # Join weather data
                        main_df = pd.merge(main_df, weadf,  how='left', left_on=['LOCATION_ID', 'HOUR', 'DATE'], right_on=['LOCATION_ID', 'H', 'DATE']).drop_duplicates()
                        main_df = main_df.drop('H', axis=1).drop_duplicates().dropna()
    
                        # Additional data preprocessing
                        main_df['DATE'] = pd.to_datetime(main_df['DATE'])
                        main_df['MONTH'] = main_df['DATE'].dt.month
                        main_df['DOW'] = main_df['DATE'].dt.weekday
                        main_df['DAY'] = main_df['DATE'].dt.day
                        main_df['YEAR'] = main_df['DATE'].dt.year
                        main_df['DATE'] = main_df['DATE'].astype(str)
    
                        # Join encoded data
                        encoded_X = current_df[['TRUCK_ID', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']].drop_duplicates()
                        main_df = pd.merge(main_df, encoded_X,  how='left', left_on=['TRUCK_ID'], right_on=['TRUCK_ID']).drop_duplicates()
    
                        # Join sum data
                        sum_X = current_df[['TRUCK_ID', 'MONTH', 'HOUR', 'DAY', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE']]
                        main_df = pd.merge(main_df, sum_X,  how='left', left_on=['TRUCK_ID', 'HOUR', 'MONTH', 'DAY'], right_on=['TRUCK_ID', 'HOUR', 'MONTH', 'DAY']).drop_duplicates()
    
                        # Fill missing values
                        main_df = main_df.fillna({
                            'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE': (main_df['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].mean()),
                            'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE': (main_df['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].mean())
                        })
    
                        # Write the main_df DataFrame to the "Trend_Input_Data" table in Snowflake
                        session.write_pandas(
                            df=main_df,
                            table_name="Trend_Input_Data",
                            database="FROSTBYTE_TASTY_BYTES",
                            schema="ANALYTICS",
                            quote_identifiers=True,
                            overwrite=False
                        )
    
                        # Terminate the loop if it is the last month and year
                        if year == '2022' and month == 10:
                            break
    
        except snowflake.connector.errors.ProgrammingError as e:
            print(f"Error connecting to Snowflake: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def winsorise(df, variable, upper_limit, lower_limit):
        """
        Winsorizes a numerical variable in a DataFrame by capping extreme values to specified upper and lower limits.
    
        Returns:
            pd.Series: A pandas Series containing the winsorized values of the variable.
    
        Raises:
            ValueError: If the variable is not present in the DataFrame or is not numerical.
            ValueError: If the upper_limit is less than or equal to the lower_limit.
        """
    
        # Check if the variable is present in the DataFrame and is numerical
        if variable not in df.columns or not pd.api.types.is_numeric_dtype(df[variable]):
            raise ValueError(f"The variable '{variable}' is not present in the DataFrame or is not numerical.")
    
        # Check if the upper limit is greater than the lower limit
        if upper_limit <= lower_limit:
            raise ValueError("The upper limit must be greater than the lower limit.")
    
        # Winsorize the variable using numpy where function
        return np.where(df[variable] > upper_limit, upper_limit, np.where(df[variable] < lower_limit, lower_limit, df[variable]))
    
    def generate_month_list(start_month, start_year, end_month, end_year):
        """
        Generate a list of months between the given start and end dates.
    
        Parameters:
            start_month (int): The starting month (1 to 12).
            start_year (int): The starting year.
            end_month (int): The ending month (1 to 12).
            end_year (int): The ending year.
    
        Returns:
            list: A list of month numbers (1 to 12) representing the months between the start and end dates.
        """
        try:
            start_date = datetime.date(start_year, start_month, 1)
            end_date = datetime.date(end_year, end_month, 1)
            month_list = []
    
            while start_date <= end_date:
                month_list.append(start_date.month)
                # Move to the next month by adding 32 days and then setting the day to 1
                start_date += datetime.timedelta(days=32)
                start_date = start_date.replace(day=1)
    
            return month_list
    
        except Exception as e:
            print(f"An error occurred while generating the month list: {e}")
            return []
    
    def get_shift_durations(start_hour, end_hour, num_of_locs):
        """
        Calculate the shift durations based on the starting and ending hours and the number of locations.
    
        Parameters:
            start_hour (int): The starting hour (0 to 23).
            end_hour (int): The ending hour (0 to 23).
            num_of_locs (int): The number of locations.
    
        Returns:
            list: A list of shift durations for each location.
        """
        try:
            starting_hour = start_hour
            ending_hour = end_hour
            working_hours = ending_hour - starting_hour
            # Calculate the base shift hours (without considering the remainder)
            shift_hours = working_hours // num_of_locs
            # Calculate the remaining hours to distribute
            remaining_hours = working_hours % num_of_locs
    
            # Create a list to store the shift hours for each shift
            shift_hours_list = [shift_hours] * num_of_locs
    
            # Distribute the remaining hours evenly across shifts
            for i in range(remaining_hours):
                shift_hours_list[i] += 1
    
            return shift_hours_list
    
        except Exception as e:
            print(f"An error occurred while calculating shift durations: {e}")
            return []
    
    def get_shift_hours(start_hour, end_hour, num_of_locs):
        """
        Calculate the shift hours for each shift given the starting hour, ending hour, and number of locations.
    
        Returns:
            list: A list of lists representing the shift hours for each location.
        """
        try:
            starting_hour = start_hour
            ending_hour = end_hour
            working_hours = ending_hour - starting_hour
    
            # Calculate the base shift hours (without considering the remainder)
            shift_hours = working_hours // num_of_locs
    
            # Calculate the remaining hours to distribute
            remaining_hours = working_hours % num_of_locs
    
            # Create a list to store the shift hour arrays
            shift_hours_list = []
    
            # Calculate the shift hours for each shift
            current_hour = starting_hour
            for i in range(num_of_locs):
                # Calculate the end hour for the current shift
                end_shift_hour = current_hour + shift_hours
    
                # Add the hours for the current shift to the list
                shift_hours_list.append(list(range(current_hour, end_shift_hour)))
    
                # Adjust the current hour for the next shift
                current_hour = end_shift_hour
    
                # Distribute remaining hours evenly across shifts
                if remaining_hours > 0:
                    shift_hours_list[i].append(current_hour)
                    current_hour += 1
                    remaining_hours -= 1
    
            return shift_hours_list
    
        except Exception as e:
            print(f"An error occurred while calculating shift hours: {e}")
            return []
    
    def haversine_distance(df, max_distance):
        """
        Calculate the haversine distance between two sets of latitude and longitude coordinates.
    
        Returns:
            pandas.DataFrame: A DataFrame containing the rows with distances within the maximum distance.
        """
        try:
            # Copy the input DataFrame to avoid modifying the original
            df = df.copy()
    
            # Convert latitude and longitude from degrees to radians
            df['LAT_rad'] = df['LAT'].apply(math.radians)
            df['LONG_rad'] = df['LONG'].apply(math.radians)
            df['LAT2_rad'] = df['LAT2'].apply(math.radians)
            df['LONG2_rad'] = df['LONG2'].apply(math.radians)
    
            # Haversine formula
            df['dlon'] = df['LONG2_rad'] - df['LONG_rad']
            df['dlat'] = df['LAT2_rad'] - df['LAT_rad']
            df['a'] = (df['dlat'] / 2).apply(math.sin)**2 + df['LAT_rad'].apply(math.cos) * df['LAT2_rad'].apply(math.cos) * (df['dlon'] / 2).apply(math.sin)**2
            df['c'] = 2 * df['a'].apply(lambda x: math.atan2(math.sqrt(x), math.sqrt(1 - x)))
            df['DISTANCE'] = 6371 * df['c']  # Radius of the Earth in kilometers
    
            # Filter rows based on max_distance
            df = df[df['DISTANCE'] <= max_distance]
    
            # Drop intermediate columns
            df.drop(['LAT_rad', 'LONG_rad', 'LAT2_rad', 'LONG2_rad', 'dlon', 'dlat', 'a', 'c'], axis=1, inplace=True)
    
            # Reset the index of the resulting DataFrame
            df.reset_index(drop=True, inplace=True)
    
            return df
    
        except Exception as e:
            print(f"An error occurred while calculating haversine distance: {e}")
            return pd.DataFrame()
    
    def find_distance(df1, df2):
        """
        Calculate the haversine distance between two sets of latitude and longitude coordinates.
    
        Returns:
            float: The haversine distance between the two locations in kilometers.
        """
        try:
            # Radius of the Earth in kilometers
            R = 6371
    
            lat1 = df1['LAT']
            lon1 = df1['LONG']
            lat2 = df2['LAT']
            lon2 = df2['LONG']
    
            # Convert latitude and longitude to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)
    
            # Difference between latitudes and longitudes
            delta_lat = lat2_rad - lat1_rad
            delta_lon = lon2_rad - lon1_rad
    
            # Haversine formula
            a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c
    
            return distance
    
        except Exception as e:
            print(f"An error occurred while calculating the distance: {e}")
            return None
    
    def format_time_range(hours_list):
        """
        Format a list of hours into a time range string.
    
        Returns:
            str: A formatted time range string.
        """
        try:
            if len(hours_list) == 0:
                return "No hours provided"
            elif len(hours_list) == 1:
                return format_hour(hours_list[0])
            else:
                start_hour = format_hour(hours_list[0])
                end_hour = format_hour(hours_list[-1])
                return f"{start_hour} to {end_hour}"
    
        except Exception as e:
            print(f"An error occurred while formatting the time range: {e}")
            return ""
    
    def format_hour(hour):
        """
        Format an hour (0 to 23) into a string representation.
    
        Returns:
            str: A formatted string representation of the hour.
        """
        try:
            if hour == 0:
                return "12am"
            elif hour < 12:
                return f"{hour}am"
            elif hour == 12:
                return "12pm"
            else:
                return f"{hour - 12}pm"
    
        except Exception as e:
            print(f"An error occurred while formatting the hour: {e}")
            return ""
    
    def number_of_months(start_month, start_year, end_month, end_year):
        """
        Calculate the number of months between two dates.
    
        Returns:
            int: The number of months between the two dates.
        """
        try:
            start_date = datetime.date(start_year, start_month, 1)
            end_date = datetime.date(end_year, end_month, 1)
    
            months_diff = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
    
            return months_diff
    
        except Exception as e:
            print(f"An error occurred while calculating the number of months: {e}")
            return 0
    
    def get_highest_predicted(df):
        """
        Get the highest predicted value for each day based on the given DataFrame.
    
        Returns:
            pandas.DataFrame: A DataFrame with the highest predicted value for each day, along with the corresponding "LOCATION_ID".
        """
        try:
            # Group by "LOCATION_ID" and "DAY" and calculate the sum of "Predicted"
            summed_df = df.groupby(["LOCATION_ID", "DAY"])["Predicted"].sum().reset_index()
    
            # Find the maximum summed predicted value for each day
            max_predicted_df = summed_df.groupby("DAY")["Predicted"].max().reset_index()
    
            # Merge with the original DataFrame to get the corresponding "LOCATION_ID"
            result_df = pd.merge(max_predicted_df, summed_df, on=["DAY", "Predicted"])
    
            return result_df
    
        except Exception as e:
            print(f"An error occurred while getting the highest predicted values: {e}")
            return pd.DataFrame()
    
    import matplotlib.ticker as ticker
    def create_monthly_sales_graph(monthly_df,total_revenue):
        """
        Create a monthly sales graph from the given DataFrame. Shows monthly and total revenue, saves graph as png,
        """
    
        try:
            # Convert value to thousands (K)
            df = monthly_df.copy()
            df['Value'] = monthly_df['Value'] / 1000
    
            # Increase the figure size before plotting
            plt.figure(figsize=(10, 6))
    
            # Plot time series line chart for monthly_df (green)
            plt.plot(df['Months'], df['Value'], color='green', label='Current Predictions')
    
            plt.xlabel('Month')
            plt.ylabel('Total Revenue (K)')
            plt.title('Monthly Sales')
            plt.xticks(rotation=45)
            plt.legend()
    
            # Format y-axis tick labels with '$' sign
            plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}K'))
    
            # Add total revenue annotation
            plt.annotate(f'Total Revenue: ${total_revenue}', xy=(0.01, 0.3), xycoords='axes fraction', fontsize=12)
    
            # Set y-axis limits
            plt.ylim(0)
    
            # Save the plot to a file (to be displayed later in Streamlit)
            plt.savefig("monthly_sales_graph.png")
            plt.close()
        except Exception as e:
            print(f"An error occurred while creating the monthly_sales graph: {e}")
    
    def create_revenue_per_hour_graph(monthly_df,num_of_months ,work_hours,work_days,original_df,previous_df):
        """
        Create a monthly sales graph from the given DataFrame. Shows monthly and average revenue per hour for predicted, original, and previous year's data, saves graph as png,
        """
    
        try:
            # Calculate the YoY Growth of predicted revenue per hour
            predicted_revenue_per_hour = monthly_df['Value'] / (num_of_months * work_hours * (2+len(work_days) * 4))
            previous_year_revenue_per_hour = previous_df['Value'] / (num_of_months * previous_df['Hours'] * previous_df['Days'])
            yoy_growth = ((predicted_revenue_per_hour - previous_year_revenue_per_hour) / previous_year_revenue_per_hour) * 100
    
            # Calculate the revenue per hour increase between predicted and original
            original_revenue_per_hour = original_df['Value'] / (num_of_months * original_df['Hours'] * original_df['Days'])
            predicted_increase = ((predicted_revenue_per_hour - original_revenue_per_hour) / original_revenue_per_hour) * 100
    
            # Calculate monthly average revenue per hour
            monthly_df['Value'] = monthly_df['Value'] / work_hours / (2+len(work_days)*4)
            original_df['Value'] = original_df['Value'] / original_df['Hours'] / original_df['Days']
            previous_df['Value'] = previous_df['Value'] / previous_df['Hours'] / previous_df['Days']
    
            # Calculate the average YoY Growth and predicted increase compared to original
            avg_yoy_growth = yoy_growth.mean()
            avg_predicted_increase = predicted_increase.mean()
    
            # Increase the figure size before plotting
            plt.figure(figsize=(10, 8))
    
            # Plot time series line chart for monthly_df (green)
            plt.plot(monthly_df['Months'], monthly_df['Value'], color='green', label='Current Predictions')
    
            # Plot time series line chart for original_df (red)
            plt.plot(original_df['Months'], original_df['Value'], color='red', label='Original Data')
    
            # Plot time series line chart for previous_df (purple)
            plt.plot(previous_df['Months'], previous_df['Value'], color='purple', label='Previous Year')
    
            plt.xlabel('Month')
            plt.ylabel('Total Revenue /hour')
            plt.title('Monthly Revenue Per Hour')
            plt.xticks(rotation=45)
            plt.legend()
    
            # Format y-axis tick labels with '$' sign
            plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    
            # Add total revenue annotation
            og = (original_df['Value'].sum() / num_of_months).round(2)
            prev = (previous_df['Value'].sum()/ num_of_months).round(2)
            avg = (monthly_df['Value'].sum()/ num_of_months).round(2)
            
            plt.annotate(f'Average Revenue Per Hour: ${avg}/hr', xy=(0.01, 0.6), xycoords='axes fraction', fontsize=12)
            plt.annotate(f'Original Data: ${og}/hr', xy=(0.01, 0.5), xycoords='axes fraction', fontsize=12)
            plt.annotate(f'Previous Year: ${prev}/hr', xy=(0.01, 0.4), xycoords='axes fraction', fontsize=12)
    
            # Display difference between predicted and original as annotation
            plt.annotate(f'Average Predicted Increase from Original: {avg_predicted_increase:.2f}%', xy=(0.01, 0.3), xycoords='axes fraction', fontsize=12, color='blue')
    
            # Display average YoY Growth as annotation
            plt.annotate(f'Average YoY Growth: {avg_yoy_growth:.2f}%', xy=(0.01, 0.2), xycoords='axes fraction', fontsize=12, color='blue')
    
            # Set y-axis limits
            plt.ylim(0)
    
            # Save the plot to a file (to be displayed later in Streamlit)
            plt.savefig("monthly_revenue_per_hour_graph.png")
            plt.close()
        except Exception as e:
            print(f"An error occurred while creating the monthly_revenue_per_hour graph: {e}")
    
    def create_x_holdout_graph(df_predictions):
        """
        Create a scatter plot comparing predicted values against holdout values.
        """
    
        try:
            # Plot the predicted values against the holdout values
            plt.figure(figsize=(20, 10))
            plt.scatter(df_predictions['Holdout'], df_predictions['Predicted'], c='blue', label='Predicted vs Holdout')
    
            # Add a reference line
            plt.plot([df_predictions['Holdout'].min(), df_predictions['Holdout'].max()],
                     [df_predictions['Holdout'].min(), df_predictions['Holdout'].max()],
                     c='red', label='Perfect Prediction')
    
            # Set labels and title
            plt.xlabel('Holdout')
            plt.ylabel('Predicted')
            plt.title('Prediction Accuracy')
    
            # Show the legend
            plt.legend()
    
            # Save the plot to a file (to be displayed later in Streamlit)
            plt.savefig("x_holdout_graph.png")
            plt.close()
    
        except Exception as e:
            print(f"An error occurred while creating the x_holdout graph: {e}")
    
    import joblib
    try:
        xgb = joblib.load('model.joblib')
    except Exception as e:
            print(f"An error occurred while loading the model from the file: {e}")
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    import numpy as np
    # Streamlit web app code
    def main():
        try:
            st.title("Food Truck Revenue Forecast Trend")
    
            # Disable button if there's a warning message
            button_disabled = False
    
            # Overview of the web app tab
            st.write("This tab presents food truck revenue trend forecasts, including optimal route predictions and revenue estimates on a daily basis for the selected time-frame. It allows users to view monthly graphs illustrating revenue and revenue/hr per month and also offers a comparison with the previous year's and original year's revenue data without the optimized routing algorithm. Additionally, users can observe the machine learning model's performance metrics and explore the feature importance.")
    
            # Input fields for truck data
            st.title('Input Section to determine optimal routing and forecast time-frame')
    
            # Dictionary to map truck details to their IDs
            truck_details_to_id = {
                "Cheeky Greek, Gyros, Denver, David Miller (27)": 27,
                "Peking Truck, Chinese, Denver, David Miller (28)": 28,
                "Peking Truck, Chinese, Seattle, Brittany Williams (43)": 43,
                "Nani's Kitchen, Indian, Seattle, Mary Sanders (44)": 44,
                "Freezing Point, Ice Cream, Boston, Brittany Williams (46)": 46,
                "Smoky BBQ, BBQ, Boston, Mary Sanders (47)": 47
            }
    
            # Predefined list for truck details
            truck_details_list = list(truck_details_to_id.keys())
    
            # Selectbox widget to choose a truck detail
            selected_truck_detail = st.selectbox("Select Food Truck (ID)", truck_details_list)
    
            # Get the corresponding truck ID using the dictionary
            truck_id = truck_details_to_id[selected_truck_detail]
    
            # Define the minimum and maximum allowed date range
            min_date = datetime.date(2020, 1, 1)
            max_date = datetime.date(2022, 10, 31)
    
            # Date range input widget to choose the forecast period
            date_range = st.date_input('Select a date range forecast (only month and year) For Jan 2020 to Oct 2022 only', (min_date, max_date))
    
            # Validate the selected date range
            if len(date_range) == 1:
                date_range = (date_range[0], date_range[0])  # Fix for handling a single date selection
    
            # Enforce the minimum 3 months date range
            if date_range[1] - date_range[0] < datetime.timedelta(days=3 * 30):
                st.warning("Please select a date range with a minimum of 3 months.")
                button_disabled = True
    
            # Ensure the range has at most 12 months
            if date_range[1] - date_range[0] > datetime.timedelta(days=12 * 30):
                st.warning("Please select a date range with a maximum of 12 months.")
                button_disabled = True
                # Adjust the range to have at most 12 months
                end_date = min(date_range[0] + datetime.timedelta(days=12 * 30), max_date)
                date_range = (date_range[0], end_date)
    
            # Limit the user from selecting dates beyond the minimum and maximum dates
            if date_range[0] < min_date:
                st.warning("Please select a start date within the allowed range.")
                button_disabled = True
                date_range = (min_date, date_range[1])
            elif date_range[1] > max_date:
                st.warning("Please select an end date within the allowed range.")
                button_disabled = True
                date_range = (date_range[0], max_date)
    
            # Extract the start and end year and month from the selected date range
            start_year = date_range[0].year
            start_month = date_range[0].month
            end_year = date_range[1].year
            end_month = date_range[1].month
    
            # Slider widget to select working hours range
            working_hours = st.slider('Select working hours (24h-Notation)', 1, 24, (8, 12))
    
            # Ensure the range has at least 2 hours
            if working_hours[1] - working_hours[0] < 2:
                st.warning("Please select a range with at least 2 hours.")
                button_disabled = True
                # Adjust the range to have at least 2 hours
                end_hour = min(working_hours[0] + 2, 23)
                working_hours = (working_hours[0], end_hour)
    
            # Extract the start and end hour from the selected working hours range
            start_hour = int(working_hours[0])
            end_hour = int(working_hours[1])
    
            # Handle special case when end hour is 24 (midnight)
            if end_hour == 24:
                end_hour = 0
    
            # Calculate total working hours
            work_hours = end_hour - start_hour
    
            # Number input widget to select the number of locations
            num_of_locs = st.number_input("Number of Locations", min_value=1, value=2, max_value=work_hours)
    
            # Validate the number of locations
            if num_of_locs > 8:
                st.warning("Please select a smaller number of locations (maximum 8)")
                button_disabled = True
                num_of_locs = 8
    
            # Number input widget to select the maximum travel distance for each location
            each_location_travel_distance = st.number_input("Each Location Max Travel Distance (km)", min_value=0, value=5, max_value=50)
    
            # Dictionary to map weekday names to integers
            weekdays_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
            # List of weekday names
            weekdays_names = list(weekdays_dict.keys())
    
            # Multi-select widget to select work days
            selected_weekdays_names = st.multiselect("Select Work Days", weekdays_names, default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    
            # Convert selected weekday names to corresponding integer values
            work_days = [weekdays_dict[name] for name in selected_weekdays_names]
    
            # If no weekdays are selected, default to all weekdays (Monday=0, ..., Friday=4)
            if not work_days:
                st.warning("Please select at least one weekday")
                work_days = [0, 1, 2, 3, 4]
                button_disabled = True
    
            # Date input widget to select a specific date for the optimal route
            route_date = st.date_input('Select a specific date to see its optimal route', date_range[0])
    
            # Check if the selected date is within the allowed range
            if route_date < date_range[0]:
                # Display a warning message and adjust the date to the start of the date range
                st.warning("Please select a start date within the allowed range.")
                route_date = date_range[0]
                button_disabled = True
            elif route_date > date_range[1]:
                # Display a warning message and adjust the date to the end of the date range
                st.warning("Please select an end date within the allowed range.")
                route_date = date_range[1]
                button_disabled = True
            elif route_date.weekday() not in work_days:
                # Display a warning message if the selected date is not a working weekday
                st.warning("Please select a date that is during one of your selected working weekdays.")
                route_date = date_range[0]
                button_disabled = True
                
            # Display the selected truck id and date range
            st.subheader("Selected Truck ID: {}".format(truck_id))
            st.subheader("Selected date range: {} to {}".format(date_range[0].strftime("%B %Y"), date_range[1].strftime("%B %Y")))
            
        except Exception as e:
            print(f"An error occurred with the input section: {e}")
            
        try:
            # Process the inputs and display the results when the "Process Data" button is clicked
            if st.button("Forecast Data (Main)", disabled=button_disabled):
                # Calculate the maximum total travel distance based on each location's max travel distance and the number of locations
                max_total_travel_distance = each_location_travel_distance * num_of_locs
    
                # Generate the list of months within the selected date range
                months_list = generate_month_list(start_month, start_year, end_month, end_year)
    
                # Convert the selected working hours to a list of hours
                hours_list = list(range(start_hour, end_hour + 1))
    
                # Initialize variables
                year = start_year
                shift_hours_list = get_shift_hours(start_hour, end_hour, num_of_locs)
                month_value_list = []
                final_df = pd.DataFrame()
    
                # DataFrames for storing original and previous year's revenue information
                original_df = pd.DataFrame()
                previous_df = pd.DataFrame()
    
                # Retrieve sales data for the selected truck from the Snowflake database
                session.use_schema("ANALYTICS")
                query = 'Select * from "Sales_Forecast_Training_Data" WHERE TRUCK_ID = {}'.format(truck_id)
                X_final_scaled = session.sql(query).to_pandas()
                X_final_scaled.rename(columns={"Profit": "Revenue"}, inplace=True)
                sales_pred = session.sql("select * from ANALYTICS.SALES_PREDICTION").to_pandas()
                X_final_scaled = X_final_scaled.merge(sales_pred["l_w5i8_DATE"].astype(str).str[:4].rename('YEAR'), left_index=True, right_index=True)
                X_final_scaled = X_final_scaled[['Revenue', 'YEAR', 'MONTH', 'DAY', 'HOUR']]
    
                for month in months_list:
                    # Adjust the year if the date range spans multiple years
                    if start_year != end_year:
                        if month == 1:
                            year += 1
    
                    # Fetch input data for the current month, hours, and working weekdays from the Snowflake database
                    query = 'SELECT * FROM "Trend_Input_Data" WHERE TRUCK_ID = {} AND YEAR = {} AND MONTH = {} AND HOUR IN ({}) AND DOW IN ({});'.format(
                        truck_id, year, month, ', '.join(map(str, hours_list)), ', '.join(map(str, work_days)))
                    input_data = session.sql(query).to_pandas()
    
                    # Make predictions using the loaded machine learning model for the current input data
                    predict_df = input_data[['TRUCK_ID', 'MONTH', 'HOUR', 'DOW', 'DAY', 'PUBLIC_HOLIDAY', 'LAT', 'LONG', 'LOCATION_ID', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', 'WEATHERCODE', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']]
                    predict_df['Predicted'] = xgb.predict(predict_df)
    
                    # Initialize a list to store DataFrames for each shift's predicted values
                    shifts_df_list = []
                    value = 0
                    for i in range(num_of_locs):
                        # Filter data for the current shift
                        current_shift = predict_df[predict_df['HOUR'].isin(shift_hours_list[i])]
    
                        if i > 0:
                            # Merge data from the previous shift to calculate the distance between locations
                            previous_shift = pd.merge(shifts_df_list[i-1], predict_df[["LOCATION_ID", 'LAT', 'LONG']].drop_duplicates(), on=["LOCATION_ID"])
                            previous_shift.rename(columns={'LAT': 'LAT2', 'LONG': 'LONG2'}, inplace=True)
                            current_shift = pd.merge(current_shift, previous_shift[['LAT2','LONG2','DAY']], on=['DAY']).drop_duplicates()
                            current_shift = haversine_distance(current_shift, each_location_travel_distance)
    
                        # Get the highest predicted revenue for each day in the current shift
                        highest_df = get_highest_predicted(current_shift)
                        value += highest_df['Predicted'].sum()
                        shifts_df_list.append(highest_df)
                        highest_df['HOUR'] = shift_hours_list[i][0]
                        highest_df['MONTH'] = month
                        highest_df['YEAR'] = year
                        final_df = pd.concat([final_df, highest_df])
    
                    # Store the total predicted revenue value for the current month
                    month_value_list.append(value)
    
                    # Calculate original and previous year's revenue for the current month
                    og = X_final_scaled[(X_final_scaled['MONTH'] == month) & (X_final_scaled['YEAR'] == str(year))]
                    og_df = pd.DataFrame({'Value': [og['Revenue'].sum()], 'Hours': [og['HOUR'].nunique()], 'Days': [og['DAY'].nunique()], 'YEAR': [year], 'Months': [month]})
                    original_df = pd.concat([original_df, og_df])
                    prev = X_final_scaled[(X_final_scaled['MONTH'] == month) & (X_final_scaled['YEAR'] == str(year-1))]
                    pre_df = pd.DataFrame({'Value': [prev['Revenue'].sum()], 'Hours': [prev['HOUR'].nunique()], 'Days': [prev['DAY'].nunique()], 'YEAR': [year-1], 'Months': [month]})
                    previous_df = pd.concat([previous_df, pre_df])
    
                # Create DataFrame to store the monthly revenue values
                monthly_df = pd.DataFrame({
                    'Months': months_list,
                    'Value': month_value_list
                })
    
                # Convert months to three-letter abbreviations
                monthly_df['Months'] = monthly_df['Months'].apply(lambda x: calendar.month_abbr[x])
                previous_df['Months'] = previous_df['Months'].apply(lambda x: calendar.month_abbr[x])
                original_df['Months'] = original_df['Months'].apply(lambda x: calendar.month_abbr[x])
    
                # Calculate the total revenue for the selected time frame
                total_revenue = monthly_df['Value'].sum()
    
                # Calculate the number of months within the selected date range
                num_of_months = number_of_months(start_month, start_year, end_month, end_year)
    
                # Format total revenue with commas as thousands separator
                total_revenue = f'{total_revenue:,.0f}'
    
                # Generate and save the monthly sales and revenue per hour graphs to image files
                create_monthly_sales_graph(monthly_df, total_revenue)
                create_revenue_per_hour_graph(monthly_df, num_of_months, work_hours, work_days, original_df, previous_df)
    
                # Display the monthly sales graph
                st.image("monthly_sales_graph.png", use_column_width=True)
    
                # Display the monthly revenue per hour graph
                st.image("monthly_revenue_per_hour_graph.png", use_column_width=True)
    
                # Retrieve additional data for the current route date from the Snowflake database
                X_final_scaled = session.sql('Select * from "Sales_Forecast_Training_Data";').to_pandas()
                final_df = pd.merge(final_df.copy(), X_final_scaled[['LOCATION_ID', 'LAT', 'LONG']].drop_duplicates(), on=["LOCATION_ID"])
                shift_durations = get_shift_durations(start_hour, end_hour, num_of_locs)
                distance_travelled = 0
                revenue_earned = 0
    
                # Loop through each shift to display details for each location and calculate revenue
                for i in range(num_of_locs):
                    current_df = final_df[(final_df['DAY'] == route_date.day) & (final_df['MONTH'] == route_date.month) & (final_df['YEAR'] == route_date.year)].iloc[i]
                    if i == num_of_locs - 1:
                        shift_hours_list[i].append(shift_hours_list[i][-1] + 1)
                    time_range = format_time_range(shift_hours_list[i])
                    st.subheader("Shift: {}".format(str(i+1)))
                    st.write(time_range)
                    st.write('Shift Hours: ', shift_durations[i])
                    st.write('Current Location Number: ', current_df['LOCATION_ID'].round(0))
                    st.write('Predicted Revenue: ', current_df['Predicted'].round(2))
                    st.write('Predicted Revenue per hour: ', (current_df['Predicted'] / shift_durations[i]).round(2))
                    revenue_earned += current_df['Predicted']
    
                # Calculate the distance travelled and the revenue earned per kilometer
                for i in range(num_of_locs - 1):
                    distance_travelled += find_distance(final_df[(final_df['DAY'] == route_date.day) & (final_df['MONTH'] == route_date.month) & (final_df['YEAR'] == route_date.year)].iloc[i], final_df[(final_df['DAY'] == route_date.day) & (final_df['MONTH'] == route_date.month) & (final_df['YEAR'] == route_date.year)].iloc[i + 1])
    
                if distance_travelled > 0:
                    rev_dis = round(revenue_earned / distance_travelled, 2)
                else:
                    rev_dis = round(revenue_earned, 2)
    
                # Display the overall route information
                st.subheader('Overall Route')
                st.write('Maximum possible distance travelled throughout all the shifts: ', max_total_travel_distance, 'km')
                st.write('Total distance travelled: ', round(distance_travelled, 2), 'km')
                st.write('Dollars earned by km travelled: $', rev_dis, '/km')
        except Exception as e:
            print(f"An error occurred while processing the data: {e}")
    
        try:
            # Create a button to show feature importance and performance
            if st.button('Show Model Performance'):
                # Load data from the Snowflake database
                session.use_schema("ANALYTICS")
                X_final_scaled = session.sql('Select * from "Sales_Forecast_Training_Data";').to_pandas()
                X_final_scaled.rename(columns={"Profit": "Revenue"}, inplace=True)
    
                # Winsorize the target and some features to reduce the impact of outliers
                X_final_scaled['Revenue'] = winsorise(X_final_scaled, 'Revenue', X_final_scaled['Revenue'].quantile(0.85), X_final_scaled['Revenue'].quantile(0))
                X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'] = winsorise(X_final_scaled, 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].quantile(0.85), X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].quantile(0))
                X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'] = winsorise(X_final_scaled, 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].quantile(0.8), X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].quantile(0.5))
    
                # Split the dataset into features (X) and target (y)
                X = X_final_scaled.drop("Revenue", axis=1)
                y = X_final_scaled["Revenue"]
    
                # Split the dataset into training and testing datasets
                X_training, X_holdout, y_training, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.2, random_state=42)
    
                # Create a DataFrame with holdout values and predicted values
                df_predictions = X_holdout.copy()
                df_predictions['Holdout'] = y_holdout
                holdout_predictions = xgb.predict(X_holdout)
                df_predictions['Predicted'] = holdout_predictions
    
                # Add a column for the differences
                df_predictions['Difference'] = df_predictions['Predicted'] - df_predictions['Holdout']
    
                # Get feature importance as a DataFrame
                feature_importance = pd.DataFrame({'Feature': X_final_scaled.drop(columns='Revenue').columns, 'Importance': xgb.feature_importances_})
    
                # Display the feature importance DataFrame
                st.subheader('Feature Importance')
                st.dataframe(feature_importance)
    
                # Calculate performance metrics
                y_true = df_predictions['Holdout']
                y_pred = df_predictions['Predicted']
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                r2 = r2_score(y_true, y_pred)
    
                # Display the performance metrics
                st.subheader('Model Performance on Holdout data')
                st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
                st.write(f'Mean Squared Error (MSE): {mse:.2f}')
                st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
                st.write(f'R-squared (R2) score: {r2:.2f}')
    
                # Generate and save the holdout vs. predicted graph to an image file
                create_x_holdout_graph(df_predictions)
    
                # Display the holdout vs. predicted graph using the saved image file
                st.image("x_holdout_graph.png", use_column_width=True)
    
                # Display the true and predicted values in a DataFrame
                result_df = pd.DataFrame({'True Values': y_true, 'Predicted Values': y_pred})
                st.subheader('True vs. Predicted Values')
                st.dataframe(result_df)
        except Exception as e:
            print(f"An error occurred while showing the model performance: {e}")

    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            print(f"An error occurred with the streamlit web app: {e}")
        

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
        
        datetime_object = datetime.datetime.strptime(date, '%Y-%m-%d')
        
        
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
            #session.use_schema("RAW_POS")
            #query = "SELECT * FROM TRUCK WHERE TRUCK_ID = '{}'".format(truck_id)
            #query = "SELECT * FROM TRUCK"
            #truck_df=session.sql(query).toPandas()
            #truck_df.head(30)
            #truck_df.to_csv('truck_df.csv',index=False)
            truck_df=pd.read_csv('truck_df.csv')
            truck_df=truck_df[truck_df['TRUCK_ID']==truck_id]
            truck_df=truck_df.reset_index()
            truck_df=truck_df.drop(['index'],axis=1)
    
    
            city = truck_df['PRIMARY_CITY'].iloc[0]
        
            #query = "SELECT * FROM LOCATION WHERE CITY = '{}'".format(city)
            #session.use_schema('RAW_POS')
            #query = "SELECT * FROM LOCATION"
            #location_df=session.sql(query).toPandas()
            #location_df.head()
            #location_df.to_csv('location_df.csv',index=False)
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
            
            #sales_pred=session.sql("select * from ANALYTICS.SALES_PREDICTION").to_pandas() #this is the problem.
        
            #sales_pred.to_csv('sales_pred.csv')
            sales_pred=pd.read_csv('sales_pred.csv')
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
            merged_df=merged_df.dropna(subset=['LOCATION_ID'])
            merged_df['LOCATION_ID'] = merged_df['LOCATION_ID'].astype(int)
            initial_df_position = merged_df[['TRUCK_ID', 'MONTH', 'HOUR', 'DOW', 'DAY', 'PUBLIC_HOLIDAY', 'LAT', 'LONG', 'LOCATION_ID', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', 'WEATHERCODE', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']]
        
            predictions = model.predict(initial_df_position)
            initial_df_position['Predicted'] = predictions
           
        
            
            data_for_avg_revenue=[truck_id,x,initial_df_position['Predicted'].mean()]
            average_revenue_for_hour.loc[len(average_revenue_for_hour)]=data_for_avg_revenue
            
       
        
        
        #Initialize variables
        average_revenue_for_hour['rolling_average']=0
        working_hours=hours-1
        max_revenue = 0
        optimal_hours = []
        for i in range(len(average_revenue_for_hour) - working_hours):
            
            total_revenue=0
            # Calculate the total revenue for the current 5-hour window
            total_revenue = average_revenue_for_hour.loc[i:i+working_hours, 'AVERAGE REVENUE PER HOUR'].sum()
            average_revenue_for_hour['rolling_average'].loc[i] = total_revenue
            
             # Check if the current total revenue is greater than the previous maximum
            if total_revenue > max_revenue:
                max_revenue = total_revenue
                optimal_hours = average_revenue_for_hour.loc[i:i+working_hours, 'HOUR'].tolist()
        values=[optimal_hours,max_revenue]



    
        return values

        
    if st.button("Run Algorithm"):
        # Display a loading message while the algorithm is running
        with st.spinner("Running the algorithm..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(find_optimal_hour, truck_id, date, no_of_hours)
            while not future.done():
                pass
            output = future.result()
            
                                      
    
        # Show the output once the algorithm is done
        st.success("Algorithm completed!")
        st.write("Output:")
        st.text("Optimal Hours: "+str(output[0]))
        st.text("Maximum Revenue: "+str(output[1]))




    



with tab4: #Aryton
    print('Aryton')


with tab5: #vibu
    
   

# read csv from a URL
    @st.cache_data
    def get_data() -> pd.DataFrame:
        df=pd.read_csv('truck_manager_merged_df.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df["Date"]=pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df.sort_values(by='Date', inplace=True)  
        df['predicted_earning'] = df['predicted_earning'].apply(lambda x: [int(float(i)) for i in x.strip('[]').split(',')])
        df['total_sales']=df['predicted_earning'].apply(lambda x: sum(x))
        percentile_25 = df['total_sales'].quantile(0.25)
        percentile_50 = df['total_sales'].quantile(0.5)
        
        

        # Define the function to map earnings to letter grades
        def assign_letter_grade(earnings):
      
            if earnings > percentile_50:
                return 'A'
            elif earnings > percentile_25:
                return 'B'
            else:
                return 'C'

        # Apply the mapping function to create a new column with letter grades
        df['Letter_Grade'] = df['total_sales'].apply(assign_letter_grade)
        df['start_time'] = df['start_time'].apply(lambda x: [int(float(i)) for i in x.strip('[]').split(',')])
                
        
        return  df

    df = get_data()
    
    @st.cache_data
    def get_sales_time() -> pd.DataFrame:
        sales_over_time_df = df.copy()
        sales_over_time_df['start_time'] = sales_over_time_df['start_time'].apply(lambda x: x[1:])
        sales_over_time_df = sales_over_time_df.explode('start_time')
        
        sales_over_time_df = sales_over_time_df.explode('predicted_earning')
        sales_over_time_df.drop_duplicates(subset=['predicted_earning', 'start_time'], inplace=True)
        
       

        sales_over_time_df.sort_values(by='start_time', inplace=True)
        
        return sales_over_time_df
    
    sales_time =get_sales_time()
        
    
    
   
    # dashboard title
    st.title("Food Truck Competitor Analysis Dashboard")

    # top-level filters
    truck_filter = st.selectbox("Select the Truck", pd.unique(df["Truck_ID"]))
    percentile_50 = df['total_sales'].quantile(0.5)
    percentile_dis_50 = df['total_distance_traveled'].quantile(0.5)

    # creating a single-element container
    placeholder = st.empty()

 

  


    
     # dataframe filter
    filter_df = df[df["Truck_ID"] ==truck_filter]
    
    filter_sales_time = sales_time[sales_time["Truck_ID"] ==truck_filter]


    with placeholder.container():

            # create three columns
            kpi1, kpi2, kpi3,kpi4 = st.columns(4)
                        
            kpi1.metric( 
                label="Predicted Performance",
                 value=filter_df["Letter_Grade"].values[0],
                 delta=None)

            # fill in those three columns with respective metrics or KPIs
            kpi2.metric(
                label="Sales",
                value=filter_df["total_sales"].sum(),
                delta=filter_df["total_sales"].sum()-percentile_50
            )
            
        
            
            kpi3.metric(
                label="Distance Travelled",
                value=filter_df["total_distance_traveled"].sum().round(2),
                delta=(filter_df["total_distance_traveled"].sum()-percentile_dis_50).round(2)
            )
            
            kpi4.metric(
                label="Working Hour",
                value=filter_df["working_hour"].sum(),
                delta=filter_df["working_hour"].sum()-df["working_hour"].median()
            )

                
             

            st.markdown("### Top 5 Trucks")
                
            df_sorted = df.sort_values(by='total_sales', ascending=False)
            top_5_trucks = df_sorted.head(5)
            top_5_trucks['Truck_ID'] = top_5_trucks['Truck_ID'].astype(str)
            fig2 = px.bar(top_5_trucks, x='Truck_ID', y='total_sales', 
             labels={'Truck_ID': 'Truck ID', 'Total_Earnings': 'Total Predicted Earnings'})
            fig2.update_xaxes(type='category')

            st.write(fig2)

         

        
        
        
        
        
        
        
        
        
        
        
        #vibu
        
    #     df= pd.read_csv('truck_location_df.csv')
    #     df['Date'] = pd.to_datetime(df['Date'])
    #     df["Date"]=pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # # Sort the DataFrame by Date
    #     df.sort_values(by='Date', inplace=True)



    # # Dashboard title
    #     st.title("Food Truck Competitor Analysis Dashboard")

    # # Overview Section
    #     st.header("Overview")
    #     df['predicted_earning'] = df['predicted_earning'].apply(lambda x: [int(float(i)) for i in x.strip('[]').split(',')])
    #     total_sales=df['predicted_earning'].apply(lambda x: sum(x)).sum()

    #     average_predicted_earnings = df['predicted_earning'].apply(lambda x: sum(x) / len(x)).mean()
    #     total_locations_visited = df['Num_of_locs'].sum()

    #     st.write(f"Total Sales (Last 2 weeks): ${round(total_sales, 2)}")
    #     st.write(f"Average Predicted Earnings: ${round(average_predicted_earnings, 2)}")
    #     st.write(f"Total Locations Visited: {total_locations_visited}")

    # # # Sales Performance Section
    # #     st.header("Sales Performance")
    # #     fig_sales = px.bar(df, x='Date', y='predicted_earning', title='Predicted Earnings Over Time')
    # #     st.plotly_chart(fig_sales)

    # # Efficiency Metrics Section
    #     st.header("Efficiency Metrics")
    #     fig_hours = px.pie(df, names='Truck_ID', values='working_hour', title='Distribution of Working Hours')
    #     fig_shifts = px.bar(df, x='Truck_ID', y='Num_of_locs', title='Number of Locations Visited')
    #     st.plotly_chart(fig_hours)
    #     st.plotly_chart(fig_shifts)


    # # Prioritization Analysis Section
    #     st.header("Prioritization Analysis")
    #     df['Priority_Order'] = df['Truck_ID'].rank(method='first')
    #     fig_priority = px.bar(df, x='Truck_ID', y='predicted_earning', color='Priority_Order',
    #                       labels={'Truck_ID': 'Truck ID', 'predicted_earning': 'Predicted Earnings'},
    #                       title='Truck Prioritization Based on Sales Performance')
    #     st.plotly_chart(fig_priority)

    
        

 
    
