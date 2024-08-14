import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import geopandas as gpd
from shapely import wkt
from streamlit_folium import st_folium
from geopy.distance import distance
st.set_page_config(
    page_title='Fair Market Housing Data Exploration',
    layout="wide")

@st.cache_data
def load_housing_data():
    return pd.read_csv('cleaned_housing_data.csv')

@st.cache_data
def load_geojson_data():
    census_tracts = gpd.read_file('zcta520_simplified.shp', crs='EPSG:4326')
    census_tracts.rename(columns={'ZCTA5CE20': 'ZIP Code'}, inplace=True)
    census_tracts = census_tracts[['ZIP Code', 'geometry']]
    census_tracts['ZIP Code'] = census_tracts['ZIP Code'].astype(int)
    census_tracts['geometry'] = census_tracts['geometry'].apply(lambda x: x.simplify(0.001, preserve_topology=True))
    return census_tracts

def limit_data_by_radius(df, center_lat, center_lon, radius_miles):
    def is_within_radius(row):
        point = (row['Latitude'], row['Longitude'])
        center_point = (center_lat, center_lon)
        return distance(center_point, point).miles <= radius_miles
    
    return df[df.apply(is_within_radius, axis=1)]

def limit_data_by_area(df, lat, lon, radius):
    """Limit data to within a certain radius (in degrees) of a point."""
    df['distance'] = df['geometry'].apply(lambda x: x.centroid.distance(Point(lon, lat)))
    limited_df = df[df['distance'] < radius]
    return limited_df


def merge_data(housing_data, geojson_data):
    map_df = pd.merge(housing_data, geojson_data, on='ZIP Code')
    map_df = gpd.GeoDataFrame(map_df, geometry='geometry', crs='EPSG:4326')
    map_df['geometry'] = map_df['geometry'].apply(lambda x: x.buffer(0))
    return map_df

def create_folium_map(df, zip_data):
    m = folium.Map(location = [float(zip_data['Latitude'].iloc[0]), float(zip_data['Longitude'].iloc[0])], zoom_start = 12, min_zoom=10, max_zoom=15)

    tooltip = folium.GeoJsonTooltip(fields=['ZIP Code', 'City', 'Studio', '1BR', '2BR', '3BR', '4BR'],
                                    aliases=['ZIP Code:', 'City:', 'Studio:', '1BR:', '2BR:', '3BR:', '4BR:'],
                                    localize=True, sticky=False, labels=True, 
                                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"))
    g = folium.GeoJson(
        gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326'),
        name='geojson',
        style_function=lambda x: {'fillColor': 'orange'},
        tooltip=tooltip,
        highlight_function=lambda x: {'weight': 3, 'fillColor': 'blue'},
        ).add_to(m)
    return m

# Load data
df = load_housing_data()
census = load_geojson_data()
map_df = merge_data(df, census)

def home_page():
    # Title
    st.title('Fair Market Housing Data Exploration')

    # Description
    st.write('This is a simple web app that allows you to explore the Fair Market Housing data for the 2024 Fiscal Year.')
    st.write('The data is sourced from the U.S. Department of Housing and Urban Development (HUD) and uses UnitedStatesZipCodes.org for location specific data.')

    # Display the data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(df)



def map_page():
    # Add Folium Map
    st.subheader('Map of Fair Market Rent')
    chosen_location = st.selectbox('Select a Location', map_df['Location'].unique(), index=None)
    chosen_radius = st.slider('Select a Radius (miles)', 5, 250, 10)

    if chosen_location:
        location_data = map_df[map_df['Location'] == chosen_location].iloc[:1]
        chosen_lat = location_data['Latitude'].iloc[0]
        chosen_lon = location_data['Longitude'].iloc[0]
        radius_miles = chosen_radius
        filtered_df = limit_data_by_radius(map_df, chosen_lat, chosen_lon, radius_miles)
        st.subheader('Data for Selected Location')
        st.write(filtered_df[['ZIP Code', 'City', 'State', 'Studio', '1BR', '2BR', '3BR', '4BR']])
        map = create_folium_map(filtered_df, location_data)
        st_folium(map, width=1200, height=900)
        


def filter_data_page():
    st.title('Filter the Data')

    # Choose how many bedrooms to display
    bedrooms = st.multiselect("Select number of bedrooms to display", 
                            ['Studio','1BR','2BR','3BR','4BR',])

    # Radio for city, metro area, or zip code
    filter_by = st.radio('Filter by', ['City', 'Metro Area', 'ZIP Code'])
    if filter_by == 'City':
        filter_var = 'Location'
        filter_proper = 'Citie'
        filter_options = df['Location'].unique()
    elif filter_by == 'Metro Area':
        filter_var = 'Metro Area'
        filter_proper = "Metro Area"
        filter_options = df['Metro Area'].unique()
    else:
        filter_var = 'ZIP Code'
        filter_proper = 'ZIP Code'
        filter_options = df['ZIP Code'].unique()

    # Filter
    selected_areas = st.multiselect(f'Select {filter_proper}s', sorted(filter_options))

    filtered_df1 = df[df[filter_var].isin(selected_areas)]

    # Filter by bedrooms
    pre = ['ZIP Code','Metro Area','Location']
    post = ['County','Latitude','Longitude']
    cols = pre + sorted(bedrooms) + post
    filtered_df2 = filtered_df1[cols]

    st.subheader('Display Filtered Data')
    st.write(filtered_df2)

    # Group by Chosen Area
    if not filtered_df2.empty and len(selected_areas) > 0 and len(bedrooms) > 0:
        st.subheader(f'Display Fair Market Rent by {filter_var}')

        grouped = filtered_df2.groupby(filter_var)[sorted(bedrooms)].mean()
        st.write(round(grouped, 2))

def compare_data_page():
    st.title('Compare Data')
    st.write('Compare the average fair market rent for two different cities.')
    st.write('---')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('City 1')
        city1 = st.selectbox('Select a City', df['Location'].unique(), key='city1')
        city1_data = df.groupby('Location').get_group(city1)[['Studio', '1BR', '2BR', '3BR', '4BR']].agg(['mean', 'std'])
        st.write(round(city1_data.T),0)
    with col2:
        st.subheader('City 2')
        city2 = st.selectbox('Select a City', df['Location'].unique(),key='city2')
        city2_data = df.groupby('Location').get_group(city2)[['Studio', '1BR', '2BR', '3BR', '4BR']].agg(['mean', 'std']).apply(lambda x: round(x, 0))
        st.write(round(city2_data.T),0)
    with col3:
        st.subheader('Comparison')
        st.write('##')
        if city1_data is not None and city2_data is not None:
            comparison = city1_data - city2_data
            st.write(round(comparison.T),0)
    st.write('---')
# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home :house:", "Map :earth_americas:", "Filter Data :bar_chart:","Compare Cities :sleuth_or_spy:"])

# Display the selected page
if page == "Home :house:":
    home_page()
elif page == "Filter Data :bar_chart:":
    filter_data_page()
elif page == "Map :earth_americas:":
    map_page()
elif page == "Compare Cities :sleuth_or_spy:":
    compare_data_page()