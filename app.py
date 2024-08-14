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
    census_tracts = gpd.read_file('tl_2020_us_zcta520.shp', crs='EPSG:4326')
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

# Title
st.title('Fair Market Housing Data Exploration')

# Description
st.write('This is a simple web app that allows you to explore the Fair Market Housing data for the 2024 Fiscal Year.')
st.write('The data is sourced from the U.S. Department of Housing and Urban Development (HUD) and uses UnitedStatesZipCodes.org for location specific data.')

# Display the data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

st.write('---')

# Add Folium Map
st.subheader('Map of Fair Market Rent')
chosen_location = st.selectbox('Select a Location', map_df['Location'].unique(), index=None)
chosen_radius = st.slider('Select a Radius (miles)', 5, 250, 50)


st.write('---')

if chosen_location:
    location_data = map_df[map_df['Location'] == chosen_location].iloc[:1]
    chosen_lat = location_data['Latitude'].iloc[0]
    chosen_lon = location_data['Longitude'].iloc[0]
    radius_miles = chosen_radius
    filtered_df = limit_data_by_radius(map_df, chosen_lat, chosen_lon, radius_miles)
    
    map = create_folium_map(filtered_df, location_data)
    st_folium(map, width=1200, height=900)

    st.write('---')

with st.sidebar:
    st.title('Filter the Data')

    # Choose how many bedrooms to display
    bedrooms = st.multiselect('Select number of bedrooms', 
                            ['Studio','1BR','2BR','3BR','4BR',])

    # Filter by Metro Area
    metro_areas = df['Metro Area'].unique()
    selected_areas = st.multiselect('Select Metro Areas', sorted(metro_areas))

    filtered_df1 = df[df['Metro Area'].isin(selected_areas)]

    # Filter by bedrooms
    pre = ['ZIP Code','Metro Area','City','State']
    post = ['County','Latitude','Longitude']
    cols = pre + sorted(bedrooms) + post
    filtered_df2 = filtered_df1[cols]

st.subheader('Display Filtered Data')
st.write(filtered_df2)

# Group by Metro Area
if not filtered_df2.empty:
    st.subheader('Display Fair Market Rent by Metro Area')
    grouped = filtered_df2.groupby('Metro Area')[sorted(bedrooms)].mean()
    st.write(round(grouped, 2))
    
