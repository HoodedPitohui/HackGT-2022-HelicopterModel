import matplotlib
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

save_path = os.path.join("data")
save_image_path = os.path.join("saved_figures")

def parse_data(given_path):
    file_name = 'metars_cache_1.csv'
    metars_path = os.path.join(given_path, file_name)
    metars_data = pd.read_csv(metars_path)
    return metars_data

def clean_null_rows(given_data):
    cleaned_data = given_data.copy()
    cleaned_data.dropna(subset = ['temp_c'], inplace = True)
    cleaned_data.dropna(subset = ['dewpoint_c'], inplace = True)
    cleaned_data.dropna(subset = ['visibility_statute_mi'], inplace = True)
    return cleaned_data

def get_na_data(given_data):
    na_data = given_data[given_data.latitude > 31]
    na_data = na_data[na_data.latitude < 51]
    na_data = na_data[na_data.longitude > -180]
    na_data = na_data[na_data.longitude < -50]
    return na_data


def geoplot_data_world(given_data, save_name):
    geometry = [Point(xy) for xy in zip(given_data['longitude'], given_data['latitude'])]
    gdf = GeoDataFrame(given_data, geometry=geometry)   
    #this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=0.5)
    plt.savefig(os.path.join(save_image_path, save_name))

def plot_data_usa(given_data):
    f1 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
                c = given_data['temp_c'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('temp_c over latitude and longitude')
    cb1 = plt.colorbar(f1)
    plt.savefig(os.path.join(save_image_path, 'temp_c_usa.png'))

    f2 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
                c = given_data['dewpoint_c'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('dewpoint_c over latitude and longitude')
    cb1.remove()
    cb2 = plt.colorbar(f2)
    plt.savefig(os.path.join(save_image_path, 'dewpoint_c_usa.png'))

    f3 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
                c = given_data['wind_dir_degrees'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('temp_c over latitude and longitude')
    cb2.remove()
    cb3 = plt.colorbar(f3)
    plt.savefig(os.path.join(save_image_path, 'wind_dir_degrees_usa.png'))

    f4 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
            c = given_data['wind_speed_kt'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('wind_speed_kt over latitude and longitude')
    cb3.remove()
    cb4 = plt.colorbar(f4)
    plt.savefig(os.path.join(save_image_path, 'wind_speed_kt_usa.png'))

    f5 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
            c = given_data['visibility_statute_mi'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('visibility_statute_mi over latitude and longitude')
    cb4.remove()
    cb5 = plt.colorbar(f5)
    plt.savefig(os.path.join(save_image_path, 'visibility_usa.png'))


def main():
    given_data = parse_data(save_path)
    #plot_data_world(given_data, 'initial_lat_lon.png')
    cleaned_data = clean_null_rows(given_data)
    #plot_data_world(cleaned_data, 'cleaned_lat_lon.png')
    na_data = get_na_data(cleaned_data)
    plot_data_usa(na_data)
    #plot_data_united_states(cleaned_data)
    print('test')

if __name__ == "__main__":
    main()