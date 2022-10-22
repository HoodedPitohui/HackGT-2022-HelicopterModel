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

def plot_data_world(given_data):
    geometry = [Point(xy) for xy in zip(given_data['longitude'], given_data['latitude'])]
    gdf = GeoDataFrame(given_data, geometry=geometry)   
    #this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=0.5)
    plt.savefig(os.path.join(save_image_path, 'countries_lat_lon_plot.png'))

def plot_data_north_america(given_data):
    fig, ax = plt.subplots(figsize=(8,6))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    countries[countries["name"] == "Germany"].plot(color = "lightgrey",
                                                        ax = ax)
    #given_data.plot(x = "longitude", y = "latitude", kind = "scatter")
    plt.show()

def main():
    given_data = parse_data(save_path)
    #plot_data_world(given_data)
    plot_data_north_america(given_data)

    print('test')

if __name__ == "__main__":
    main()