import matplotlib
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from scipy.interpolate import interp1d


save_path = os.path.join("data")
save_image_path = os.path.join("saved_figures")

def parse_data(given_path):
    file_name = 'metars_cache_1.csv'
    metars_path = os.path.join(given_path, file_name)
    metars_data = pd.read_csv(metars_path)
    metars_data.sort_values('latitude')
    #sort by latitude
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
    plt.title('Air Temperature over latitude and longitude')
    cb1 = plt.colorbar(f1)
    plt.savefig(os.path.join(save_image_path, 'temp_c_usa.png'))

    f2 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
                c = given_data['dewpoint_c'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('Dewpoint Temperature over latitude and longitude')
    cb1.remove()
    cb2 = plt.colorbar(f2)
    plt.savefig(os.path.join(save_image_path, 'dewpoint_c_usa.png'))

    f3 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
                c = given_data['wind_dir_degrees'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('Direction from which the wind is blowing (0 = variable) over latitude and longitude')
    cb2.remove()
    cb3 = plt.colorbar(f3)
    plt.savefig(os.path.join(save_image_path, 'wind_dir_degrees_usa.png'))

    f4 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
            c = given_data['wind_speed_kt'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('Wind speed over latitude and longitude')
    cb3.remove()
    cb4 = plt.colorbar(f4)
    plt.savefig(os.path.join(save_image_path, 'wind_speed_kt_usa.png'))

    f5 = plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5,
            c = given_data['visibility_statute_mi'])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('latitude (degrees)')
    plt.title('Horizontal visibility over latitude and longitude')
    cb4.remove()
    cb5 = plt.colorbar(f5)
    plt.savefig(os.path.join(save_image_path, 'visibility_usa.png'))
    cb5.remove()

def plot_user_points(given_data, given_lats, given_lons, save_name):
    plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5)
    plt.scatter(given_lons, given_lats, s = 20)
    plt.savefig(os.path.join(save_image_path, save_name))
    plt.close()
    #plt.show()

def find_relevant_points(given_data, user_lats, user_lons):
    max_lat = max(user_lats)
    min_lat = min(user_lats)
    max_lon = max(user_lons)
    min_lon = min(user_lons)

    relevant_data = given_data[given_data.latitude < max_lat]
    relevant_data = relevant_data[relevant_data.latitude > min_lat]
    relevant_data = relevant_data[relevant_data.longitude < max_lon]
    relevant_data = relevant_data[relevant_data.longitude > min_lon]
    return relevant_data

def generate_risk_assessments(given_data, wind_speed, temp_c, dewpoint_c):
    risk_array = np.empty((len(given_data), 1))
    for i in range(0, len(given_data)):
        risk_array[i] = wind_speed[i] ** 2 + temp_c[i] ** (1/2) + abs(dewpoint_c[i]) ** (1/2)
    return risk_array

def interpolate_metrics(given_data, point_density, risk_array): #given_data should be sorted ascending in altitude
    #example point_density = 0.01
    lat_range = max(given_data['latitude']) - min(given_data['latitude'])
    lat_length = round(lat_range / point_density)
    long_range = max(given_data['longitude']) - min(given_data['longitude'])
    long_length = round(long_range / point_density)
    interp_vec = np.empty((lat_length * long_length, 3))
    for i in range(0, lat_length):
        for j in range(0, long_length):
            interp_vec[i * long_length + j, 0] = min(given_data['latitude']) + point_density * i
            interp_vec[i * long_length + j, 1] = min(given_data['longitude']) + point_density * j
            neighbor_array, distance_array = find_nearest_neighbors(np.asarray(given_data['latitude']), 
                                                                    np.asarray(given_data['longitude']), 
                                                                    interp_vec[i * long_length + j, 0], 
                                                                    interp_vec[i * long_length + j, 1], 5)
        
            interp_vec[i * long_length + j, 2] = calculate_weighted_risk(neighbor_array, distance_array, risk_array)

    return interp_vec

def calculate_weighted_risk(neighbor_array, distance_array, risk_array):
    weighted_risk = 0
    for i in range(0, len(neighbor_array)):
        test = neighbor_array[i]
        weighted_risk = weighted_risk + risk_array[int(neighbor_array[i])] / (distance_array[0, i] ** 2 + 1)
    return weighted_risk
    
def find_nearest_neighbors(given_lats, given_lons, lat, lon, num_neighbors):
    distances = np.empty((1, len(given_lats)))
    for i in range(0, len(given_lats)):
        distances[0, i] = ((lat - given_lats[i]) **2 + (lon - given_lons[i]) **2) **(1/2)
    
    sorted_distances = np.sort(distances)
    neighbor_array = np.empty((num_neighbors, 1))
    for i in range(0, len(neighbor_array)):
        f_index = np.where(distances == sorted_distances[0, i])
        if (len(f_index[1]) > 1):
            neighbor_array[i] = f_index[0][0]
        else:
            neighbor_array[i] = f_index[1]

    return neighbor_array, sorted_distances[0: 5]

def main():
    given_data = parse_data(save_path)
    #plot_data_world(given_data, 'initial_lat_lon.png')

    #clean out null rows
    cleaned_data = clean_null_rows(given_data)
    #plot_data_world(cleaned_data, 'cleaned_lat_lon.png')
    na_data = get_na_data(cleaned_data)

    #get the data in the USA
    plot_data_usa(na_data)
    #plot_data_united_states(cleaned_data)
    print('test')

    #pick two points and find useful datapoints from there
    lat1, lon1 = 33.7490, -84.3880
    lat2, lon2 = 35.9940, -78.8986

    given_lats = [lat1, lat2]
    given_lons = [lon1, lon2]
    plot_user_points(na_data, given_lats, given_lons, 'atlanta-durham_general.png')
    relevant_user_data = find_relevant_points(na_data, given_lats, given_lons)
    plot_user_points(relevant_user_data, given_lats, given_lons, 'atlanta-durham_narrowed.png')
    risk_array = generate_risk_assessments(relevant_user_data, np.asarray(relevant_user_data['wind_speed_kt']), 
                                            np.asarray(relevant_user_data['temp_c']), 
                                            np.asarray(relevant_user_data['dewpoint_c']))
    weighted_risk_matrix = interpolate_metrics(relevant_user_data, 0.01, risk_array)
    interpolate_metrics(given_data, 0.01)

if __name__ == "__main__":
    main()