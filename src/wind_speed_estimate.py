from cmath import cos, log, sin
import math
import matplotlib
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

save_path = os.path.join("data")
save_image_path = os.path.join("saved_figures")

def parse_data(given_path):
    file_name = 'metars_cache_1.csv'
    metars_path = os.path.join(given_path, file_name)
    metars_data = pd.read_csv(metars_path)
    return metars_data


def filter_usa(df: pd.DataFrame) -> pd.DataFrame:
    """Takes in world data and crudely restricts to just lat/long values corresponding to the lower 48"""

    # We'll be using latitude bounds of [24, 49] and longitude bounds of [-125, -67] based on https://www.findlatitudeandlongitude.com/l/Lower+48/4315442/
    return df.loc[(df['latitude'] >= 24) & (df['latitude'] <= 49) & (df['longitude'] >= -125) & (df['longitude'] <= -67)]

def ifspeci(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df['metar_type'] == 'SPECI')]
## speci data report filter that plots longitude and latitude of only speci data reports
##def ifspeci(metars_data):
    ##metar_types = metars_data['metar_type']
    ##latfilt = metars_data['latitude'][metar_types == 'SPECI']
    ##longfilt = metars_data['longitude'][metar_types == 'SPECI']
    ##return longfilt, latfilt

## estimate of wind speed at given height assuming average weather station measured
## from 5 m off of ground
def wind_est(metars_data, height, ref_height):
    wind_speed = metars_data['wind_speed_kt']
    wind_est = wind_speed * ((math.log(height/ref_height))**2)

    wind_dir = metars_data['wind_dir_degrees']

    wind_est_arr = np.array(wind_est)
    wind_dir_arr = np.array(wind_dir)
    x_comp = wind_est_arr * np.cos(wind_dir_arr)

    y_comp = wind_est_arr * np.sin(wind_dir_arr)

    widths = np.linspace(0, 200, metars_data['longitude'].size)
    plt.quiver(metars_data['longitude'], metars_data['latitude'], x_comp, y_comp, linewidth=widths)
    plt.savefig(os.path.join(save_image_path, 'vectors speci'))
    plt.show()
    return wind_est_arr, wind_dir_arr, x_comp, y_comp


def main():
    given_data = parse_data(save_path)
    usa_frame = filter_usa(given_data)
    speci_frame = ifspeci(usa_frame)
    wind_est(speci_frame, 1524, 5.0) ## 1524 m or 5000 ft wanted height estimate
    print('test')


if __name__ == "__main__":
    main()