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

## speci data report filter that plots longitude and latitude of only speci data reports
def ifspeci(metars_data):
    metar_types = metars_data['metar_type']
    latfilt = metars_data['latitude'][metar_types == 'SPECI']
    longfilt = metars_data['longitude'][metar_types == 'SPECI']
    plt.scatter(longfilt, latfilt, s=0.5)
    plt.savefig(os.path.join(save_image_path, 'SPECI reports'))
    plt.show()
    print("test")
    return latfilt, longfilt

def main():
    given_data = parse_data(save_path)
    ifspeci(given_data)

if __name__ == "__main__":
    main()
