import pandas as pd
import numpy as np

def latlon_to_local(lat, lon, lat0, lon0):
    lat = np.radians(lat)
    lon = np.radians(lon)

    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    EARTH_RADIUS = 6378137.0
    north = (lat - lat0) * EARTH_RADIUS
    east = (lon - lon0) * EARTH_RADIUS * np.cos(lat0)

    return east, north


if __name__ == "__main__":
    gnssinfo = pd.read_csv("gnssinfo.csv")
    flightinfo = pd.read_csv("flightinfo.csv")

    lat0 = gnssinfo.iloc[0]["latitude"]
    lon0 = gnssinfo.iloc[0]["longitude"]

    gnssinfo["x"] = 0.0
    gnssinfo["y"] = 0.0

    gnssinfo["x"], gnssinfo["y"] = latlon_to_local(gnssinfo["latitude"].values, gnssinfo["longitude"].values, lat0, lon0)

    # i tu jest problem bo probkowania sa rozne i trzeba interpolowac
    gnssinfo["z"] = np.interp(gnssinfo["ts"], flightinfo["ts"], flightinfo["height"])

    gnssinfo["z"] -= gnssinfo["z"].iloc[0]

    result = gnssinfo[["ts", "x", "y", "z"]]

    print(result.head())

    result.to_csv("trajectory_meters.csv", index=False)