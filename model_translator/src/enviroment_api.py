import cdsapi
import rocketpy 
import xarray as xr


# date must be datetime type
# PROVIDED PATH SHOULD NOT CONTAINT FILE NAME ONLY DIRECTORY
def get_enviroment_from_date(date,filename, path="../../source_model/ERA5_weather/single/"):
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "surface_pressure"
        ],
        "year": [str(date.date.year)],
        "month": [str(date.date.month)],
        "day": [str(date.date.day)],
        "time": [date.time.isoformat(timespec='minutes')],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [54.37, 18.38, 54.12, 18.63]
    }
    
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(path+filename)
    sl = xr.open_dataset(path+filename)
    
