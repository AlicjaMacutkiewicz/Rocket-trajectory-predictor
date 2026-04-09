import cdsapi


dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "surface_pressure"
    ],
    "year": ["1940"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [54.37, 18.38, 54.12, 18.63]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
