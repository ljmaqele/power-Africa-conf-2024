#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:32:30 2024

@author: lefumaqelepo
"""
import os
import requests
import geopandas as gpd
from pyhigh import get_elevation_batch
from constants import timezones
from global_land_mask import globe


def allocate_tmz(country_abbr):
    return timezones[country_abbr]


def get_elevation(lon, lat):
    url = "https://api.opentopodata.org/v1/srtm90m"
    data = {
        "locations" : str(lat) + "," + str(lon), "interpolation" : "cubic"
        }
    try:
        res = requests.post(url, json = data).json()
        return res["results"][0]["elevation"]
    except:
        return 0
    
    
def surface_azimuth(latitude):
    if latitude < 0:
        return 0
    elif latitude > 0:
        return 180
    else:
        return 0
    
def is_land(lat, lon):
    return globe.is_land(lat, lon)



if __name__ == '__main__':
    
    lat, lon = -24.766, 30.158
    
    print(get_elevation(lon, lat))