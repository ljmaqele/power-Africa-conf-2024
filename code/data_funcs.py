#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 07:31:34 2024

@author: lefumaqelepo
"""

import os
import geopandas as gpd
from constants import country_abbrs, timezones

this_file_path = os.path.dirname(__file__)
data = os.path.abspath(os.path.join(this_file_path, '..', 'Data'))


def import_latlon_data():
    file_path = os.path.join(data, 'SSA_ADM2_centroids', 'SSA_ADM2_centroids.shp')
    df = gpd.read_file(file_path)
    df['longitude'] = df['geometry'].apply(lambda x: list(x.coords)[0][0])
    df['latitude'] = df['geometry'].apply(lambda x: list(x.coords)[0][1])
    df = df.drop(columns = 'geometry')
    return df



    