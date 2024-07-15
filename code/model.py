#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 23:59:19 2024

@author: lefumaqelepo
"""
import os
import data_funcs
import util_funcs
from pvlib.iotools import get_pvgis_tmy
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array, FixedMount, SingleAxisTrackerMount
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import retrieve_sam
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


this_file_path = os.path.dirname(__file__)
results = os.path.abspath(os.path.join(this_file_path, '..', 'Results'))

def energy_yield(latitude, longitude, timezone, altitude, name, mount_type, surface_azimuth, module, inverter, temp_model_params):
    # define location
    location = Location(latitude, longitude, tz = timezone, altitude = altitude, name = name)
    
    try:
        # Get weather
        weather = get_pvgis_tmy(latitude, longitude)[0]
        
        # define mount
        if mount_type == 'fixed':
            mount = FixedMount(surface_tilt = abs(latitude), surface_azimuth = surface_azimuth)
        elif mount_type == 'single_axis_tracker':
            mount = SingleAxisTrackerMount()
        
        # Define array
        array = Array(mount, module_parameters = module, temperature_model_parameters = temp_model_params)
        system = PVSystem(arrays = [array], inverter_parameters = inverter)
        mc = ModelChain(system, location)
        mc.run_model(weather)
        
        res = mc.results.ac.sum()
        return res
    except:
        return 0


def main():
    global data
    data = data_funcs.import_latlon_data()
    # exclude_cs = ['CPV', 'COM', 'MUS', 'SYC', 'STP']
    # data = data[~data.shapeGroup.isin(exclude_cs)]
    data['is_land'] = data.apply(lambda x: util_funcs.is_land(x.latitude, x.longitude), axis=1)
    data = data[data.is_land == True]
    data['tmz'] = data['shapeGroup'].apply(util_funcs.allocate_tmz)
    els = []
    for idx in data.index:
        series = data.loc[idx]
        el = util_funcs.get_elevation(series['longitude'], series['latitude'])
        els.append(el)
        print("Done with index {}".format(idx))
    data['elevation'] = els # data.apply(lambda x: util_funcs.get_elevation(x.longitude, x.latitude), axis=1)
    data['surface_azimuth'] = data['latitude'].apply(util_funcs.surface_azimuth)
    
    # Define module and inverter
    print("Define setup")
    cec_modules = retrieve_sam('SandiaMod')
    cec_inverters = retrieve_sam('CECInverter')
    module = cec_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_model_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    print("Starting heavy metal")
    fixed_yield = []
    for idx in data.index:
        x = data.loc[idx]
        res =  energy_yield(
            x['latitude'], 
            x['longitude'], 
            x['tmz'], 
            x['elevation'], 
            x['shapeName'], 
            'fixed', 
            x['surface_azimuth'], 
            module, 
            inverter, 
            temperature_model_params
            )
        fixed_yield.append(res)
        print("Done with iteration {} of Fixed".format(idx))
    data['fixed_yield'] = fixed_yield
    print("Half way through heavy metal")
    tracking_yield = []
    for idx in data.index:
        x = data.loc[idx]
        res =  energy_yield(
            x['latitude'], 
            x['longitude'], 
            x['tmz'], 
            x['elevation'], 
            x['shapeName'], 
            'single_axis_tracker', 
            x['surface_azimuth'], 
            module, 
            inverter, 
            temperature_model_params
            )
        tracking_yield.append(res)
        print("Done with iteration {} of Tracking".format(idx))
    data['tracking_yield'] = tracking_yield
    print("Done with heavy metal")
    data.to_csv(os.path.join(results, 'results.csv'))

if __name__ == '__main__':
    
    main()