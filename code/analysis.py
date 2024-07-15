#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 08:11:48 2024

@author: lefumaqelepo
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns
from statsmodels.stats import stattools as stl
from constants import country_abbrs
from matplotlib.text import Text
from matplotlib.collections import LineCollection


fig_ext = '.png'
this_file_path = os.path.dirname(__file__)
results = os.path.abspath(os.path.join(this_file_path, '..', 'Results'))
figures = os.path.abspath(os.path.join(this_file_path, '..', 'Figures'))
data = os.path.abspath(os.path.join(this_file_path, '..', 'Data'))

def import_results():
    data = pd.read_csv(os.path.join(results, 'results.csv'))
    cols = data.columns
    data = data.drop(columns = [cols[0]])
    data = data[(data.fixed_yield > 0 ) & (data.tracking_yield > 0)]
    return data

def import_tariffs():
    tariff_data = pd.read_excel(os.path.join(data, 'electricity_prices.xlsx'))
    tariff_dfs = []
    for col in list(tariff_data.columns)[1:]:
        df = tariff_data.filter(items=['country', col])
        df = df.rename(columns={col:'tariff'})
        tariff_dfs.append(df)
    tariff_df = pd.concat(tariff_dfs)
    tariff_df = tariff_df.groupby(['country']).mean().reset_index()
    tariff_df = tariff_df[tariff_df.country.isin(list(country_abbrs.keys()))]
    tariff_df['shapeGroup'] = tariff_df['country'].apply(lambda x: country_abbrs[x])
    return tariff_df

def import_shape_file():
    file_path = os.path.join(data, 'SSA_geoBoundariesCG_AZ_ADM2', 'ssa_ADM2.shp')
    df = gpd.read_file(file_path)
    return df

def data_merger(df1, df2, df3):
    df1_c = df1.copy(deep = True)
    df2_c = df2.copy(deep = True)
    df2_c = df2_c[df2_c.shapeName.isin(df1_c.shapeName)]
    dfs_new = pd.merge(
        df1_c,
        df2_c,
        on = ['shapeName', 'shapeID', 'shapeGroup', 'shapeType'],
        how = 'left'
        )
    dfs_new = dfs_new[dfs_new.shapeGroup.isin(df3.shapeGroup)]
    dfs_new = dfs_new.dropna()
    dfs_new = dfs_new.merge(df3, on = ['shapeGroup'], how = 'left')
    return gpd.GeoDataFrame(dfs_new)

def calcs(dataframe):
    df = dataframe.copy(deep = True)
    inc_fun = lambda x1, x2: (x2 - x1) / x1 * 100
    df['yield_increase'] = df.apply(lambda x: inc_fun(x.fixed_yield, x.tracking_yield), axis=1)
    df['marginal_revenue'] = df['yield_increase'] * df['tariff'] / 100
    df['cap_reduction'] = 100 - df['fixed_yield'] / df['tracking_yield'] * 100
    return df


def mapper(df, ax, cmap, variable, varname=None, norm='LogNorm'):
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    if varname == None:
        varname = variable
    if norm == 'LogNorm':
        norm = colors.LogNorm(vmin=df[variable].min(), vmax=df[variable].max())
    else:
        norm = colors.Normalize(vmin=df[variable].min(), vmax=df[variable].max())
    df.plot(column = variable, 
            ax = ax, 
            legend = True, 
            cmap = cmap, 
            norm = norm,
            edgecolors = 'w', 
            lw = 0.05, 
            missing_kwds = {'color':'none', 'hatch':'none'},
            legend_kwds = {'shrink':0.5, 'label':'{}'.format(varname)})
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis("off")


def plot_yield_gain(df):
    fig, ax = plt.subplots(dpi=150)
    mapper(df, ax, 'plasma', 'yield_increase', 'Annual Energy Yield Increase [%]', norm=1)
    ax.text(-15, -16, 'Min = {:.2f}%'.format(df.yield_increase.min()), fontsize=7)
    ax.text(-15, -20, 'Mean = {:.2f}%'.format(df.yield_increase.mean()), fontsize=7)
    ax.text(-15, -24, 'Median = {:.2f}%'.format(df.yield_increase.median()), fontsize=7)
    ax.text(-15, -28, 'Max = {:.2f}%'.format(df.yield_increase.max()), fontsize=7)
    plt.savefig(os.path.join(figures, 'yield_gain' + fig_ext), bbox_inches='tight')
    
def plot_yield_gain_cdf(df):
    mean = df.yield_increase.mean()
    median = df.yield_increase.median()
    stddev = df.yield_increase.std()
    yi = df.yield_increase.values
    kurtosis = np.mean(((yi - mean)/stddev)**4)
    skewness = np.mean(((yi - mean)/stddev)**3)
    fig, ax = plt.subplots(dpi=150)
    sns.ecdfplot(data=df, x='yield_increase', ax=ax, linewidth=1, label = 'CDF')
    ax.text(4, 0.25, 'Skewness = {:.2f}'.format(skewness), fontsize=10)
    ax.text(4, 0.2, 'Kurtosis = {:.2f}'.format(kurtosis), fontsize=10)
    ax.axvline(mean, ls = '--', lw = 1, color = 'k', label = 'Mean')
    ax.axvline(median, ls = '--', lw = 1, color = 'b', label = 'Median')
    ax.set_xlabel('Yield increase [%]')
    fig.legend(ncol=3, fontsize=8, loc=3, bbox_to_anchor=(0.25, -0.07), frameon=True)
    plt.savefig(os.path.join(figures, 'yield_gain_cdf' + fig_ext), bbox_inches='tight')
    

def plot_marginal_revenue(df):
    fig, ax = plt.subplots(dpi=150)
    mapper(df, ax, 'plasma', 'marginal_revenue', 'Marginal Revenue [US$c/kWh]', norm=1)
    df.plot(ax=ax, edgecolor='b', lw=0.08, facecolor='none')
    plt.savefig(os.path.join(figures, 'marginal_revenue' + fig_ext), bbox_inches='tight')
    

def plot_marginal_revenue_dists(df):
    df = dfc.copy(deep = True)
    dfm = df[['shapeGroup', 'marginal_revenue']]
    dfm = dfm.groupby(['shapeGroup']).mean().reset_index()
    dfm.columns = [dfm.columns[0], 'mean_marginal_revenue']
    df = df.merge(dfm, on=['shapeGroup'], how='left')
    df = df.sort_values(by=['mean_marginal_revenue'])
    countries = df.shapeGroup.unique()
    allowance = 0.1
    lines = []
    colors_ = []
    cmap = plt.colormaps['viridis']
    dmin = df.marginal_revenue.min()
    dmax = df.marginal_revenue.max()
    norm = colors.Normalize(vmin=dmin, vmax=dmax)
    for idx in range(len(countries)):
        country = countries[idx]
        data = df[df.shapeGroup == country]
        xvals = data.marginal_revenue.values
        norms = (xvals - dmin) / (dmax - dmin) * 0.99
        lower_bound = idx + allowance - 0.5
        upper_bound = idx + 0.5 - allowance 
        lines += [[(xval, lower_bound), (xval, upper_bound)] for xval in xvals]
        colors_ += [str(colors.rgb2hex(cmap(norm))) for norm in norms]
        
    line_collections = LineCollection(lines, colors=colors_)
    
    fig, ax = plt.subplots(dpi = 150)
    for idx in range(len(countries)):
        ax.axhline(idx + 0.5, linewidth = 0.1, color = 'k', ls = '--')
    ax.add_collection(line_collections)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=6)
    axcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), fraction = 0.03, aspect = 40, pad = 0.02)
    axcb.set_label('Marginal Revenue [$c/kWh]')
    ax.autoscale()
    
    ytick_labels = ax.get_yticklabels()
    yticks = [xtl.get_position()[1] for xtl in ytick_labels]
    ytlabs = [xtl.get_text() for xtl in ytick_labels]
    ax.set_yticks(yticks, labels = ytlabs, fontsize=6)
    y_major_ticks = ax.yaxis.get_major_ticks()
    y_major_tick_labels = ax.yaxis.get_majorticklabels()
    for y in yticks:
        if (y + 1) % 2 == 1:
            y_major_ticks[y].tick1line.set_markersize(7)
        elif (y + 1) % 2 == 0:
            y_major_ticks[y].tick1line.set_markersize(16)
            y_major_tick_labels[y].set_x(-0.04)
    ax.set_xlabel('$c/kWh')
    plt.savefig(os.path.join(figures, 'marginal_revenue_dists' + fig_ext), bbox_inches='tight')
    #ax.xaxis.set_visible(False)
    

def plot_capital_reduction(df):
    fig, ax = plt.subplots(dpi=150)
    mapper(df, ax, 'plasma', 'cap_reduction', 'Solar PV Capital Reduction [%]', norm=1)
    df.plot(ax=ax, edgecolor='b', lw=0.08, facecolor='none')
    plt.savefig(os.path.join(figures, 'capital_reduction' + fig_ext), bbox_inches='tight')
    
    
def plot_correlation_capred_yieldinc(df):
    fig, ax = plt.subplots(dpi = 150)
    ax.scatter('cap_reduction', 'yield_increase', data=df, s=2)
    
def plot_country_variations(dfc):
    df = dfc.copy(deep = True)
    dfm = df[['shapeGroup', 'yield_increase']]
    dfm = dfm.groupby(['shapeGroup']).mean().reset_index()
    dfm.columns = [dfm.columns[0], 'mean_yield_increase']
    df = df.merge(dfm, on=['shapeGroup'], how='left')
    df = df.sort_values(by=['mean_yield_increase'])
    fig, ax = plt.subplots(dpi=150)
    sns.boxplot(data=df, 
                x='shapeGroup',
                y='yield_increase',
                ax=ax,
                fill=False, 
                legend=False,
                linewidth = 0.6,
                fliersize=2,
                flierprops = {'marker': '.', 'markerfacecolor':'r', 'markersize':5, 'label':'Outlier'},
                showmeans=True,
                meanprops = {'marker':'P', 'markersize':4, 'label':'Mean'}
                )
    xtick_labels = ax.get_xticklabels()
    xticks = [xtl.get_position()[0] for xtl in xtick_labels]
    xtlabs = [xtl.get_text() for xtl in xtick_labels]
    ax.set_xticks(xticks, xtlabs, rotation=90, fontsize=7)
    x_major_ticks = ax.xaxis.get_major_ticks()
    x_major_tick_labels = ax.xaxis.get_majorticklabels()
    for x in xticks:
        if (x + 1) % 2 == 1:
            x_major_ticks[x].tick1line.set_markersize(7)
        elif (x + 1) % 2 == 0:
            x_major_ticks[x].tick1line.set_markersize(24)
            x_major_tick_labels[x].set_y(-0.08)
            
    ax.set_xlabel('Country')
    ax.set_ylabel('Annual yield increase [%]')
    
    hs, ls = ax.get_legend_handles_labels()
    lbls = ['Outlier', 'Mean']
    hndls = []
    for lbl in lbls:
        hndls.append(hs[ls.index(lbl)])
    fig.legend(hndls, lbls, ncol=1, fontsize=8, loc=3, bbox_to_anchor=(-0.02, -0.10), frameon=True)
    plt.savefig(os.path.join(figures, 'country_vars' + fig_ext), bbox_inches='tight')
    
def plot_tariffs(dft):
    df = dft.copy(deep = True)
    df = df.sort_values(by = ['tariff'])
    x = np.arange(0, len(df))
    fig, ax = plt.subplots(dpi=150)
    ax.set_ylim(0, 50)
    bc = ax.bar(x, df.tariff.values, color = 'b', label = 'Tariff')
    ax.bar_label(bc, label_type = 'edge', rotation = 90, fontsize = 6, fmt = '%.2f', padding = 4)
    ax.axhline(df.tariff.mean(), color = 'darkgray', ls = ':', label = 'Mean tariff')
    ax.set_xticks(x, df.shapeGroup, rotation = 90, fontsize = 7)
    x_major_ticks = ax.xaxis.get_major_ticks()
    x_major_tick_labels = ax.xaxis.get_majorticklabels()
    for i in x:
        if (i + 1) % 2 == 1:
            x_major_ticks[i].tick1line.set_markersize(7)
        elif (i + 1) % 2 == 0:
            x_major_ticks[i].tick1line.set_markersize(24)
            x_major_tick_labels[i].set_y(-0.08)
    ax.set_xlabel('Country')
    ax.set_ylabel('US$c/kWh')
    fig.legend(loc = 7, frameon = False, fontsize = 'x-small', bbox_to_anchor = (0.3, 0.8))
    plt.savefig(os.path.join(figures, 'tariffs' + fig_ext), bbox_inches='tight')
    
def plot_locations(dfc):
    fig, ax = plt.subplots(dpi=150)
    ax.set_axis_off()
    dfc.plot(ax=ax, edgecolor='b', lw=0.08, facecolor='none')# legend=True, legend_kwds = {'label':'Admin Level 2 Border'})
    ax.scatter('longitude', 'latitude', data=dfc, s=0.5, color='r', label='Centroid')
    fig.legend(loc = 7, frameon = True, fontsize = 'x-small', bbox_to_anchor = (0.35, 0.2))
    plt.savefig(os.path.join(figures, 'locations' + fig_ext), bbox_inches='tight')
    

def main():
    
    global dfc
    df = import_results()
    dft = import_tariffs()
    dfs = import_shape_file()
    df_merged = data_merger(df, dfs, dft)
    dfc = calcs(df_merged)
    plot_yield_gain(dfc)
    plot_marginal_revenue(dfc)
    plot_marginal_revenue_dists(dfc)
    plot_capital_reduction(dfc)
    plot_correlation_capred_yieldinc(dfc)
    # plot_fixed_vs_tracking(dfc)
    plot_yield_gain_cdf(dfc)
    plot_country_variations(dfc)
    plot_tariffs(dft)
    plot_locations(dfc)
    

if __name__ == '__main__':
    
    main()