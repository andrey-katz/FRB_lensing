#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:28:43 2019

@author: katza
"""

import numpy as np 
# Units class

class units:
    # Energy and mass
    eV    = 1.
    keV   = 1.e3
    MeV   = 1.e6
    GeV   = 1.e9
    TeV   = 1.e12
    PeV   = 1.e15
    kg    = 5.62e35*eV
    grams = 0.001*kg
    
    # Length and time
    m     = 5.076e6
    meter = m
    km    = 1000*m
    cm    = 0.01*m
    nm    = 1.e-9*m
    fm    = 1.e-15*m
    AU    = 1.4960e11*m
    pc    = 30.857e15*m
    kpc   = 1.e3*pc
    Mpc   = 1.e6*pc
    Gpc   = 1.e9*pc
    ly    = 9460730472580800*m  # light year
    sec   = 1.523e15
    hours = 3600*sec
    days  = 24*hours
    yrs   = 365*days,
    Hz    = 1./sec
    kHz   = 1.e3*Hz
    MHz   = 1.e6*Hz
    GHz   = 1.e9*Hz
    
    # Various astrophysical constants
    GN    = 6.708e-39/1e18  # eV^-2, Newton's constant
    MPl   = 1.22093e19*GeV   # Planck mass, PDG 2013
    Msun  = 1.989e30*kg
    Rsun  = 6.9551e8*meter
    
    # cosmology
    h       = 0.688                          # according to Planck, see Wikipedia; HE97
    H0      = h * 100. * km / sec/ Mpc       # Hubble parameter
    rho_c0  = 3. * H0**2/(8. * np.pi * GN)   # critical density today, Kolb Turner eq. (3.14)
    Omega_m = 0.14 / h**2                    # total matter density
    
    # particle physics
    alpha_em = (1./137.035999139)            # electromagnetic fine structure constant (PDG 2018)
    m_e      = 0.5109989461 * MeV            # electron mass (PDG 2018)


# Routines 

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import pandas as pd
from pandas import Series, DataFrame

import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline 

from scipy.fftpack import fft

def autocorrelation(vec):
    return np.array([np.mean(np.conj(vec) * np.roll(vec, n))/(np.mean(vec)**2) for n in range(len(vec))])


def read_events_metadata(filename):
    metadata_columns = ['M', 'x_diff_ism', 'DS', 'DL', 'DLS', 'zL', 'y', 'Dsc']
    u = units()
    with open(filename.format(), 'rb') as f:
        summary = pickle.load(f)
    metadata = {el: [] for el in metadata_columns}
    for el in summary:
        for ent in metadata_columns:
            metadata[ent].append(el[ent])
    metadata = DataFrame(metadata)
    metadata.M = metadata.M / u.Msun
    metadata.DS = metadata.DS / u.Gpc
    metadata.DLS = metadata.DLS / u.Gpc
    metadata.DL = metadata.DL / u.Gpc
    metadata.Dsc = metadata.Dsc / u.Gpc
    metadata.x_diff_ism = metadata.x_diff_ism / u.cm
    return metadata

def fit_power_spectrum(f_range, spectrum):
    lr = linear_model.Ridge(alpha = 1e-13)
    pf = PolynomialFeatures(degree = 10)
    fs = (f_range / np.mean(f_range)).reshape((-1, 1))
    frp = pf.fit_transform(fs)
    spectrum = np.log(spectrum)
    spectrum = spectrum.reshape((-1, 1))
    lr.fit(frp, spectrum)
    return spectrum, (lr.predict(frp)).reshape((-1,))

def running_average(x, steps):
    return np.convolve(x, np.ones(steps), mode = 'same')/steps    

def read_event(event_number, filename):
    u = units()
    event_columns = ['omega_range', 'spect_table']
    with open(filename.format(), 'rb') as f:
        summary = pickle.load(f)
    event = {el: summary[event_number][el] for el in event_columns}
    event = DataFrame(event)
    event['power'] = abs(event['spect_table'])**2
    event.omega_range = event.omega_range/ u.GHz
    event['f_range'] = event.omega_range/ 2 / np.pi
    event['autocorr_power'] = autocorrelation(event['power'])
    event['autocorr_spectrum'] = autocorrelation(event['spect_table'])
    lgp, pred = fit_power_spectrum(np.array(event.f_range), np.array(event.power))
    event['logpower'] = lgp
    event['fit'] = pred
    event['sub_power'] = event.logpower - event.fit
    return event

scintillation_time = lambda x_diff, dssc, omega: dssc/omega**2 / x_diff**2

round_to_1 = lambda x: round(x, -int(np.floor(np.log10(abs(x)))))

round_to_2 = lambda x: round(x, 1-int(np.floor(np.log10(abs(x)))))

def rE(M, DL, DS):
    u = units()
    return np.sqrt(4 * u.GN * M * DL * (DS - DL)/DS)

def lens_time_delay(M, DL, DS, y):
    xE = rE(M, DL, DS)
    thetaE = xE / DL
    #DLS = DL-DS
    DLS = DS - DL
    beta =  y * thetaE
    theta_p = 0.5 * (beta + np.sqrt(beta**2 + 4 * thetaE**2))
    theta_m = 0.5 * (beta - np.sqrt(beta**2 + 4 * thetaE**2)) 
    return np.abs(DL * DS / DLS * (0.5 * (theta_p - beta)**2 - thetaE**2 * np.log(np.abs(theta_p)) 
                                   - 0.5* (theta_m - beta)**2 + 
                                   thetaE**2 * np.log(np.abs(theta_m))))
    
def fourier_transform(frequencies, spectrum):
    u = units()
    epsilon = 1e-8
    df = np.real((frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)) * u.GHz
    dt = np.real(1 / (frequencies[-1]*u.GHz - frequencies[0]*u.GHz))
    t_range = np.arange(-1/2/df, (1+ epsilon)/2/df, dt) / u.sec
    transform = np.abs(fft(spectrum))**2
    return t_range, transform

def find_peak(X, y, width = 1, log_input = False):
    if not len(X) == len(y):
        print('Dimensions do not match!')
        return None
    start = len(X)//2
    Xshort = X[start:]
    if log_input:
        yshort = y[start:]
    else:
        yshort = np.log(y[start:])
    ## Rescale Xshort:
    Xshort = Xshort/np.mean(Xshort)
    sigmas = []
    all_indices = range(len(Xshort))
    for i in range(len(Xshort)):
        if i - width < 0 or i+width > len(Xshort):
            sigmas.append(0.0)
            continue
        else:
            peak_indices = range(i - width, i+width, 1)
            fit_indices = sorted(list(set(all_indices) - set(peak_indices)))
            X_transformed = Xshort[fit_indices].reshape((-1, 1))
            y_transformed = yshort[fit_indices].reshape((-1, 1))
            significance = 0.0
            for a in [0.1, 0.5, 1., 4.]:
                for d in [2, 5]:
                    model = Pipeline(steps = [('pf', PolynomialFeatures(degree = d)), 
                                              ('linreg', linear_model.Ridge(alpha = a))])
                    model.fit(X_transformed, y_transformed)
                    dev = np.mean(np.abs(y_transformed - model.predict(X_transformed)))
                    s = np.sum(yshort[peak_indices].reshape((-1, 1)) - 
                               model.predict(Xshort[peak_indices].reshape((-1, 1))))
                    significance += (s/dev/len(peak_indices))
            sigmas.append(significance/8)
    return sigmas

def find_peak_index(sigmas, cutoff = 5):
    last = 0
    descending = False
    probing_peak = (-1, -1)
    output_indices = []
    for i, s in enumerate(sigmas):
        if s >= last:
            descending = False
        else:
            descending = True
        last = s
        if (s < cutoff or descending) and probing_peak[0] == -1:
            continue
        if s >= cutoff and probing_peak[0] == -1:
            probing_peak = (i, s)
        if not probing_peak[0] == -1:
            if s > probing_peak[1]:
                probing_peak = (i, s)
            elif s > cutoff and s > probing_peak[1]/2:
                continue
            else:
                output_indices.append(probing_peak)
                probing_peak = (-1, -1)
    return output_indices


def plot_event(num, events, metadata, smearing = [10], peak_width = 2, sigma_cut = 2, produce_plot = True, output = False, sample = 1):
    u = units()
    delta_f = round_to_2(np.real(events[num].loc[1].f_range - events[num].loc[0].f_range))
    convs = []
    for i in smearing:
        p = int(np.log10(i * delta_f) - 1)
        convs.append((i * delta_f * 10**(-p), p))
    sf = ScalarFormatter()
    sf.set_scientific(True)
    sf.set_powerlimits((-2, 2))
    mass = metadata.loc[num].M
    x = int(np.log10(metadata.loc[num].x_diff_ism))
    t, fp = fourier_transform(np.array(events[num].f_range), 
                              np.array(events[num].power))
    startP = len(t)//2
    fp = np.roll(fp, startP)
    ts, fs = fourier_transform(np.array(events[num].f_range), 
                               np.exp(np.array(events[num].sub_power)))
    startS = len(ts)//2
    fs = np.roll(fs, startS)
    smeared = [running_average(np.array(events[num].sub_power), sm) for sm in smearing]
    tsc = []
    fsc = []
    for sm in smeared:
        t, f = fourier_transform(np.array(events[num].f_range), np.exp(sm))
        tsc.append(t)
        fsc.append(f)
    startSC = len(tsc[0])//2
    fsc = [np.roll(f, startSC) for f in fsc]
    ## Run routines to find the peak locations
    delta_t_lens = lens_time_delay(metadata.loc[num].M*u.Msun, metadata.loc[num].DL*u.Gpc, 
                                   metadata.loc[num].DS*u.Gpc, metadata.loc[num].y) / u.sec
    sigmasP = find_peak(t, fp, width = peak_width)
    sigmasS = find_peak(ts, fs, width = peak_width)
    sigmasSC = [find_peak(tsc[0], f, width = peak_width) for f in fsc]
    peaksP = find_peak_index(sigmasP, cutoff = sigma_cut)
    key = lambda tup: tup[1]
    peaksP.sort(key = key, reverse = True)
    peak_timeP = [t[startP:][pl[0]] for pl in peaksP[:2]]
    sigma_P = 0
    dist_P = 100
    for i in range(len(peak_timeP)):
        dist = abs((peak_timeP[i] - delta_t_lens) / (peak_timeP[i] + delta_t_lens))
        if dist < 0.15 and dist < dist_P:
            sigma_P, dist_P = peaksP[i][1], dist
    peaksS = find_peak_index(sigmasS, cutoff = sigma_cut)
    peaksS.sort(key = key, reverse = True)
    peak_timeS = [ts[startS:][pl[0]] for pl in peaksS[:2]]
    sigma_S = 0
    dist_S = 100
    for i in range(len(peak_timeS)):
        dist = abs((peak_timeS[i] - delta_t_lens) / (peak_timeS[i] + delta_t_lens))
        if dist < 0.15 and dist < dist_S:
            sigma_S, dist_S = peaksS[i][1], dist
    peak_timeSC = []
    sigma_SC = []
    dist_SC = []
    for s in sigmasSC:
        peak = find_peak_index(s, cutoff = sigma_cut)
        peak.sort(key = key, reverse = True)
        temp_time_SC = [tsc[0][startSC:][pl[0]] for pl in peak[:2]]
        peak_timeSC.append(temp_time_SC)
        dist_temp = 100
        sigma_temp = 0
        for i in range(len(temp_time_SC)):
            dist = abs((temp_time_SC[i] - delta_t_lens) / (temp_time_SC[i] + delta_t_lens))
            if dist < 0.15 and dist < dist_temp:
                sigma_temp, dist_temp = peak[i][1], dist
        sigma_SC.append(sigma_temp)
        dist_SC.append(dist_temp)
    if produce_plot:
        colors = ['b', 'r', 'darkgreen', 'cyan']
        colors_vlines = ['tomato', 'darkorchid']
        ws = [2.5, 2.0, 1.5, 1.]
        styles = ['solid', 'dashed', 'dashdot', 'dotted']
        fig, ax = plt.subplots(2, 3, figsize = (14, 7))
        ax[0, 0].plot(events[num].f_range, events[num].power, color = 'b', lw = 1.5)
        ax[0, 0].set_yscale('log')
        ax[0, 0].annotate('Power spectrum', xy=(0, 0.5), xytext=(-ax[0, 0].yaxis.labelpad - 7, 0), 
          xycoords=ax[0, 0].yaxis.label, textcoords='offset points', ha='right', va='center', 
          rotation = 90, fontsize = 15)
        ax[0, 0].annotate('Unaltered spectrum', xy=(0.5, 1), xytext=(0, 7), 
          xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline', 
          fontsize = 15)
        ax[1, 0].plot(t[startP:], fp[startP:], color = 'b', lw = 1.5)
        ax[1, 0].set_yscale('log')
        full = [(peak_timeP[i], peaksP[i][1]) for i in range(len(peak_timeP)) if peaksP[i][1] > 5]
        dashed = [(peak_timeP[i], peaksP[i][1]) for i in range(len(peak_timeP)) if peaksP[i][1] > 3 and peaksP[i][1] <= 5]
        dotted = [(peak_timeP[i], peaksP[i][1]) for i in range(len(peak_timeP)) if peaksP[i][1] > 1 and peaksP[i][1] <= 3]
        if len(full) > 0:
            for i in range(len(full)):
                ax[1, 0].vlines(full[i][0], min(fp), max(fp), color = colors_vlines[i], lw = 2., 
                  label = r'detected, $\sigma = {}$'.format(round(full[i][1], 1)))
        if len(dashed) > 0:
            for i in range(len(dashed)):
                ax[1, 0].vlines(dashed[i][0], min(fp), max(fp), color = colors_vlines[i], lw = 2., ls = 'dashed', 
                  label = r'detected, $\sigma = {}$'.format(round(dashed[i][1], 1)))
        if len(dotted) > 0:
            for i in range(len(dotted)):
                ax[1, 0].vlines(dotted[i][0], min(fp), max(fp), color = colors_vlines[i], lw = 2., ls = 'dotted', 
                  label = r'detected, $\sigma = {}$'.format(round(dotted[i][1], 1)))
        ax[1, 0].vlines(delta_t_lens, min(fp), max(fp), color = 'k', ls = '--', lw = 2.3, 
          label = 'BH time delay')
        ax[1, 0].legend(loc = 'upper right')
        ax[1, 0].annotate('Fourier transform', xy=(0, 0.5), 
          xytext=(-ax[1, 0].yaxis.labelpad - 7, 0), xycoords=ax[1, 0].yaxis.label, 
          textcoords='offset points', ha='right', va='center', rotation = 90, fontsize = 15)
        ax[0, 1].plot(events[num].f_range, events[num].sub_power, color = 'b', lw = 1.5)
        ax[0, 1].annotate('Subtracted spectrum', xy=(0.5, 1), xytext=(0, 7), 
          xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline', 
          fontsize = 15)
        ax[1, 1].plot(ts[startS:], fs[startS:], color = 'b', lw = 1.5)
        ax[1, 1].set_yscale('log')
        full = [(peak_timeS[i], peaksS[i][1]) for i in range(len(peak_timeS)) if peaksS[i][1] > 5]
        dashed = [(peak_timeS[i], peaksS[i][1]) for i in range(len(peak_timeS)) if peaksS[i][1] > 3 and peaksS[i][1] <= 5]
        dotted = [(peak_timeS[i], peaksS[i][1]) for i in range(len(peak_timeS)) if peaksS[i][1] > 1 and peaksS[i][1] <= 3]
# =============================================================================
#         ax[1, 1].vlines(peak_timeS, min(fs), max(fs), color = 'tomato', lw = 2., 
#           label = r'detected, $\sigma = {}$'.format(round(sigma_S, 1)))
# =============================================================================
        if len(full) > 0:
            for i in range(len(full)):
                ax[1, 1].vlines(full[i][0], min(fs), max(fs), color = colors_vlines[i], lw = 2., 
                  label = r'detected, $\sigma = {}$'.format(round(full[i][1], 1)))
        if len(dashed) > 0:
            for i in range(len(dashed)):
                ax[1, 1].vlines(dashed[i][0], min(fs), max(fs), color = colors_vlines[i], lw = 2., ls = 'dashed', 
                  label = r'detected, $\sigma = {}$'.format(round(dashed[i][1], 1)))
        if len(dotted) > 0:
            for i in range(len(dotted)):
                ax[1, 1].vlines(dotted[i][0], min(fs), max(fs), color = colors_vlines[i], lw = 2., ls = 'dotted', 
                  label = r'detected, $\sigma = {}$'.format(round(dotted[i][1], 1)))
        ax[1, 1].vlines(delta_t_lens, min(fs), max(fs), color = 'k', ls = '--', lw = 2.3, 
          label = 'BH time delay')
        ax[1, 1].legend(loc = 'upper right')
        for i, sm in enumerate(smeared):
            ax[0, 2].plot(events[num].f_range, sm, color = colors[i], lw = ws[i], ls = styles[i])
        ax[0, 2].annotate('Subtracted and convolved spectrum', xy=(0.5, 1), xytext=(0, 7), 
          xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline', 
          fontsize = 15)
        for i, f in enumerate(fsc): 
            ax[1, 2].plot(tsc[0][startSC:], f[startSC:], color = colors[i], lw = ws[i], 
              ls = styles[i], 
              label = r'$\Delta t_{{conv}} = {0} \times 10^{{{1}}}\, {{\rm GHz}}$, $\sigma_{{BH}} = {2}$'.format(round(convs[i][0], 2), 
                                  convs[i][1], round(sigma_SC[i], 1)))
        ax[1, 2].legend(loc  = 'upper right')
        ax[1, 2].set_yscale('log')
        min_coord = np.min(np.array(fsc))
        max_coord = np.max(np.array(fsc))
        ax[1, 2].vlines(delta_t_lens, min_coord, max_coord, color = 'k', ls = '--', lw = 1.9)
        fig.suptitle(r'$M = {0}\, M_\odot, \ \ \ r_{{diff}} = 10^{{{1}}}\, {{\rm cm}}$'.format(mass, x), 
                     fontsize = 18)
        for i in range(3):
            ax[0, i].set_ylabel('Intensity', fontsize = 13)
            ax[0, i].set_xlabel('Frequency [GHz]', fontsize = 13)
            ax[0, i].xaxis.set_major_formatter(sf)
        for i in range(3):
            ax[1, i].set_ylabel('Intensity', fontsize = 13)
            ax[1, i].set_xlabel('Time delay [sec]', fontsize = 13)
            ax[1, i].xaxis.set_major_formatter(sf)
        fig.tight_layout()
        fig.subplots_adjust(left = 0.1, top=0.87)
        if output:
            fname = 'sample_' + int(sample) + '_point_' + str(num) + '.pdf'
            plt.savefig(fname)
    return sigma_P, dist_P, sigma_S, dist_S, sigma_SC, dist_SC

    

