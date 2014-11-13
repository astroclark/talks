#!/usr/bin/env python

import numpy as np
from numpy import random
from pylal import Fr,antenna
import lal
import pmns_utils
import pmns_simsig

import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
matplotlib.rc('font', size=16) 
from matplotlib import pyplot as pl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


#
# Load frame data
#
h1data = Fr.frgetvect('H-H1-871727590-513.gwf', 'H1:LSC-STRAIN', span=500)

#
# Put frame data in lal TimeSeries
#
noise = lal.CreateREAL8TimeSeries('H1', lal.LIGOTimeGPS(h1data[1]), 0,
        h1data[3][0], lal.StrainUnit, len(h1data[0]))
noise.data.data = h1data[0]

nonoise = lal.CreateREAL8TimeSeries('H1', lal.LIGOTimeGPS(h1data[1]), 0,
        h1data[3][0], lal.StrainUnit, len(h1data[0]))
nonoise.data.data = np.zeros(len(h1data[0]))

#
# Generate a post-merger signal
#
epoch = h1data[1]+250
trigtime = epoch+0.5
distance = 5

# Sky angles
inj_ra  = -1.0*np.pi + 2.0*np.pi*np.random.random()
inj_dec = -0.5*np.pi + np.arccos(-1.0 + 2.0*np.random.random())
inj_pol = 2.0*np.pi*np.random.random()
inj_inc = 0.5*(-1.0*np.pi + 2.0*np.pi*np.random.random())
inj_phase = 2.0*np.pi*random.random()

# Antenna response
det1_fp, det1_fc, det1_fav, det1_qval = antenna.response(
        epoch, inj_ra, inj_dec, inj_inc, inj_pol, 
        'radians', 'H1')

injoverhead=True
if injoverhead:
    # set the injection distance to that which yields an effective distance
    # equal to the targeted fixed-dist
    inj_distance = det1_qval*distance
else:
    inj_distance = np.copy(distance)

ext_params = pmns_simsig.ExtParams(distance=inj_distance, ra=inj_ra,
        dec=inj_dec, polarization=inj_pol, inclination=inj_inc, phase=inj_phase,
        geocent_peak_time=trigtime)

waveform = pmns_utils.Waveform('nl3_135135_lessvisc')
waveform.compute_characteristics()

# Construct the time series for these params
waveform.reproject_waveform(theta=ext_params.inclination,
        phi=ext_params.phase)

det1_data = pmns_simsig.DetData(det_site="H1", noise_curve='aLIGO',
        waveform=waveform, ext_params=ext_params, duration=1, seed=101,
        epoch=epoch, f_low=10, taper='False', signal_only=True)

signal = lal.CreateREAL8TimeSeries('signal',
        epoch, 0, det1_data.td_response.delta_t,
        lal.StrainUnit, len(det1_data.td_response))
signal.data.data = det1_data.td_response.data

# Sum Noise and signal
noisysignal = lal.AddREAL8TimeSeries(noise, signal)
noiselesssignal = lal.AddREAL8TimeSeries(nonoise, signal)

noisy = lal.HighPassREAL8TimeSeries(noisysignal, 500, 1, 8)


# Plotting
times = np.arange(0, len(h1data[0])*h1data[3][0], h1data[3][0]) - 100.5

peaktime=times[np.argmax(abs(noiselesssignal.data.data))]
times-=peaktime

f,ax=pl.subplots()
ax.plot(1000*times, noisysignal.data.data, 'k')
ax.plot(1000*times, noiselesssignal.data.data, 'r', linewidth=2)

#ax.set_xlim(1000*(trigtime-h1data[1]-0.0001-100.5-peaktime),
#        1000*(trigtime-100.5-h1data[1]+0.025-peaktime))
ax.set_xlim(1000*-0.004, 1000*0.025)

ax.set_ylim(-0.75e-20, 0.75e-20)

#
# CWT
#
import cwt

########################
# Wavelets Decomposition

#
# CWT from pyCWT
#
def scale2hertz(motherf0, scale, sample_rate):
    fNy=0.5*sample_rate
    return 2*fNy * motherf0 / scale

#scales = np.logspace(np.log2(1), np.log2(256), base=2)

sample_rate = 16384

startidx = np.argmax(abs(noiselesssignal.data.data)) - 2.5e-3 * 16384

data = noisysignal.data.data[startidx:startidx+400]
#data = noiselesssignal.data.data[startidx:startidx+512]
data_noisefree = noiselesssignal.data.data[startidx:startidx+400]

motherfreq = 1.5
scalerange = motherfreq/(np.array([10,0.5*sample_rate])*(1.0/sample_rate))

scales = np.arange(scalerange[1],scalerange[0])

mother_wavelet = cwt.Morlet(len_signal = len(data), scales = scales,
        sampf=sample_rate, f0=motherfreq)

#mother_wavelet = cwt.SDG(len_signal = len(data), scales = scales, normalize = True,
#        fc = 'center')

wavelet = cwt.cwt(data, mother_wavelet)

# --- Plotting

# convert to pseudo frequency
freqs = scale2hertz(mother_wavelet.fc, scales, sample_rate)

peaktime = np.argmax(abs(data_noisefree))/16384.
time = np.arange(0, len(data)/16384.0, 1.0/16384) - peaktime

#collevs=np.linspace(0, max(map(max,abs(wavelet.coefs)**2)), 100)

fpeakmin=2000.0
collevs=np.linspace(0, 1*max(wavelet.get_wps()[freqs>fpeakmin]), 100)

fig, ax_cont = pl.subplots(figsize=(10,8))
c=ax_cont.contourf(time,freqs,np.abs(wavelet.coefs)**2, levels=collevs,
        cmap=cm.gnuplot2)


ax_cont.set_xlim(min(time),max(time))
ax_cont.set_ylim(1,0.5*sample_rate)
ax_cont.set_xlabel('Time [s]')
ax_cont.set_ylabel('Frequency [Hz]')
ax_cont.set_ylim(1000,4000)

#pl.show()
#sys.exit()

divider = make_axes_locatable(ax_cont)

# time-series
ax_ts = divider.append_axes("top", 2.5, sharex=ax_cont)
ax_ts.plot(time, data, 'k')
ax_ts.plot(time, data_noisefree, 'r', linewidth=2)
ax_cont.set_xlim(min(time),max(time))
ax_ts.set_ylim(-max(abs(data)), max(abs(data)))
ax_ts.minorticks_on()

# fourier spectrum
ax_fs = divider.append_axes("right", 1.2, sharey=ax_cont)
ax_fs.plot(wavelet.get_wps(),freqs, 'k')
ax_fs.set_ylim(1000,4000)#0.5*sample_rate)
#ax_fs.set_xlim(0.01*max(wavelet.get_wps()),1.1*max(wavelet.get_wps()))
ax_fs.set_xlim(0,1e-41)
ax_fs.set_xticks(np.arange(0, 1.5e-41, 5e-42))
ax_fs.minorticks_on()
pl.setp(ax_ts.get_xticklabels()+ax_fs.get_yticklabels(),visible=False)

pl.setp(ax_ts.get_xticklabels(),visible=False)

pl.tight_layout()

pl.show()
