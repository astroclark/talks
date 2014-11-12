#!/usr/bin/env python

import numpy as np
from numpy import random
from pylal import Fr,antenna
import lal
import pmns_utils
import pmns_simsig

from matplotlib import pyplot as pl

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
epoch = h1data[1]+100
trigtime = epoch+0.5
distance = 10

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
pl.show()


