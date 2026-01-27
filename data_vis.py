import os
import numpy as np
import pandas as pd
import matplotlib
import scipy.io as scipyio
import matplotlib.pyplot as plt

from tqdm import tqdm

# Absolute path of the folder containing the units' folders and scenarioX.csv
scenario_folder = '/home/changyuan/wireless_context/scenario9_dev'

# Fetch scenario CSV
try:
    csv_file = [f for f in os.listdir(scenario_folder) if f.endswith('csv')][0]
    csv_path = os.path.join(scenario_folder, csv_file)
except:
    raise Exception(f'No csv file inside {scenario_folder}.')

# Load CSV to dataframe
dataframe = pd.read_csv(csv_path)
print(f'Columns: {dataframe.columns.values}')
print(f'Number of Rows: {dataframe.shape[0]}')

N_BEAMS = 64
n_samples = 100
pwr_rel_paths = dataframe['unit1_pwr_60ghz'].values
pwrs_array = np.zeros((n_samples, N_BEAMS))

for sample_idx in tqdm(range(n_samples)):
    pwr_abs_path = os.path.join(scenario_folder,
                               pwr_rel_paths[sample_idx])
    pwrs_array[sample_idx] = np.loadtxt(pwr_abs_path)

# Select specific samples to display
selected_samples = [5, 10, 20]
beam_idxs = np.arange(N_BEAMS) + 1
plt.figure(figsize=(10,6))
plt.plot(beam_idxs, pwrs_array[selected_samples].T)
plt.legend([f'Sample {i}' for i in selected_samples])
plt.xlabel('Beam indices')
plt.ylabel('Power')
plt.savefig("beam_power.png", dpi=300, bbox_inches="tight")
plt.close()

img_rel_paths = dataframe['unit1_rgb'].values
fig, axs = plt.subplots(figsize=(10,4), ncols=len(selected_samples), tight_layout=True)
for i, sample_idx in enumerate(selected_samples):
    img_path = os.path.join(scenario_folder, img_rel_paths[sample_idx])
    img = plt.imread(img_path)
    axs[i].imshow(img)
    axs[i].set_title(f'Sample {sample_idx}')
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)

save_path = "selected_samples_rgb.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to {save_path}")




lidar_rel_paths =  dataframe['unit1_lidar_SCR'].values

# Compare noisy with denoised.
lidar_sample_size = 216

# Using the first 40 samples (almost the complete first car pass)
n_samp_first_seq = 40

# Append the lidar samples to array to show across a pass
lidar_frame = np.zeros((n_samp_first_seq, lidar_sample_size))
for sample_idx in range(n_samp_first_seq):
    lidar_file = os.path.join(scenario_folder, lidar_rel_paths[sample_idx])
    lidar_frame[sample_idx] = scipyio.loadmat(lidar_file)['data'][:,0]

angle_lims = [-90,90]
sample_lims = [0,n_samp_first_seq]
plt.figure(figsize=(6,2), dpi=120)
plt.imshow(np.fliplr(np.flipud(lidar_frame)),
           extent=[angle_lims[0], angle_lims[1], sample_lims[0], sample_lims[1]],
           aspect='equal')

plt.xlabel('Angle [º]')
plt.ylabel('Sample index')
    
save_path = "lidar.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()




sample_idx = 20
radar_rel_paths = dataframe['unit1_radar'].values
radar_data = scipyio.loadmat(os.path.join(scenario_folder, radar_rel_paths[sample_idx]))['data']



RADAR_PARAMS = {'chirps':            128, # number of chirps per frame
                'tx':                  1, # transmitter antenna elements
                'rx':                  4, # receiver antenna elements
                'samples':           256, # number of samples per chirp
                'adc_sampling':      5e6, # Sampling rate [Hz]
                'chirp_slope': 15.015e12, # Ramp (freq. sweep) slope [Hz/s]
                'start_freq':       77e9, # [Hz]
                'idle_time':           5, # Pause between ramps [us]
                'ramp_end_time':      60} # Ramp duration [us]

samples_per_chirp = RADAR_PARAMS['samples']
n_chirps_per_frame = RADAR_PARAMS['chirps']
C = 3e8
chirp_period = (RADAR_PARAMS['ramp_end_time'] + RADAR_PARAMS['idle_time']) * 1e-6

RANGE_RES = ((C * RADAR_PARAMS['adc_sampling']) /
                    (2*RADAR_PARAMS['samples'] * RADAR_PARAMS['chirp_slope']))

VEL_RES_KMPH = 3.6 * C / (2 * RADAR_PARAMS['start_freq'] *
                          chirp_period * RADAR_PARAMS['chirps'])

min_range_to_plot = 5
max_range_to_plot = 15 # m
# set range variables
acquired_range = samples_per_chirp * RANGE_RES
first_range_sample = np.ceil(samples_per_chirp * min_range_to_plot /
                            acquired_range).astype(int)
last_range_sample = np.ceil(samples_per_chirp * max_range_to_plot /
                            acquired_range).astype(int)
round_min_range = first_range_sample / samples_per_chirp * acquired_range
round_max_range = last_range_sample / samples_per_chirp * acquired_range

# Range-Velocity Plot
vel = VEL_RES_KMPH * n_chirps_per_frame/2
ang_lim = 75 # comes from array dimensions and frequencies

def minmax(arr):
    return (arr - arr.min())/ (arr.max()-arr.min())

def range_velocity_map(data):
    data = np.fft.fft(data, axis=1) # Range FFT
    # data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, axis=2) # Velocity FFT
    data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis = 0) # Sum over antennas
    data = np.log(1+data)
    return data

def range_angle_map(data, fft_size = 64):
    data = np.fft.fft(data, axis = 1) # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis = 0) # Angle FFT
    data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis = 2) # Sum over velocity
    return data.T

fig, axs = plt.subplots(figsize=(8,6), ncols=2, tight_layout=True)

# # Range-Angle Plot
radar_range_ang_data = range_angle_map(radar_data)[first_range_sample:last_range_sample]
axs[0].imshow(minmax(radar_range_ang_data), aspect='auto',
              extent=[-ang_lim, +ang_lim, round_min_range, round_max_range],
              cmap='seismic', origin='lower')
axs[0].set_xlabel('Angle [°]')
axs[0].set_ylabel('Range [m]')


radar_range_vel_data = range_velocity_map(radar_data)[first_range_sample:last_range_sample]
axs[1].imshow(minmax(radar_range_vel_data), aspect='auto',
              extent=[-vel, +vel, round_min_range, round_max_range],
              cmap='seismic', origin='lower')
axs[1].set_xlabel('Velocity [km/h]')
axs[1].set_ylabel('Range [m]')
    
save_path = "doppler.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()


