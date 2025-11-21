#Data files are NetCDF. Read with python.
import cupy as cp
import numpy as np
from netCDF4 import Dataset
from scipy.io import wavfile

# data from https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html

def read_netcdf_to_wavetable(filename):
    ds = Dataset(filename, "r")

    # Read variables into GPU arrays
    time = cp.array(ds['time'][:])
    lat  = cp.array(ds['lat'][:])
    lon  = cp.array(ds['lon'][:])

    # Load SST as NumPy, then move to GPU
    sst_np = ds['sst'][:]   # still a masked array or ndarray

    # Move to GPU and replace fill values with NaN
    sst_cp = cp.array(sst_np, dtype=cp.float32)
    sst_cp = cp.where(sst_cp <= -9.99, cp.nan, sst_cp)

    # Example target coordinates

    #dataset uses 0–360 for longitude, convert:

    # Extract SST value- day, lat, lon
    sst_value = sst_cp[15, -20, 220]

    print(sst_value)


    '''
    My intent is to create a bounding box around an area of ocean, then loop through all lat/lon points in that box.
    There is no need to test if over land or not, since the box is all ocean. There may be some islands, so I 
    still have to text for invalid values in the sst data. And negative number for sst > -9.99 is invalid. This will
    give a good number of data points to work with. Rough area is the pacfic, with latitude from 122w to 178w and
    longitude from 60s to 55 n. Just over 100,000 data points. 400KB of data. Theres 86,400 seconds in a day, so this is
    is more than enough data for a piece of music hours long.

    What should I do with the data? It should be smooth so I could make it an LFO or VCO. Or I could use it as a wavetable.
    '''

    # Pick one day (e.g., day 100)
    day_wave = sst_cp[100, :, :].flatten()

    # Normalize to [-1, 1] 
    # Chop, normalize, select 64 waves
    waves_np = chop_forwardfill_normalize_select(day_wave, segment_len=512, num_select=128)

    # Write concatenated wavetable file
    write_wavetable_float32(waves_np, "sst_wavetable128.wav", sample_rate=44100)

def forward_fill_nan_cpu(waves_cp):
    """
    Forward-fill NaNs along axis=1 (each wave independently).
    Uses NumPy since CuPy lacks maximum.accumulate.
    """
    waves = waves_cp.get()  # back to NumPy
    isnan = np.isnan(waves)
    idx = np.arange(waves.shape[1])
    valid = np.where(~isnan, idx, 0)
    last_valid = np.maximum.accumulate(valid, axis=1)
    waves_ff = waves[np.arange(waves.shape[0])[:, None], last_valid]
    return cp.asarray(waves_ff)  # back to GPU

def chop_forwardfill_normalize_select(wave_cp, segment_len=512, num_select=64):
    """
    Chop a 1D CuPy array into fixed-length segments,
    forward-fill NaNs, normalize each wave, and select a subset.
    """
    total_len = wave_cp.size
    num_segments = total_len // segment_len

    # Trim to multiple of segment_len
    trimmed = wave_cp[:num_segments * segment_len]

    # Reshape into (num_segments, segment_len)
    waves = trimmed.reshape(num_segments, segment_len)

    # Forward-fill NaNs (CPU side)
    waves_ff = forward_fill_nan_cpu(waves)

    # Normalize per wave (vectorized, GPU side)
    means = cp.mean(waves_ff, axis=1, keepdims=True)              # (num_segments, 1)
    centered = waves_ff - means                                   # (num_segments, 512)
    max_vals = cp.max(cp.abs(centered), axis=1, keepdims=True)    # (num_segments, 1)
    waves_norm = cp.where(max_vals > 0, centered / max_vals, centered)

    # Down-select evenly spaced waves
    if num_select >= num_segments:
        selected = waves_norm
    else:
        indices = cp.linspace(0, num_segments - 1, num_select).astype(cp.int32)
        selected = waves_norm[indices]

    return selected.get().astype(np.float32)  # back to NumPy for writing

def write_wavetable_float32(waves_np, filename="sst_wavetable.wav", sample_rate=44100):
    """
    Concatenate selected waves into one long float32 WAV file.
    """
    wavetable = waves_np.reshape(-1).astype(np.float32)
    wavfile.write(filename, sample_rate, wavetable)


def main():
    filename = "sst.day.mean.2024.nc"
    read_netcdf_to_wavetable(filename)

    # This block ensures main() runs when you execute the script directly
if __name__ == "__main__":
    main()