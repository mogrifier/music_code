import cupy as cp 
import numpy as np
import xarray as xr
from scipy.io import wavfile
import matplotlib.pyplot as plt



'''
Make a wavetable from the time derivative of SST data. There are 64 waves in each wavetable and each wave is 512 samples long.
This will choose the day with the maximum variance in the wavetable after scaling by standard deviation and applying a soft limiter.
'''

# data from https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html

# -------------------------------
# Helper functions
# -------------------------------

def scale_by_std_soft(series, target_amp=0.9):
    """Normalize series by std, apply tanh soft limiter, scale to target_amp."""
    series = np.nan_to_num(series).astype(np.float32)
    mean_val = np.mean(series)
    std_val = np.std(series)
    if std_val == 0:
        return cp.zeros_like(series, dtype=cp.float32)
    normalized = (series - mean_val) / std_val
    limited = np.tanh(normalized)  # smooth saturation instead of hard clipping
    '''
    using amplification only results in a usable file but the level is very low so I have to boost it 
    in the synth.
    '''
    return cp.asarray(limited * target_amp)

def chop(wave_cp, segment_len=512):
    """Chop a 1D CuPy array into fixed-length segments."""
    total_len = wave_cp.size
    num_segments = total_len // segment_len
    trimmed = wave_cp[:num_segments * segment_len]
    return trimmed.reshape(num_segments, segment_len)

def select_even(waves_cp, num_select):
    """Select evenly spaced frames across the dataset."""
    num_segments = waves_cp.shape[0]
    if num_segments == 0:
        raise ValueError("No frames available for selection.")
    if num_select >= num_segments:
        return waves_cp
    idx = cp.linspace(0, num_segments - 1, num_select).astype(cp.int32)
    return waves_cp[idx]


def get_wavetable_variance(waves_cp):
    """Compute variance of concatenated frames."""
    wavetable = waves_cp.reshape(-1).get().astype(np.float32)
    if wavetable.size == 0:
        raise ValueError("Wavetable is empty, cannot compute variance.")
    return wavetable.var()

def write_wavetable_float32(waves_cp, filename="sst_derivative_soft.wav", sample_rate=44100):
    """Write concatenated frames to float32 WAV."""
    wavetable = waves_cp.reshape(-1).get().astype(np.float32)
    if wavetable.size == 0:
        raise ValueError("Wavetable is empty, nothing to write.")
    wavfile.write(filename, sample_rate, wavetable)
    print(f"Wrote {filename}, samples: {wavetable.shape[0]}")
    print("Output min/max:", wavetable.min(), wavetable.max(),
          "mean:", wavetable.mean(), "var:", wavetable.var())

    # Diagnostic plot of first few frames
    '''
    plt.figure(figsize=(10, 6))
    for i in range(min(5, waves_cp.shape[0])):
        plt.plot(waves_cp[i].get(), label=f"Frame {i}")
    plt.title("First few derivative frames (soft limited)")
    plt.legend()
    plt.show()
    '''
    
# -------------------------------
# Main pipeline
# -------------------------------

def build_derivative_wavetable(deriv_series, segment_len=512, num_select=64, filename="sst_derivative_soft.wav"):
    scaled_series = scale_by_std_soft(deriv_series)   # <-- std normalization + soft limiter
    
    waves = chop(scaled_series, segment_len)
    waves_sel = select_even(waves, num_select)
    write_wavetable_float32(waves_sel, filename=filename)
    return waves_sel

# -------------------------------
# Entry point
# -------------------------------

def main():
    ds = xr.open_dataset("sst.day.mean.2024.nc")

    # Mask fill values (-999.9) as NaN
    sst = ds["sst"].where(ds["sst"] > -100)

    # Compute derivatives along time (best oscillations)
    dtime = sst.differentiate("time")

    # Example: pick a lat/lon box and scan one day
    lat_min, lat_max = -50, 60
    lon_min, lon_max = 185, 230
    subset_dtime = dtime.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    day_idx = 20
    dtime_day = subset_dtime.isel(time=day_idx).values.flatten()

    max_variance = 0
    bestday = 0
    # Build wavetable with soft limiting- I want the max variance table
    for i in range(366):
        dtime_day = subset_dtime.isel(time=i).values.flatten()
        scaled_series = scale_by_std_soft(dtime_day)   # <-- std normalization + soft limiter

        waves = chop(scaled_series, segment_len=512)
        waves_sel = select_even(waves, num_select=64)
        variance = get_wavetable_variance(waves_sel)
        if (variance > max_variance):
            max_variance = variance
            bestday = i

    
    print("Best day index for max variance:", bestday, "Variance:", max_variance)
    # Now build the wavetable for the best day
    dtime_day = subset_dtime.isel(time=bestday).values.flatten()
    build_derivative_wavetable(dtime_day, segment_len=512, num_select=64, filename="sst_derivative_soft_bestday.wav")

if __name__ == "__main__":
    main()
