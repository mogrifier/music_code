import cupy as cp
import numpy as np
import xarray as xr
from scipy.io import wavfile
import matplotlib.pyplot as plt

# data from https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html

# -------------------------------
# Helper functions
# -------------------------------

def scale_whole_series(series, target_amp=0.9):
    """Scale the entire series into [-target_amp, target_amp] before chopping."""
    series = np.nan_to_num(series).astype(np.float32)
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return cp.zeros_like(series, dtype=cp.float32)
    scaled = (series - min_val) / (max_val - min_val)  # 0..1
    scaled = 2 * scaled - 1                            # -1..+1
    return cp.asarray(scaled * target_amp)

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

def write_wavetable_float32(waves_cp, filename="sst_wavetable.wav", sample_rate=44100):
    """Write concatenated frames to float32 WAV."""
    wavetable = waves_cp.reshape(-1).get().astype(np.float32)
    if wavetable.size == 0:
        raise ValueError("Wavetable is empty, nothing to write.")
    wavfile.write(filename, sample_rate, wavetable)
    print(f"Wrote {filename}, samples: {wavetable.shape[0]}")
    print("Output min/max:", wavetable.min(), wavetable.max(),
          "mean:", wavetable.mean(), "var:", wavetable.var())

    # Diagnostic plot of first few frames
    plt.figure(figsize=(10, 6))
    for i in range(min(5, waves_cp.shape[0])):
        plt.plot(waves_cp[i].get(), label=f"Frame {i}")
    plt.title("First few frames (scaled before chop)")
    plt.legend()
    plt.show()

# -------------------------------
# Main pipeline
# -------------------------------

def build_wavetable(day_wave_cp, segment_len=512, num_select=64, filename="sst_wavetable.wav"):
    # Scale entire series first
    scaled_series = scale_whole_series(day_wave_cp)
    # Chop into frames
    waves = chop(scaled_series, segment_len)
    # Select evenly spaced frames
    waves_sel = select_even(waves, num_select)
    # Write directly
    write_wavetable_float32(waves_sel, filename=filename)
    return waves_sel

# -------------------------------
# Entry point
# -------------------------------

def main():
    ds = xr.open_dataset("sst.day.mean.2024.nc")

    # Mask fill values (-999.9) as NaN, then drop them
    sst_np = ds["sst"].values
    sst_np = np.where(sst_np < -100, np.nan, sst_np)

    # Example: pick a lat/lon box over the ocean
    lat_min, lat_max = -50, 60
    lon_min, lon_max = 185, 230
    subset = ds["sst"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    day_box = subset.isel(time=100).values

    # Drop NaNs completely
    valid = day_box[~np.isnan(day_box)]
    day_wave_cp = cp.asarray(valid)

    print("Day slice stats:",
          "min:", np.nanmin(day_box),
          "max:", np.nanmax(day_box),
          "var:", np.nanvar(day_box),
          "NaNs:", np.sum(np.isnan(day_box)),
          "valid samples:", valid.size)

    # Build wavetable with 64 evenly spaced frames
    waves_sel = build_wavetable(day_wave_cp,
                                segment_len=512,
                                num_select=64,
                                filename="sst_wavetable_64scaled.wav")

    print("Final wavetable shape:", waves_sel.shape)

if __name__ == "__main__":
    main()
