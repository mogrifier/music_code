I am using data from https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html to make wavetables. 

The output of the pipeline is a proper wavefile made of 32-bit floats. The table has 64 waves of length 512 samples.

The time-based derivative of the sea surface temperature data makes a good wavetable. The code derivative_sst.py will find the day with the max variance and write the table for it. Max variance simply means the data is the most spread out. Sonically, this should sound the most interesting.
