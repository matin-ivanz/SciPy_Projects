import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq
from scipy import stats
from scipy.optimize import minimize

# 600 samples per minute (unit: second)
t = np.linspace(0, 60, 600)

# baseline heart rate (around 70bpm)
heart_rate = 75 + 5*np.sin(2*np.pi*1.2*t)

# making noise
noise = np.random.normal(0, 2, len(t))
signal = heart_rate + noise

# create missing data
mask = np.random.choice([0, 1], size=len(t), p=[0.95, 0.05])
signal[mask == 1] = np.nan

# save data
os.chdir("D://python/SciPy_projects/Heart_Rate")
data = pd.DataFrame({"time": t, "heart_rate": signal})
data.to_csv("heart_rate_data.csv", index=False)

# reading data
print(data.head(), "\n")
print(data.isnull().sum(), "\n")

# delete NaN
mask = ~np.isnan(data["heart_rate"])

x = data["time"][mask]
y = data["heart_rate"][mask]

# build function
f = interp1d(x, y, kind="cubic", fill_value="extrapolate")

data["heart_rate_filled"] = f(data["time"])

# filtering
filtered = savgol_filter(data["heart_rate_filled"], window_length=31,
                         polyorder=3)
data["heart_rate_clean"] = filtered

# draw a comparison
plt.plot(data["time"], data["heart_rate_filled"], label="Filled")
plt.plot(data["time"], data["heart_rate"], label="Raw")
plt.plot(data["time"], data["heart_rate_clean"], label="Clean")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Heart Rate")
plt.title("Preprocessing Heart Signal")

plt.show()

signal = data["heart_rate_clean"].values
time = data["time"].values

# sampling interval
dt = time[1] - time[0]
# sampling frequency
fs = 1 / dt

N = len(signal)
yf = fft(signal)
xf = fftfreq(N, dt)

idx = xf > 0
freqs = xf[idx]
power = np.abs(yf[idx])

plt.plot(freqs, power)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Frequency Spectrum (FFT)")
plt.show()

main_freq = freqs[np.argmax(power)]
bpm = main_freq * 60

print("Main Frequency:", main_freq)
print("Estimated BPM:", bpm, "\n")

hr = data["heart_rate_clean"].values

mean = np.mean(hr)
std = np.std(hr)
median = np.median(hr)

print("Mean:", mean)
print("Std:", std)
print("Median:", median, "\n")

stat, p = stats.shapiro(hr)

print("p-value:", p)

if p > 0.05:
    print("Data is Normal.")
else:
    print("Data is not Normal.")

z_scores = stats.zscore(hr)

outliers = np.where(np.abs(z_scores) > 3)

print("Outliers count:", len(outliers[0]), "\n")

plt.plot(data["time"], hr, label="Heart Rate")
plt.scatter(data["time"].iloc[outliers], hr[outliers],
            color="red", label="Anomaly")
plt.legend()
plt.title("Anomaly Detection")
plt.show()

rest = np.ones(len(hr)) * 70
t, p = stats.ttest_ind(hr, rest)
print("p-value:", p)

if p < 0.05:
    print("Different from rest.\n")
else:
    print("Normal vs Rest.\n")

raw = data["heart_rate_filled"].values


def loss(params):
    window = int(params[0])

    # being an individual
    if window % 2 == 0:
        window += 1

    if window < 5 or window > 101:
        return 1e6

    # limitation
    filtered = savgol_filter(raw, window, 3)

    # difference with raw signal
    error = np.mean((filtered - raw)**2)
    return error


result = minimize(loss, x0=[21], bounds=[(5, 101)])
best_window = int(result.x[0])
print("Best window:", best_window)

best_filtered = savgol_filter(raw, best_window, 3)
data["optimized_signal"] = best_filtered

plt.plot(data["time"], raw, label="Filled")
plt.plot(data["time"], best_filtered, label="Optimized")
plt.legend()
plt.title("Find Optimized Signal")
plt.show()
