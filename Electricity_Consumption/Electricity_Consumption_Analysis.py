import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy import signal
from scipy import stats
from scipy.optimize import minimize
from scipy.integrate import trapezoid
from scipy.fft import fft, fftfreq

# 720 samples in month (unit: day)
t = np.linspace(0, 30, 720)

# power consumption signal
consumption_signal = 5*np.sin(2*np.pi*1.2*t)

# making noise
noise = np.random.normal(0, 2, len(t))
electricity_signal = consumption_signal + noise

# create missing data
mask = np.random.choice([0, 1], size=len(t), p=[0.95, 0.05])
electricity_signal[mask == 1] = np.nan

# save data
os.chdir("D://python/SciPy_projects/Electricity_Consumption")
data = pd.DataFrame({"time": t, "signal": electricity_signal})
data.to_csv("electricity_data.csv", index=False)

# delete NaN
mask = ~np.isnan(data["signal"])

x = data["time"][mask]
y = data["signal"][mask]

# creating a function and filling in the blanks
f = interp1d(x, y, kind="cubic", fill_value="extrapolate")

data["signal_filled"] = f(data["time"])

# signal filtering
b, a = signal.butter(1, 10, fs=720, btype="low")
data["signal_clean"] = signal.savgol_filter(
    signal.filtfilt(b, a, data["signal_filled"]),
    window_length=31,
    polyorder=3)

# draw the signal
plt.plot(data["time"], data["signal_filled"], label="Filled")
plt.plot(data["time"], data["signal"], label="Raw")
plt.plot(data["time"], data["signal_clean"], label="Clean")

plt.legend()
plt.xlabel("Time (day)")
plt.ylabel("Electricity Consumption")
plt.title("Power Consumption Preprocessing Signal")

plt.show()

# checking the normality of the data
stat_sh, p_sh = stats.shapiro(data["signal_clean"])
if p_sh > 0.05:
    print("The data is normal.\n")
else:
    print("The data isn't normal\n")


# checking the difference in consumption in the two halves of the month
def ttest(col):
    number_index = int(len(col)/2)
    first_half, second_half = col[:number_index], col[number_index:]
    stat_ttest, p_ttest = stats.ttest_ind(first_half, second_half)

    if np.abs(stat_ttest) > 1 and p_ttest < 0.05:
        print("Electricity consumption is different at the beginning "
              "and end of the month.\n")
    else:
        print("Electricity consumption does not differ at the beginning "
              "and end of the month.\n")


ttest(data["signal_clean"])

# algebraic calculations
print("Data properties:\n",
      f"Mean:{stats.tmean(data['signal_clean'])}\n",
      f"Std:{stats.tstd(data['signal_clean'])}\n",
      f"Var:{stats.tvar(data['signal_clean'])}\n")

input("")

# histogram chart
sns.histplot(data["signal_filled"], bins=30)
plt.grid(axis="y")
plt.show()


# build function
def model(x, params):
    a, b, c, d, e, f = params
    return a*x + b*np.sin(c*x) + d*np.cos(e*x) + f


def loss(params):
    y_pred = model(data["time"], params)
    error = np.mean((y_pred - data["signal_clean"])**2)
    return error


x0 = np.array([0.1, 5, 1, 2, 1, 0])
result = minimize(loss, x0, method="L-BFGS-B")
data["model_prediction"] = model(data["time"], result.x)

plt.plot(data["time"], data["model_prediction"], label="Predicted")
plt.plot(data["time"], data["signal_clean"], label="Original")

plt.xlabel("Time (day)")
plt.ylabel("Electricity Consumption")
plt.title("Comparison of forecast and actual value")

plt.legend()
plt.show()

pec = trapezoid(data["model_prediction"], data["time"])
mpc = trapezoid(data["signal_clean"], data["time"])
print(f"Model error: {np.abs(pec - mpc)/ mpc*100}%\n")

input("")

# error analysis
error = data["model_prediction"] - data["signal_clean"]

plt.subplot(1, 2, 1)
plt.plot(data["time"], error, "--")
plt.title("Prediction Error Over Time")
plt.xlabel("Time")
plt.ylabel("Error")

plt.subplot(1, 2, 2)
sns.histplot(error, bins=30)
plt.title("Error Distribution")

plt.show()

# frequency analysis
n = len(data)
dt = t[1] - t[0]

yf = fft(data["signal_clean"])
xf = fftfreq(n, dt)

plt.plot(xf[:n//2], np.abs(yf[:n//2]))
plt.title("Frequency Spectrum")
plt.show()

# forecast for the next 10 days
future_t = np.linspace(30, 40, 240)
future_pred = model(future_t, result.x)

plt.plot(data["time"], data["signal_clean"])
plt.plot(future_t, future_pred, "--")
plt.title("forecast for the next 10 days")
plt.show()
