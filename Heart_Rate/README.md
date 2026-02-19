# Heart Rate

## Overview
- In this project, a person's heart signals are analyzed and visualized for one minute.

## Objective
- The goal of this project is to learn various SciPy methods and capabilities and increase your skills working in Python and other libraries.

## Key Skills Demonstrated
- Working with os, numpy, matplotlib, pandas libraries
- Use of interpolate, signal, stats, optimize, fft

## Feature Engineering
Create features from the heartbeat signal in the following order:
- time
- heart_rate
- heart_rate_filled
- heart_rate_clean
- optimized_signal

## Visualizations
- From the line graph to show the difference between the original signal, the filled and filtered signal.
- Next, a graph of the frequency spectrum of abnormal heart rate points (with a scatter plot) and a prediction graph are drawn.

## Conclusion
- The signal is randomly generated, contains noise and empty values, and is saved in a file in (csv) format.
- Empty values ​​are filled and noise is eliminated.
- The normality of the data is checked.
- The heart rate frequency spectrum is examined.
- The original and estimated frequency (BPM) is printed.
- The mean, standard deviation, and median of the heart rate are printed.
- The normality of the data is checked and the number of abnormal heart rate points is checked.
- The difference in heart rate frequency between resting and normal states is examined.
- We optimize the signal by constructing a function.