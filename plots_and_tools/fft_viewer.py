import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import time

# ------------------------
# Configuration
# ------------------------
COM_PORT = '/dev/ttyACM0'   
BAUD_RATE = 115200
BUFFER_SIZE = 1024          
UPDATE_INTERVAL = 50        

# ------------------------
# Setup
# ------------------------
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
signal_buffer = deque(maxlen=BUFFER_SIZE)  # full ADC value
lsb_buffer = deque(maxlen=BUFFER_SIZE)     # 4 LSB combined as integer
lsb_bits_buffer = [deque(maxlen=BUFFER_SIZE) for _ in range(4)]  # per LSB
timestamps = deque(maxlen=BUFFER_SIZE)
total_samples = 0

fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(9, 6))

# Time domain plot
time_line, = ax_time.plot([], [], color='tab:blue')
ax_time.set_xlim(0, BUFFER_SIZE)
ax_time.set_ylim(0, 4095)
ax_time.set_title("Raw ADC Signal (Time Domain)")
ax_time.set_xlabel("Sample index")
ax_time.set_ylabel("ADC value")

# Frequency domain plot
freq_lines = [ax_freq.plot([], [], label=f"Bit{i}")[0] for i in range(4)]
combined_line, = ax_freq.plot([], [], color='tab:orange', lw=2, label='4 LSB combined')
ax_freq.set_xlim(0, 1)  # placeholder
ax_freq.set_ylim(0, 1)
ax_freq.set_title("FFT Magnitude (Frequency Domain)")
ax_freq.set_xlabel("Frequency [Hz]")
ax_freq.set_ylabel("Normalized magnitude")
ax_freq.legend()

text_total = ax_time.text(0.7, 0.9, "Samples: 0", transform=ax_time.transAxes)
text_rate = ax_time.text(0.7, 0.85, "Sample rate: 0 Hz", transform=ax_time.transAxes)

def update(frame):
    global total_samples

    try:
        line_bytes = ser.readline()
        if not line_bytes:
            return [time_line, combined_line] + freq_lines + [text_total, text_rate]

        line_str = line_bytes.decode('utf-8', errors='ignore').strip()

        adc_value = int(line_str)
        signal_buffer.append(adc_value)
        timestamps.append(time.time())
        total_samples += 1

        # Extract 4 LSBs
        lsb_bits = [(adc_value >> i) & 1 for i in range(4)]
        for i, bit in enumerate(lsb_bits):
            lsb_bits_buffer[i].append(bit*2 - 1)  # map 0->-1, 1->1
        lsb_value = sum(bit << i for i, bit in enumerate(lsb_bits))
        lsb_buffer.append(lsb_value)

        # Update time domain
        time_data = np.array(signal_buffer)
        time_line.set_data(np.arange(len(time_data)), time_data)
        text_total.set_text(f"Samples: {total_samples}")

        # Compute sample rate
        if len(timestamps) > 1:
            Ts = np.mean(np.diff(np.array(timestamps)))
            sample_rate = 1 / Ts
        else:
            sample_rate = 0
        text_rate.set_text(f"Sample rate: {sample_rate:.1f} Hz")

        # FFTs
        if len(signal_buffer) >= BUFFER_SIZE and sample_rate > 0:
            freqs = fftfreq(BUFFER_SIZE, d=1/sample_rate)[:BUFFER_SIZE//2]

            # Per-bit FFT
            for i in range(4):
                bit_array = np.array(lsb_bits_buffer[i])
                fft_vals = fft(bit_array)
                fft_mag = np.abs(fft_vals[:BUFFER_SIZE//2])
                fft_mag /= np.max(fft_mag) if np.max(fft_mag) != 0 else 1
                freq_lines[i].set_data(freqs, fft_mag)
                # Spectral entropy
                spec_entropy = entropy(fft_mag/np.sum(fft_mag))
                print(f"Bit{i} spectral entropy:", spec_entropy)

            # Combined 4 LSBs FFT
            combined_array = np.array(lsb_buffer)
            fft_vals = fft(combined_array - np.mean(combined_array))
            fft_mag = np.abs(fft_vals[:BUFFER_SIZE//2])
            fft_mag /= np.max(fft_mag) if np.max(fft_mag) != 0 else 1
            combined_line.set_data(freqs, fft_mag)
            combined_entropy = entropy(fft_mag/np.sum(fft_mag))
            print("Combined 4 LSB spectral entropy:", combined_entropy)

            ax_freq.set_xlim(0, sample_rate/2)
            ax_freq.set_ylim(0, 1.1)

    except Exception as e:
        print("Error:", e)
        return [time_line, combined_line] + freq_lines + [text_total, text_rate]

    return [time_line, combined_line] + freq_lines + [text_total, text_rate]

# ------------------------
# Animation 
# ------------------------
ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, cache_frame_data=False)
plt.tight_layout()
plt.show()
