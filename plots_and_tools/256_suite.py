import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import entropy

# ------------------------
# Configuration
# ------------------------
COM_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SAMPLES_PER_BLOCK = 64  # 64 samples × 4 LSBs = 256 bits
BUFFER_BLOCKS = 50      # Number of blocks to keep in plot

# ------------------------
# Setup
# ------------------------
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

# Buffers for plotting
entropy_buffer = []
bitfreq_buffer = []
autocorr_buffer = []
runs_buffer = []
numSamples = 0

block = []

# ------------------------
# Analysis functions
# ------------------------
def extract_4lsb(adc_value):
    return [(adc_value >> i) & 1 for i in range(4)]

def compute_shannon_entropy(bits):
    counts = np.bincount(bits, minlength=2)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

def autocorrelation(bits):
    bits = np.array(bits)
    if len(bits) < 2:
        return 0
    return np.corrcoef(bits[:-1], bits[1:])[0, 1]

def count_runs(bits):
    bits = np.array(bits)
    if len(bits) < 2:
        return 1
    return np.diff(bits).astype(bool).sum() + 1

def process_block(block):
    all_bits = []
    for sample in block:
        all_bits.extend(extract_4lsb(sample))
    all_bits = np.array(all_bits)

    sh_entropy = compute_shannon_entropy(all_bits)
    bit_freq = np.mean(all_bits)
    auto_corr = autocorrelation(all_bits)
    runs = count_runs(all_bits)

    return sh_entropy, bit_freq, auto_corr, runs

# ------------------------
# Setup plots
# ------------------------
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axs[0].set_ylabel("Entropy")
axs[1].set_ylabel("Fraction of 1s")
axs[2].set_ylabel("Autocorr (lag1)")
axs[3].set_ylabel("Runs")
axs[3].set_xlabel("Block index")

lines = [axs[i].plot([], [])[0] for i in range(4)]

def update(frame):
    global block, entropy_buffer, bitfreq_buffer, autocorr_buffer, runs_buffer, numSamples

    # Read new sample
    line_bytes = ser.readline()
    if not line_bytes:
        return lines
    line_str = line_bytes.decode('utf-8', errors='ignore').strip()
    parts = [p for p in line_str.replace('\r','\n').split('\n') if p]

    for part in parts:
        try:
            adc_value = int(part.strip())
            block.append(adc_value)

            # Process block if full
            if len(block) >= SAMPLES_PER_BLOCK:
                numSamples += 1
                sh_entropy, bit_freq, auto_corr, runs = process_block(block)
                entropy_buffer.append(sh_entropy)
                bitfreq_buffer.append(bit_freq)
                autocorr_buffer.append(auto_corr)
                runs_buffer.append(runs)


                block.clear()  # reset block for next batch

                # Update plot lines
                lines[0].set_data(range(len(entropy_buffer)), entropy_buffer)
                lines[1].set_data(range(len(bitfreq_buffer)), bitfreq_buffer)
                lines[2].set_data(range(len(autocorr_buffer)), autocorr_buffer)
                lines[3].set_data(range(len(runs_buffer)), runs_buffer)

                print(numSamples)
                print(f"Average entropy: {sum(entropy_buffer)/numSamples}")
                print(f"Average bitfreq: {sum(bitfreq_buffer)/numSamples}")
                print(f"Average autocorr: {sum(autocorr_buffer)/numSamples}")
                print(f"Average runs: {sum(runs_buffer)/numSamples}")

                for i, ax in enumerate(axs):
                    ax.relim()
                    ax.autoscale_view()

        except ValueError:
            # Ignore parts that aren't integers
            continue

    return lines

ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.tight_layout()
plt.show()
