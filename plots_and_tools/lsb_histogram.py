import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ------------------------
# Configuration
# ------------------------
COM_PORT = '/dev/ttyACM0'  
BAUD_RATE = 115200
BUFFER_SIZE = 500  

# ------------------------
# Setup
# ------------------------
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

data_bits = deque(maxlen=BUFFER_SIZE)
total_samples = 0

fig, ax = plt.subplots()
bit_labels = ['Bit0', 'Bit1', 'Bit2', 'Bit3']

bars = ax.bar(range(8), [0]*8, color=['skyblue','orange']*4)
ax.set_xticks(range(8))
ax.set_xticklabels([f"{b} = 0" if i%2==0 else f"{b} = 1" for i,b in enumerate(bit_labels*2)])
ax.set_ylim(0, BUFFER_SIZE)
ax.set_ylabel("Count")
ax.set_title("Histogram of bottom 4 ADC bits in raw signal")

text_total = ax.text(0.7, 0.95, "Total samples: 0", transform=ax.transAxes)

def update(frame):
    global total_samples
    try:
        line_bytes = ser.readline()
        if not line_bytes:
            return bars

        line_str = line_bytes.decode('utf-8', errors='ignore').strip()
        if line_str.startswith("ADC raw"):
            adc_value = int(line_str.split('=')[1].strip())
            lsb_bits = [(adc_value >> i) & 1 for i in range(4)]  # Bit0..Bit3
            data_bits.append(lsb_bits)
            total_samples += 1

            counts = []
            for i in range(4):
                bit_column = [sample[i] for sample in data_bits]
                count_0 = bit_column.count(0)
                count_1 = bit_column.count(1)
                counts.extend([count_0, count_1])

            for rect, count in zip(bars, counts):
                rect.set_height(count)

            text_total.set_text(f"Total samples: {total_samples}")

    except Exception as e:
        print("Error:", e)
    return bars

# ------------------------
# Nice Animation
# ------------------------
ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()
