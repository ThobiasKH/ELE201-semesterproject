import serial
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ----------------------
# Configuration
# ----------------------
COM_PORT = '/dev/ttyACM0'   # Change to your port (Windows: COM7)
BAUD_RATE = 115200
BUFFER_SIZE = 100            # Number of points to display

# ----------------------
# Set up serial connection
# ----------------------
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

# ----------------------
# Data storage
# ----------------------
data = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)  # Circular buffer

# ----------------------
# Set up plot
# ----------------------
plt.style.use('ggplot')       # simple and safe
fig, ax = plt.subplots()
line, = ax.plot(range(BUFFER_SIZE), data)
ax.set_ylim(0, 4095)  # 12-bit ADC
ax.set_xlabel('Samples')
ax.set_ylabel('ADC Value')
ax.set_title('Live ADC Readings')

# ----------------------
# Update function
# ----------------------
def update(frame):
    try:
        line_bytes = ser.readline()
        if line_bytes:
            line_str = line_bytes.decode('utf-8').strip()
            # Expect format: "ADC raw = 1234"
            if line_str.startswith("ADC raw"):
                value = int(line_str.split('=')[1].strip())
                data.append(value)
                line.set_ydata(data)
    except Exception as e:
        print("Error:", e)
    return line,

# ----------------------
# Animate
# ----------------------
ani = FuncAnimation(fig, update, interval=100)
plt.show()
