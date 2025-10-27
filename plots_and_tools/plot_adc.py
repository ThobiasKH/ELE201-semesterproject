import serial
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ----------------------
# Configuration
# ----------------------
COM_PORT = '/dev/ttyACM0'   
BAUD_RATE = 115200
BUFFER_SIZE = 100            

# ----------------------
# Serial connection
# ----------------------
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

# ----------------------
# Data storage
# ----------------------
data = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE) # Lovely circular buffer 

# ----------------------
# Plot
# ----------------------
plt.style.use('ggplot')       
fig, ax = plt.subplots()
line, = ax.plot(range(BUFFER_SIZE), data)
ax.set_ylim(900, 1100) # Remember to change if we're not sampling 12-bits (prob won't)  
ax.set_xlabel('Samples')
ax.set_ylabel('ADC Value')
ax.set_title('Live ADC Readings')

def update(frame):
    try:
        line_bytes = ser.readline()
        if line_bytes:
            line_str = line_bytes.decode('utf-8').strip()
            if line_str.startswith("ADC raw"): # NOTE TO SELF: Probably more efficient to send data w/o "ADC raw" 
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
