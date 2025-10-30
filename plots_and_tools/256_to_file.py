import serial
import numpy as np

# ------------------------
# Configuration
# ------------------------
COM_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SAMPLES_PER_BLOCK = 64  # 64 samples × 4 LSBs = 256 bits
OUTPUT_FILE = "256bit_blocks.bin"

# ------------------------
# Setup
# ------------------------
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
block = []

# Open file for writing
with open(OUTPUT_FILE, "wb") as fout:
    print("Collecting 256-bit blocks... Press Ctrl+C to stop.")
    try:
        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line_str = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line_str:
                continue

            try:
                adc_value = int(line_str)
                # Append the 4 LSBs as bits mapped 0/1
                lsb_bits = [(adc_value >> i) & 1 for i in range(4)]
                block.extend(lsb_bits)

                # Once we have 256 bits, write them to file
                if len(block) >= 256:
                    # Convert bits to bytes
                    byte_vals = []
                    for i in range(0, 256, 8):
                        byte = 0
                        for j in range(8):
                            byte = (byte << 1) | block[i + j]
                        byte_vals.append(byte)
                    fout.write(bytes(byte_vals))
                    fout.flush()  # ensure it's written
                    block = []  # reset for next block
                    print("Wrote 256-bit block to file.")

            except ValueError:
                continue

    except KeyboardInterrupt:
        print("Stopped by user.")
import serial
import numpy as np

# ------------------------
# Configuration
# ------------------------
COM_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SAMPLES_PER_BLOCK = 64  # 64 samples × 4 LSBs = 256 bits
OUTPUT_FILE = "256bit_blocks.bin"

# ------------------------
# Setup
# ------------------------
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
block = []

# Open file for writing
with open(OUTPUT_FILE, "wb") as fout:
    print("Collecting 256-bit blocks... Press Ctrl+C to stop.")
    try:
        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line_str = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line_str:
                continue

            try:
                adc_value = int(line_str)
                # Append the 4 LSBs as bits mapped 0/1
                lsb_bits = [(adc_value >> i) & 1 for i in range(4)]
                block.extend(lsb_bits)

                # Once we have 256 bits, write them to file
                if len(block) >= 256:
                    # Convert bits to bytes
                    byte_vals = []
                    for i in range(0, 256, 8):
                        byte = 0
                        for j in range(8):
                            byte = (byte << 1) | block[i + j]
                        byte_vals.append(byte)
                    fout.write(bytes(byte_vals))
                    fout.flush()  # ensure it's written
                    block = []  # reset for next block
                    print("Wrote 256-bit block to file.")

            except ValueError:
                continue

    except KeyboardInterrupt:
        print("Stopped by user.")
import serial
import numpy as np

# ------------------------
# Configuration
# ------------------------
COM_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SAMPLES_PER_BLOCK = 64  # 64 samples × 4 LSBs = 256 bits
OUTPUT_FILE = "256bit_blocks.bin"

# ------------------------
# Setup
# ------------------------
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
block = []

# Open file for writing
with open(OUTPUT_FILE, "wb") as fout:
    print("Collecting 256-bit blocks... Press Ctrl+C to stop.")
    try:
        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line_str = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line_str:
                continue

            try:
                adc_value = int(line_str)
                # Append the 4 LSBs as bits mapped 0/1
                lsb_bits = [(adc_value >> i) & 1 for i in range(4)]
                block.extend(lsb_bits)

                # Once we have 256 bits, write them to file
                if len(block) >= 256:
                    # Convert bits to bytes
                    byte_vals = []
                    for i in range(0, 256, 8):
                        byte = 0
                        for j in range(8):
                            byte = (byte << 1) | block[i + j]
                        byte_vals.append(byte)
                    fout.write(bytes(byte_vals))
                    fout.flush()  # ensure it's written
                    block = []  # reset for next block
                    print("Wrote 256-bit block to file.")

            except ValueError:
                continue

    except KeyboardInterrupt:
        print("Stopped by user.")
