import serial
import time
import numpy as np
import sys
import argparse
from collections import defaultdict

# --- CONFIGURATION (Consistent with your Pi Estimator) ---
COM_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
BLOCK_SIZE_BITS = 256
BITS_PER_SAMPLE = 4       # We only extract the 4 LSBs from the 12-bit ADC value
MAX_ADC_VALUE = 4095      # Max 12-bit value
NUM_RUNS = 100             # Number of blocks to time for a reliable average

# --- GLOBAL STATE FOR PERSISTENT LFSR ---
LFSR_STATE_BITS = 256
# Initialize the state globally for continuous stream mixing.
# Note: The LFSR state must be re-initialized before each timing run to ensure fair comparison.
GLOBAL_LFSR_STATE = np.array([1] * 2 + [0] * (LFSR_STATE_BITS - 4) + [1] * 2, dtype=np.uint8)

# --- POST-PROCESSING METHODS (Taken from your Pi Estimator) ---

def lfsr_whitening(bits, taps=(9, 4, 1), state_bits=256):
    """
    LFSR whitening using a 256-bit state and primitive taps (9, 4, 1) 
    The state is maintained globally to ensure a continuous stream across reads.
    Returns (output_bits, input_consumed).
    """
    global GLOBAL_LFSR_STATE
    
    bits = np.array(bits, dtype=np.uint8)
    state = GLOBAL_LFSR_STATE 
    
    output = []
    for b in bits:
        # Calculate feedback bit (XOR of taps and the input bit)
        feedback = b
        for t in taps:
            feedback ^= state[t]
        
        output.append(feedback)
        
        # Update the state: Shift all bits and insert the new feedback bit at the MSB position
        state[:-1] = state[1:]
        state[-1] = feedback
            
    return np.array(output, dtype=np.uint8), len(bits)

def lfsr_then_xor(bits):
    """Applies LFSR whitening, then XORs adjacent bits for the final output."""
    lfsr_bits, lfsr_consumed = lfsr_whitening(bits)
    if len(lfsr_bits) < 2: return np.array([], dtype=np.uint8), lfsr_consumed
    
    # Simple XOR whitening (XOR adjacent bits)
    xor_bits = lfsr_bits[1:] ^ lfsr_bits[:-1]
    
    return xor_bits, lfsr_consumed

# --- SERIAL UTILITIES ---

def read_serial_line(ser):
    """Reads a single line from the serial port with timeout."""
    line = b''
    while True:
        try:
            char = ser.read(1)
            if not char:
                return None
            line += char
            if char == b'\n':
                return line.decode('ascii').strip()
        except Exception:
            return None

# --- TIMING FUNCTION ---

def time_single_block():
    """Measures the time taken to read, process, and finalize one 256-bit block."""
    global GLOBAL_LFSR_STATE 
    
    all_timings = []
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        print(f"Opened serial port {COM_PORT} at {BAUD_RATE} baud.")
        
    except serial.SerialException as e:
        print(f"ERROR: Could not open {COM_PORT}. Check device name and connection: {e}", file=sys.stderr)
        return

    print(f"Starting timing test for {NUM_RUNS} blocks...")
    print("This measures I/O time + Post-Processing time.")

    for i in range(NUM_RUNS):
        # Reset LFSR state for each trial to prevent corruption from previous incomplete runs
        GLOBAL_LFSR_STATE = np.array([1] * 2 + [0] * (LFSR_STATE_BITS - 4) + [1] * 2, dtype=np.uint8)

        current_bits = []
        start_time = time.time()
        
        # --- 1. DATA COLLECTION LOOP (I/O) ---
        # Collect raw 4-LSB bits until we have enough for a full 256-bit block
        while len(current_bits) < BLOCK_SIZE_BITS:
            line = read_serial_line(ser)
            if line:
                try:
                    adc_value = int(line)
                    if 0 <= adc_value <= MAX_ADC_VALUE:
                        # Extract the 4 LSBs
                        for bit_idx in range(BITS_PER_SAMPLE):
                            current_bits.append((adc_value >> bit_idx) & 1)
                except ValueError:
                    pass

        # Stop timing I/O
        io_end_time = time.time()
        
        # --- 2. POST-PROCESSING AND TIMING ---
        raw_bits = np.array(current_bits[:BLOCK_SIZE_BITS], dtype=np.uint8)
        
        # Apply the LFSR_THEN_XOR post-processing
        processed_bits, raw_consumed = lfsr_then_xor(raw_bits)
        
        # Final timing mark
        total_end_time = time.time()
        
        # Ensure we have a valid output block (it should be 255 bits after XOR)
        if len(processed_bits) < BLOCK_SIZE_BITS - 1:
            print(f"Run {i+1}: Failed to generate full processed block. Skipping.")
            continue
            
        elapsed_time = total_end_time - start_time
        io_time = io_end_time - start_time
        proc_time = total_end_time - io_end_time
        
        all_timings.append(elapsed_time)
        print(f"Run {i+1}/{NUM_RUNS}: Total Time: {elapsed_time:.4f}s | I/O Time: {io_time:.4f}s | Proc Time: {proc_time:.4f}s")
        
        # Clear buffer for next run
        ser.reset_input_buffer()

    # --- FINAL REPORT ---
    ser.close()
    
    if all_timings:
        avg_time = np.mean(all_timings)
        std_dev = np.std(all_timings)
        
        print("\n" + "="*70)
        print("      SINGLE 256-BIT BLOCK GENERATION REPORT (LFSR + XOR)")
        print("="*70)
        print(f"| Test Runs: {NUM_RUNS}")
        print("-" * 70)
        print(f"| Mean Total Time per Block (I/O + Post-processing): {avg_time:.4f} seconds")
        print(f"| Standard Deviation: {std_dev:.4f} seconds")
        print("-" * 70)
        print(f"| Derived Throughput: {1/avg_time:.2f} blocks/second")
        print("=" * 70)
    else:
        print("\nTiming test failed to collect any valid data. Check your serial connection.")

if __name__ == "__main__":
    time_single_block()
