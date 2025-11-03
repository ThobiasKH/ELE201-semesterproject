import serial
import time
import numpy as np
import sys
import threading
import argparse
from collections import defaultdict
import os

# --- PLOTTING IMPORTS ---
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
COM_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
BLOCK_SIZE_BITS = 256
BITS_PER_COORDINATE = 32
PI_TRUE = np.pi
PLOT_UPDATE_INTERVAL = 100 # Update the plot data every 100 processed blocks
# Serial reading configuration (unchanged)
BITS_PER_SAMPLE = 4
MAX_ADC_VALUE = 4095
MAX_RAW_BUFFER_SIZE = BLOCK_SIZE_BITS * 8 

# --- GLOBAL VARIABLES ---
data_lock = threading.Lock()
inside_circle_count = 0        
total_points_generated = 0     
block_counter = 0              
start_time = 0.0

# --- GLOBAL VARIABLES FOR PLOTTING ---
pi_history = []
block_history = []

# --- GLOBAL STATE FOR PERSISTENT LFSR (FIX for the 3.07 bias) ---
LFSR_STATE_BITS = 256
# Initialize the state once globally to ensure continuous stream mixing.
# The state is used by lfsr_whitening and maintained between calls.
GLOBAL_LFSR_STATE = np.array([1] * 2 + [0] * (LFSR_STATE_BITS - 4) + [1] * 2, dtype=np.uint8)


# --- MONTE CARLO UTILITIES ---

def bits_to_float(bit_array_32):
    """Converts a 32-bit numpy array (of 0s and 1s) into a float between 0 and 1."""
    if len(bit_array_32) != BITS_PER_COORDINATE:
        return 0.0 
    
    # Convert bit array to a single 32-bit unsigned integer. 
    int_value = np.sum(bit_array_32 * (2**np.arange(BITS_PER_COORDINATE - 1, -1, -1)))
    
    # Normalize by 2**32 to get a float in [0, 1)
    return int_value / (2**BITS_PER_COORDINATE)


def update_plot_data(blocks, pi_est):
    """Appends the current block count and Pi estimate to the history lists."""
    with data_lock:
        pi_history.append(pi_est)
        block_history.append(blocks)


def estimate_pi_from_block(bitstream_np, block_num):
    """Uses the 256-bit stream to generate 4 (x, y) coordinate pairs and update PI counters."""
    global inside_circle_count, total_points_generated, block_counter
    
    if len(bitstream_np) < BLOCK_SIZE_BITS: 
        return

    points_this_block = 0
    hits_this_block = 0
    coords_per_block = BLOCK_SIZE_BITS // (BITS_PER_COORDINATE * 2) # 4 pairs of (X,Y)

    for i in range(coords_per_block):
        start_index = i * BITS_PER_COORDINATE * 2
        
        x_bits = bitstream_np[start_index : start_index + BITS_PER_COORDINATE]
        y_bits = bitstream_np[start_index + BITS_PER_COORDINATE : start_index + BITS_PER_COORDINATE * 2]
        
        x = bits_to_float(x_bits)
        y = bits_to_float(y_bits)
        
        if x*x + y*y <= 1.0:
            hits_this_block += 1
            
        points_this_block += 1

    with data_lock:
        inside_circle_count += hits_this_block
        total_points_generated += points_this_block
        
    pi_estimate = 4.0 * (inside_circle_count / total_points_generated) if total_points_generated > 0 else 0.0
    
    # Check if it's time to update the plot data
    if block_counter % PLOT_UPDATE_INTERVAL == 0:
        update_plot_data(block_counter, pi_estimate)

    # Print status update
    print(f"--- BLOCK {block_num} ({BLOCK_SIZE_BITS} bits) ---")
    print(f"  Points in Block: {points_this_block} | Total Points: {total_points_generated}")
    print(f"  Current π Estimate: {pi_estimate:.6f}")
    print("-" * 50)


def show_final_plot(method, total_points):
    """Generates and saves the final convergence plot."""
    if not pi_history:
        print("Not enough data collected for plotting.")
        return

    output_filename = f"pi_convergence_{method}.png"

    try:
        plt.figure(figsize=(10, 6))
        
        # Plot the convergence line
        plt.plot(block_history, pi_history, label=f'{method.upper()}' + r' $\pi$ Estimate', color='#3b82f6', linewidth=2)
        
        # Plot the true value of Pi
        # Re-using .format() to correctly embed the PI_TRUE value in the raw string
        plt.axhline(PI_TRUE, color='#ef4444', linestyle='--', label=r'True $\pi$ ({PI_TRUE:.5f})'.format(PI_TRUE=PI_TRUE))
        
        # Re-adding the method name to the title
        plt.title(f'Monte Carlo ' + r'$\pi$' + f' Convergence ({method.upper()} Method)', fontsize=16)
        plt.xlabel(f'Blocks Processed ({PLOT_UPDATE_INTERVAL} blocks per point)', fontsize=12)
        plt.ylabel(r'Estimated Value of $\pi$', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure instead of showing the interactive window
        plt.savefig(output_filename)
        print(f"\n[SUCCESS] Convergence plot saved to: {output_filename}")
        
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred during plotting or saving: {e}")


def print_pi_report(total_blocks, method, elapsed_time):
    """Prints a summary of the Pi estimation."""
    if total_points_generated == 0:
        print("\n--- FINAL REPORT ---")
        print("No points were successfully processed for Pi estimation.")
        return

    estimated_pi = 4.0 * (inside_circle_count / total_points_generated)
    error = abs(estimated_pi - PI_TRUE)

    print("\n" + "="*50)
    print("      MONTE CARLO PI ESTIMATION REPORT")
    print("="*50)
    print(f"| Post-Processing Method: {method.upper()}")
    print("-" * 50)
    print(f"| Total Blocks Processed: {total_blocks}")
    print(f"| Total Points Generated: {total_points_generated} (Trials)")
    print(f"| True Value of $\pi$:        {PI_TRUE:.10f}")
    print(f"| ESTIMATED Value of $\pi$:   {estimated_pi:.10f}")
    print(f"| ABSOLUTE ERROR (|$\pi$-E|): {error:.10f}")
    print("-" * 50)
    print(f"| Total Elapsed Time:     {elapsed_time:.2f} seconds")
    print("=" * 50)


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


# --- POST-PROCESSING METHODS (UPDATED LFSR) ---

def xor_whitening(bits):
    bits = np.array(bits, dtype=np.uint8)
    if len(bits) < 2: return bits, 0 
    return bits[1:] ^ bits[:-1], len(bits)

def von_neumann_corrector(bits):
    input_to_process = bits[:len(bits) - (len(bits) % 2)]
    output = []
    consumed_input_len = 0
    for i in range(0, len(input_to_process) - 1, 2):
        a, b = input_to_process[i], input_to_process[i+1]
        if a != b: output.append(a)
        consumed_input_len += 2
    return np.array(output, dtype=np.uint8), consumed_input_len

def lfsr_whitening(bits, taps=(9, 4, 1), state_bits=256):
    """
    LFSR whitening using a 256-bit state and primitive taps (9, 4, 1) 
    The state is maintained globally to ensure a continuous stream.
    Returns (output_bits, input_consumed).
    """
    global GLOBAL_LFSR_STATE # Access the persistent state
    
    bits = np.array(bits, dtype=np.uint8)
    state = GLOBAL_LFSR_STATE # Use the global state
    
    output = []
    for b in bits:
        # Calculate feedback bit (XOR of taps and the input bit)
        feedback = b
        for t in taps:
            feedback ^= state[t]
        
        # The new output bit is the calculated feedback
        output.append(feedback)
        
        # Update the state: Shift all bits and insert the new feedback bit at the MSB position
        state[:-1] = state[1:]
        state[-1] = feedback
        
    # No need to explicitly set GLOBAL_LFSR_STATE = state as numpy arrays are mutable
    # and the changes to 'state' (which references GLOBAL_LFSR_STATE) are already reflected.
        
    return np.array(output, dtype=np.uint8), len(bits)

def distance_xor_mixing(bits, distance):
    bits = np.array(bits, dtype=np.uint8)
    n = len(bits)
    if n <= distance: return np.array([], dtype=np.uint8), 0
    output = bits[:n - distance] ^ bits[distance:n]
    return output, n

def lfsr_then_xor(bits):
    lfsr_bits, lfsr_consumed = lfsr_whitening(bits)
    if len(lfsr_bits) < 2: return np.array([], dtype=np.uint8), lfsr_consumed
    xor_bits = lfsr_bits[1:] ^ lfsr_bits[:-1]
    return xor_bits, lfsr_consumed

def lfsr_then_distance_xor(bits):
    DISTANCE = BLOCK_SIZE_BITS // 2
    lfsr_bits, lfsr_consumed = lfsr_whitening(bits)
    if len(lfsr_bits) <= DISTANCE: return np.array([], dtype=np.uint8), lfsr_consumed
    n = len(lfsr_bits)
    xor_bits = lfsr_bits[:n - DISTANCE] ^ lfsr_bits[DISTANCE:n]
    return xor_bits, lfsr_consumed

def von_neumann_then_xor(bits):
    vn_bits, vn_consumed = von_neumann_corrector(bits)
    if len(vn_bits) < 2: return np.array([], dtype=np.uint8), vn_consumed
    xor_bits = vn_bits[1:] ^ vn_bits[:-1]
    return xor_bits, vn_consumed 

def lfsr_then_von_neumann(bits):
    lfsr_bits, lfsr_consumed = lfsr_whitening(bits)
    vn_bits, _ = von_neumann_corrector(lfsr_bits) 
    return vn_bits, lfsr_consumed

def lfsr_then_vn_then_xor(bits):
    lfsr_bits, lfsr_consumed = lfsr_whitening(bits)
    vn_bits, _ = von_neumann_corrector(lfsr_bits)
    if len(vn_bits) < 2: return np.array([], dtype=np.uint8), lfsr_consumed
    xor_bits = vn_bits[1:] ^ vn_bits[:-1]
    return xor_bits, lfsr_consumed


# --- MAIN EXECUTION ---

def run_analyzer(method):
    global block_counter, start_time, GLOBAL_LFSR_STATE 
    
    # Reset globals
    global inside_circle_count, total_points_generated
    inside_circle_count = 0
    total_points_generated = 0
    block_counter = 0
    pi_history.clear()
    block_history.clear()
    
    # Re-initialize the LFSR state for a fresh run if an LFSR method is selected
    if method in ["lfsr", "lfsr_then_xor", "lfsr_then_distance_xor", "lfsr_then_vn", "lfsr_vn_xor"]:
        GLOBAL_LFSR_STATE = np.array([1] * 2 + [0] * (LFSR_STATE_BITS - 4) + [1] * 2, dtype=np.uint8)

    DISTANCE_XOR_DISTANCE = BLOCK_SIZE_BITS // 2
    
    postproc_map = {
        "raw": lambda x: (x, len(x)),
        "xor": xor_whitening,
        "von_neumann": von_neumann_corrector,
        # Note: lfsr_whitening now uses the global state
        "lfsr": lambda x: lfsr_whitening(x, taps=(9, 4, 1), state_bits=256), 
        "vn_then_xor": von_neumann_then_xor,
        "lfsr_then_vn": lfsr_then_von_neumann,
        "lfsr_then_xor": lfsr_then_xor,
        "lfsr_then_distance_xor": lfsr_then_distance_xor,
        "distance_xor": lambda x: distance_xor_mixing(x, distance=DISTANCE_XOR_DISTANCE),
        "lfsr_vn_xor": lfsr_then_vn_then_xor
    }
    
    postproc = postproc_map.get(method, lambda x: (x, len(x)))

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        print(f"Opened serial port {COM_PORT} at {BAUD_RATE} baud.")
        start_time = time.time()
    except serial.SerialException as e:
        print(f"ERROR: Could not open {COM_PORT}. Check device name and connection: {e}", file=sys.stderr)
        return

    current_bits = []
    try:
        print(f"Collecting and estimating Pi using {BLOCK_SIZE_BITS}-bit blocks (method: {method})")
        print(f"Plotting convergence points every {PLOT_UPDATE_INTERVAL} blocks.")
        print("Press Ctrl+C to stop the analysis.")
        while True:
            line = read_serial_line(ser)
            if line:
                try:
                    adc_value = int(line)
                    if 0 <= adc_value <= MAX_ADC_VALUE:
                        for i in range(BITS_PER_SAMPLE):
                            current_bits.append((adc_value >> i) & 1)
                except ValueError:
                    pass

            if len(current_bits) >= BLOCK_SIZE_BITS:
                bits_to_process = np.array(current_bits, dtype=np.uint8)
                processed_bits, raw_consumed = postproc(bits_to_process)
                
                if len(processed_bits) >= BLOCK_SIZE_BITS:
                    analyzed_block = processed_bits[:BLOCK_SIZE_BITS]
                    block_counter += 1
                    estimate_pi_from_block(analyzed_block, block_counter)
                    
                    if raw_consumed > 0 and len(processed_bits) > 0:
                        raw_bits_to_discard = int(raw_consumed * (BLOCK_SIZE_BITS / len(processed_bits)))
                    else:
                        raw_bits_to_discard = BLOCK_SIZE_BITS
                        
                    raw_bits_to_discard = min(raw_bits_to_discard, raw_consumed, len(current_bits))

                    current_bits = current_bits[raw_bits_to_discard:]

                elif len(current_bits) > MAX_RAW_BUFFER_SIZE:
                    print(f"Warning: Raw buffer exceeded {MAX_RAW_BUFFER_SIZE} bits with negligible output. Clearing buffer.")
                    current_bits = []
            
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user.")
    finally:
        ser.close()
        end_time = time.time()
        if block_counter > 0:
            print_pi_report(block_counter, method, end_time - start_time)
            show_final_plot(method, total_points_generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Pi Estimator with convergence plotting.")
    parser.add_argument("--method", type=str, default="raw",
                        choices=["raw", "xor", "von_neumann", "lfsr", "vn_then_xor", "lfsr_then_vn", "lfsr_then_xor", "lfsr_then_distance_xor", "distance_xor", "lfsr_vn_xor"],
                        help="Post-processing method (default: raw).")
    args = parser.parse_args()
    run_analyzer(args.method)
