import numpy as np
import matplotlib.pyplot as plt
import os
import serial  # Added for serial communication
import time    # Added for time utilities
import sys     # Added for system utilities

# --- CONFIGURATION ---
COM_PORT = '/dev/ttyACM0'  # Serial port
BAUD_RATE = 115200         # Baud rate
BLOCK_SIZE_BITS = 256
BLOCK_SIZE_BYTES = BLOCK_SIZE_BITS // 8

# Serial reading configuration
BITS_PER_SAMPLE = 4
MAX_ADC_VALUE = 4095      # Max 12-bit value
TARGET_BLOCKS = 5000      # Stop plotting after this many blocks

# --- GLOBAL STATE FOR PERSISTENT LFSR ---
LFSR_STATE_BITS = 256
# Initialize the state globally for continuous stream mixing during processing.
# This state persists throughout the entire session.
GLOBAL_LFSR_STATE = np.array([1] * 2 + [0] * (LFSR_STATE_BITS - 4) + [1] * 2, dtype=np.uint8)

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
                # Decode and strip, handling potential decode errors gracefully
                try:
                    return line.decode('ascii').strip()
                except UnicodeDecodeError:
                    return None
        except Exception:
            # Handle serial exceptions during read
            return None


# --- POST-PROCESSING SCHEME (Remains the same) ---

def lfsr_whitening(bits, taps=(9, 4, 1)):
    """
    Applies LFSR whitening using a 256-bit state maintained globally.
    The output is the mixed stream.
    """
    global GLOBAL_LFSR_STATE
    
    bits = np.array(bits, dtype=np.uint8)
    state = GLOBAL_LFSR_STATE
    output = []
    
    for b in bits:
        # Calculate feedback bit (XOR of taps and the input bit)
        # Taps are chosen based on a primitive polynomial for 256 bits.
        feedback = b
        for t in taps:
            feedback ^= state[t]
        
        # The new output bit is the calculated feedback
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

# --- CONVERSION AND ANALYSIS ---

def block_to_normalized_float(bit_array_256):
    """
    Converts a 256-bit numpy array into a single floating-point number 
    normalized between 0 and 1.
    
    NOTE: We use the first 64 bits for the float representation.
    """
    if len(bit_array_256) != BLOCK_SIZE_BITS:
        return 0.0

    # Use the first 64 bits for the float normalization (2**64)
    bits_64 = bit_array_256[:64]
    
    # Convert bit array to a single 64-bit unsigned integer.
    int_value_64 = np.sum(bits_64 * (2**np.arange(63, -1, -1)))
    
    # Normalize by 2^64
    return int_value_64 / (np.float64(2)**64)


def analyze_and_plot():
    """Reads data from USART, applies post-processing, and plots the distribution."""
    global GLOBAL_LFSR_STATE
    
    # Reset LFSR state for a fresh analysis run
    GLOBAL_LFSR_STATE = np.array([1] * 2 + [0] * (LFSR_STATE_BITS - 4) + [1] * 2, dtype=np.uint8)
    
    processed_blocks = []
    current_bits = []
    blocks_collected = 0
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        print(f"Opened serial port {COM_PORT} at {BAUD_RATE} baud.")
        print(f"Collecting up to {TARGET_BLOCKS} blocks for distribution analysis. Press Ctrl+C to stop early.")
    except serial.SerialException as e:
        print(f"ERROR: Could not open {COM_PORT}. Check device name and connection: {e}", file=sys.stderr)
        return

    try:
        while blocks_collected < TARGET_BLOCKS:
            # --- 1. Read and Extract 4 LSBs ---
            line = read_serial_line(ser)
            if line:
                try:
                    adc_value = int(line)
                    if 0 <= adc_value <= MAX_ADC_VALUE:
                        # Extract the 4 LSBs and append them as a raw bit stream
                        for bit_idx in range(BITS_PER_SAMPLE):
                            current_bits.append((adc_value >> bit_idx) & 1)
                except ValueError:
                    # Ignore lines that don't parse as integers
                    pass

            # --- 2. Process Available Bits ---
            if len(current_bits) > 0:
                bits_to_process = np.array(current_bits, dtype=np.uint8)
                
                # Apply the continuous LFSR_THEN_XOR post-processing
                processed_bits, raw_consumed = lfsr_then_xor(bits_to_process)
                
                # --- 3. Extract Full Blocks ---
                if len(processed_bits) >= BLOCK_SIZE_BITS:
                    num_available_blocks = len(processed_bits) // BLOCK_SIZE_BITS
                    
                    for i in range(num_available_blocks):
                        start = i * BLOCK_SIZE_BITS
                        end = (i + 1) * BLOCK_SIZE_BITS
                        block = processed_bits[start:end]
                        
                        normalized_float = block_to_normalized_float(block)
                        processed_blocks.append(normalized_float)
                        blocks_collected += 1
                        
                        if blocks_collected % 100 == 0:
                            print(f"Collected {blocks_collected}/{TARGET_BLOCKS} blocks...")
                            
                        if blocks_collected >= TARGET_BLOCKS:
                            break

                    # The LFSR/XOR consumes raw_consumed bits from the input buffer for the whole process
                    raw_bits_to_discard = raw_consumed 
                    
                    # Keep the remaining raw input bits for the next cycle
                    current_bits = current_bits[raw_bits_to_discard:]

                # Small sleep to prevent busy-waiting if data rate is low
                else:
                    time.sleep(0.005) 

    except KeyboardInterrupt:
        print("\nAnalysis stopped by user (Ctrl+C). Plotting available data.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    finally:
        ser.close()
        print("\nSerial connection closed.")

    # --- PLOTTING ---
    if not processed_blocks:
        print("No full blocks were successfully processed. Aborting plot.")
        return
    
    print(f"\nTotal blocks collected and analyzed: {len(processed_blocks)}")
        
    plt.figure(figsize=(12, 6))
    
    # Use 50 bins for a clear histogram visualization
    plt.hist(processed_blocks, bins=50, color='#3b82f6', edgecolor='black', alpha=0.8)
    
    plt.title('Distribution of 256-Bit Numbers After LFSR/XOR Whitening (Live USART)', fontsize=16)
    plt.xlabel('Normalized Value (0 to 1)', fontsize=14)
    plt.ylabel('Frequency (Counts)', fontsize=14)
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    
    plt.figtext(0.5, 0.01, f'Total Blocks: {len(processed_blocks)} | Source: {COM_PORT}', ha='center', fontsize=10, color='gray')
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    analyze_and_plot()
