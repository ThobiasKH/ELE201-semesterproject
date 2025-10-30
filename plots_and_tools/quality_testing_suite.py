import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import threading
from scipy.stats import chi2 # Used for statistical thresholds
from scipy.special import erfc # Used for P-value calculation

# Try to import TclError for graceful plot closing. This is common in Matplotlib.
try:
    from _tkinter import TclError 
except ImportError:
    class TclError(Exception): pass # Define a dummy class if Tkinter isn't available

# --- CONFIGURATION ---
COM_PORT = '/dev/ttyACM0'  # IMPORTANT: Change this if needed (e.g., 'COM3' on Windows)
BAUD_RATE = 115200
BLOCK_SIZE_BITS = 256
SAMPLES_PER_BLOCK = 64  # 64 samples * 4 LSBs/sample = 256 bits
BITS_PER_SAMPLE = 4     # We only care about the 4 LSBs
MAX_ADC_VALUE = 4095    # Max value for a 12-bit ADC (2^12 - 1)
ALPHA = 0.01            # Significance level (P-value threshold for PASS/FAIL)
WINDOW_SIZE = 50        # Moving average window size for the plot

# --- GLOBAL VARIABLES FOR ANALYSIS ---
# CRITICAL: Lock for thread-safe access to all global history lists and counters
data_lock = threading.Lock() 

monobit_history = []    # Tracks Monobit P-value (True/False pass)
runs_history = []       # Tracks Runs Test P-value (True/False pass)
entropy_history = []    # Tracks Serial Overlapping P-value (True/False pass)
pass_rate_history = []  # Tracks overall pass rate (True/False for 3/3 tests)

# Global counters needed for plotting and final report
block_counter = 0
last_block_time = 0.0
start_time = 0.0 # Will be set in run_analyzer

# --- SERIAL UTILITIES (Enhanced Line Reader) ---

def read_serial_line(ser):
    """
    Reads a complete line from the serial port.
    Returns the decoded line or None on error/timeout.
    """
    line = b''
    # We use a short timeout set in serial.Serial()
    while True:
        try:
            char = ser.read(1)
            if not char:
                # Timeout occurred, return None
                return None
            
            line += char
            
            if char == b'\n':
                return line.decode('ascii').strip()
            
        except serial.SerialException as e:
            print(f"Serial Error: {e}", file=sys.stderr)
            return None
        except UnicodeDecodeError:
            # Handle non-ascii/corrupted bytes
            print(f"WARNING: Corrupted data detected, discarding line.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Unexpected Error during read: {e}", file=sys.stderr)
            return None

# --- CRYPTOGRAPHIC STATISTICAL TESTS (Based on NIST 800-22) ---

def test_monobit_frequency(bitstream_np):
    """
    Test 1: Monobit Frequency Test.
    """
    n = len(bitstream_np)
    S_n = np.sum(bitstream_np)
    
    # Calculate Z-score
    Z = abs(S_n - n / 2.0) / np.sqrt(n / 4.0)
    
    # P-value based on Normal distribution approximation
    p_value = erfc(Z / np.sqrt(2.0))
    
    return p_value

def test_runs(bitstream_np):
    """
    Test 2: Runs Test.
    """
    n = len(bitstream_np)
    pi = np.sum(bitstream_np) / n
    
    # Pre-test condition: Monobit Frequency must be close to 0.5 for runs test to be meaningful
    if not (0.49 < pi < 0.51):
        # A failed pre-condition is treated as a test failure (P-value=0)
        return 0.0 
    
    # Calculate V_n: the total number of runs
    V_n = 1 + np.sum(bitstream_np[:-1] != bitstream_np[1:])
    
    # Calculate expected value (mu) and variance (sigma^2) of V_n
    mu = 2 * n * pi * (1 - pi)
    sigma2 = 2 * n * pi * (1 - pi) * (1 - 2 * pi * (1 - pi))
    
    if sigma2 <= 0:
        return 0.0 # Avoid division by zero
        
    # Calculate Z-score
    Z = abs(V_n - mu) / np.sqrt(sigma2)
    
    # P-value based on Normal distribution approximation
    p_value = erfc(Z / np.sqrt(2.0))
    
    return p_value

def test_serial_overlapping_blocks(bitstream_np, m=4):
    """
    Test 3: Serial Overlapping Blocks Test (m=4).
    """
    n = len(bitstream_np)
    bit_str = ''.join(map(str, bitstream_np.astype(int))) # Ensure integers for mapping
    
    # Total overlapping m-bit blocks is n
    N_blocks = n
    counts = {}
    
    for i in range(n):
        block = bit_str[i:i+m]
        # Wrap around for the last few blocks
        if len(block) < m:
            block = bit_str[i:] + bit_str[:m - len(bit_str[i:])] 
        
        counts[block] = counts.get(block, 0) + 1
        
    num_categories = 2**m
    expected_count = N_blocks / num_categories
    
    if expected_count < 5:
        # Chi-squared test requires an expected count of at least 5 in each bin
        return 0.0 

    chi_square_stat = 0
    for block in counts:
        chi_square_stat += (counts[block] - expected_count)**2 / expected_count
        
    # Degrees of freedom for this test is 2^m - 1
    df = num_categories - 1
    
    # Calculate P-value (survival function is 1 - CDF)
    p_value = chi2.sf(chi_square_stat, df)
    
    return p_value

def analyze_block_strigent(bitstream_np, block_num):
    """
    Executes multiple cryptographic tests, stores history safely, and prints status.
    """
    
    # Execute tests and store P-values
    p_monobit = test_monobit_frequency(bitstream_np)
    p_runs = test_runs(bitstream_np)
    p_entropy = test_serial_overlapping_blocks(bitstream_np, m=4)

    # Check for passing condition (P-value must be greater than alpha)
    pass_monobit = p_monobit >= ALPHA
    pass_runs = p_runs >= ALPHA
    pass_entropy = p_entropy >= ALPHA
    
    tests_passed = int(pass_monobit) + int(pass_runs) + int(pass_entropy)
    overall_pass = (tests_passed == 3) # True/False
    
    # --- CRITICAL FIX: THREAD-SAFE HISTORY UPDATE ---
    global data_lock
    with data_lock:
        # Store history (True/False converted to 1/0 is implicitly handled by Python lists)
        monobit_history.append(pass_monobit)
        runs_history.append(pass_runs)
        entropy_history.append(pass_entropy)
        pass_rate_history.append(overall_pass)
    
    # Console Output
    print(f"--- BLOCK {block_num} (P > {ALPHA}) ---")
    print(f"  [1] Monobit Freq: P={p_monobit:.4e} | {'PASS' if pass_monobit else '**FAIL**'}")
    print(f"  [2] Runs Test:    P={p_runs:.4e} | {'PASS' if pass_runs else '**FAIL**'}")
    print(f"  [3] Entropy (m=4): P={p_entropy:.4e} | {'PASS' if pass_entropy else '**FAIL**'}")
    print(f"  --- Overall Status: {'ALL PASS' if overall_pass else f'FAIL ({tests_passed}/3)'} ---")
    
    return overall_pass

# --- PLOTTING THREAD ---

def plot_data():
    """Real-time plot update function run in a separate thread."""
    
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line, = ax.plot([], [], 'b-', linewidth=2, label=f'Overall Pass Rate (Avg over {WINDOW_SIZE} blocks)')
    ax.axhline(1.0 - ALPHA, color='g', linestyle='--', label=f'Target Pass Rate ({int((1.0 - ALPHA) * 100)}%)')
    
    ax.set_xlabel(f'Block Number ({BLOCK_SIZE_BITS}-bits/block)')
    ax.set_ylabel('Overall Block Pass Rate (Moving Average)')
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    
    def update_plot():
        # --- CRITICAL FIX: Use Lock to safely copy data and globals ---
        global data_lock, block_counter
        with data_lock:
            current_block_count = block_counter
            history_copy = pass_rate_history.copy()
            current_start_time = start_time
        
        if current_block_count >= WINDOW_SIZE:
            # 1. Calculate rolling pass rate 
            # Boolean True/False convert to 1.0/0.0 automatically here
            pass_history_np = np.array(history_copy).astype(float)
            moving_avg = np.convolve(pass_history_np, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
            
            # 2. Correct X-axis Indexing (offset by WINDOW_SIZE - 1)
            x_data = np.arange(len(moving_avg)) + (WINDOW_SIZE - 1)
            y_data = moving_avg
            
            # 3. Update plot data
            line.set_xdata(x_data)
            line.set_ydata(y_data)
            
            # 4. Update axis limits
            ax.set_xlim(0, current_block_count + WINDOW_SIZE) # Give a little extra space
            
            # 5. Update title with throughput
            current_time = time.time()
            if current_block_count > 0 and current_time > current_start_time:
                # Estimate throughput 
                throughput = BLOCK_SIZE_BITS * current_block_count / (current_time - current_start_time)
                ax.set_title(f'RNG Quality | Blocks: {current_block_count} | Throughput: {throughput:.0f} bits/sec')
            else:
                 ax.set_title(f'RNG Quality | Blocks: {current_block_count} | Analyzing...')

            # 6. Redraw plot
            try:
                fig.canvas.draw()
                fig.canvas.flush_events()
            except TclError:
                # If TclError occurs here (e.g. plot closed), we re-raise to catch in the outer loop
                raise TclError 
            
    while True:
        try:
            update_plot()
            time.sleep(0.5) # Update every half second
        except TclError:
            print("Plot window closed by user.")
            break
        except Exception as e:
            # Catch other potential plotting errors gracefully
            print(f"Plotting error: {e}", file=sys.stderr)
            break

# --- MAIN EXECUTION ---

def run_analyzer():
    # Only need to declare globals we will WRITE to
    global block_counter, last_block_time, start_time, data_lock
    
    # --- 1. SETUP ---
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.01) # Low timeout for responsiveness
        ser.reset_input_buffer()
        print(f"Successfully opened serial port: {COM_PORT} at {BAUD_RATE} baud.")
        
        # Set start time globally
        with data_lock:
             start_time = time.time()
             
    except serial.SerialException as e:
        print(f"ERROR: Could not open serial port {COM_PORT}. Check connection and permissions.", file=sys.stderr)
        print(e, file=sys.stderr)
        return

    # Start the plotting thread
    plot_thread = threading.Thread(target=plot_data, daemon=True)
    plot_thread.start()
    
    # Initialize variables for the current block
    current_block_bits = []
    
    # --- 2. MAIN LOOP ---
    try:
        print(f"Collecting and testing {BLOCK_SIZE_BITS}-bit blocks. Press Ctrl+C to stop.")
        while True:
            # 2.1. Read and Validate Sample
            line = read_serial_line(ser)
            
            if line is None: continue
            
            try:
                adc_value = int(line)
                if not (0 <= adc_value <= MAX_ADC_VALUE):
                    print(f"WARNING: Discarding out-of-range value: {line}.", file=sys.stderr)
                    continue
            except ValueError:
                print(f"WARNING: Discarding non-integer line: '{line}'", file=sys.stderr)
                continue
            
            # 2.2. Extract and Pack Bits
            for i in range(BITS_PER_SAMPLE):
                bit = (adc_value >> i) & 1
                current_block_bits.append(bit)
            
            # 2.3. Analyze Block
            if len(current_block_bits) >= BLOCK_SIZE_BITS:
                bitstream_np = np.array(current_block_bits[:BLOCK_SIZE_BITS])
                
                # CRITICAL: Update the block_counter and last_block_time inside the lock
                with data_lock:
                    block_counter += 1
                    last_block_time = time.time()
                    
                # Perform analysis (which updates histories safely with its own lock)
                analyze_block_strigent(bitstream_np, block_counter)
                
                # Reset for the next block, keeping any extra bits
                current_block_bits = current_block_bits[BLOCK_SIZE_BITS:]
                
                if block_counter % 20 == 0:
                     print("-" * 50)
                     
    except KeyboardInterrupt:
        print("\n\nAnalysis stopped by user (Ctrl+C).")
        
    finally:
        # --- 3. CLEANUP & FINAL REPORT ---
        if 'ser' in locals() and ser.is_open:
            ser.close()
            
        print("\n\n" + "="*50)
        print("                 FINAL REPORT")
        print("="*50)
        
        # Use lock to access final global states
        with data_lock:
            total_time = time.time() - start_time
            history_copy = pass_rate_history.copy()

        if block_counter > 0 and total_time > 0:
            overall_pass_pct = np.mean(history_copy) * 100
            throughput = BLOCK_SIZE_BITS * block_counter / total_time
            
            print(f"Total Blocks Analyzed: {block_counter}")
            print(f"Total Run Time: {total_time:.1f} seconds")
            print(f"Overall Throughput: {throughput:.0f} bits/sec")
            print(f"Overall Block Pass Rate (3/3 Tests): {overall_pass_pct:.2f}%")
            print("="*50)

        # Keep the plot window open
        if plot_thread.is_alive():
             print("\nDisplaying final plot. Close the window to exit.")
             plt.ioff()
             try:
                 plt.show() 
             except TclError:
                 pass # Plot was already closed

if __name__ == "__main__":
    try:
        import serial
        import numpy
        import matplotlib
        import scipy
    except ImportError:
        print("ERROR: Required libraries not found. Please install them:", file=sys.stderr)
        print("pip install pyserial numpy matplotlib scipy", file=sys.stderr)
        sys.exit(1)
        
    run_analyzer()
