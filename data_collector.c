#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libserialport.h> // For serial communication

#define COM_PORT "/dev/ttyACM0"
#define BAUD_RATE 115200
#define BLOCK_SIZE_BITS 256
#define BLOCK_SIZE_BYTES (BLOCK_SIZE_BITS / 8) // 32 bytes
#define TARGET_BLOCKS 40000 
#define TARGET_BYTES (TARGET_BLOCKS * BLOCK_SIZE_BYTES) // 1,280,000 bytes

// Function to handle serial communication and data collection
int collect_data() {
    struct sp_port *port;
    FILE *fout = NULL;
    int ret;
    int blocks_written = 0;
    unsigned char byte_buffer[BLOCK_SIZE_BYTES];
    int bit_index = 0; // Tracks the current bit being packed (0 to 255)
    
    // --- 1. Open Serial Port ---
    ret = sp_get_port_by_name(COM_PORT, &port);
    if (ret != SP_OK) { /* Handle error */ return 1; }
    sp_open(port, SP_MODE_READ);
    sp_set_baudrate(port, BAUD_RATE);
    // ... set other serial parameters (data bits, parity, stop bits) ...

    // --- 2. Open Output File ---
    fout = fopen("256bit_blocks.bin", "wb");
    if (fout == NULL) { /* Handle error */ return 1; }

    printf("Collecting %d blocks... (%d bits/block)\n", TARGET_BLOCKS, BLOCK_SIZE_BITS);

    while (blocks_written < TARGET_BLOCKS) {
        char line_buffer[16];
        int num_read;
        
        // Read a line from the serial port (this is the tricky, platform-specific part)
        // This is highly simplified and assumes the microcontroller sends clean, line-delimited numbers.
        // A robust implementation requires careful reading and parsing of the incoming stream.
        num_read = sp_blocking_read(port, line_buffer, sizeof(line_buffer) - 1, 100);
        if (num_read <= 0) { continue; }
        line_buffer[num_read] = '\0';
        
        // --- 3. Parse and Pack Data ---
        int adc_value = atoi(line_buffer); 
        if (adc_value == 0 && line_buffer[0] != '0') { continue; } // Failed parsing

        for (int i = 0; i < 4; i++) {
            // Extract the 4 LSBs
            int bit = (adc_value >> i) & 1;
            
            // Calculate byte index and bit position within the byte
            int byte_idx = bit_index / 8;
            int bit_pos_in_byte = bit_index % 8;
            
            // Clear or set the bit in the buffer (assuming MSB first packing)
            if (bit_pos_in_byte == 0) {
                byte_buffer[byte_idx] = 0; // Start new byte
            }
            // Pack the bits one by one (this depends on your desired bit order)
            byte_buffer[byte_idx] = (byte_buffer[byte_idx] << 1) | bit; 
            
            bit_index++;
        }
        
        // --- 4. Write Block to File ---
        if (bit_index >= BLOCK_SIZE_BITS) {
            fwrite(byte_buffer, 1, BLOCK_SIZE_BYTES, fout);
            fflush(fout);
            blocks_written++;
            bit_index = 0;
            
            if (blocks_written % 1000 == 0) {
                printf("Wrote %d/%d blocks.\n", blocks_written, TARGET_BLOCKS);
            }
        }
    }

    // --- 5. Cleanup ---
    sp_close(port);
    sp_free_port(port);
    fclose(fout);
    printf("Data collection complete.\n");
    return 0;
}

int main() {
    return collect_data();
}
