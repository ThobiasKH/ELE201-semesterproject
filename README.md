# ELE201 Semester project 

This repo constitutes work towards my semester project in ELE201 Microcontrollers & Data Networks. 

The goal is to analyze the randomness of data collected with an analog sensor (like a photoresistor), which we pass through ADC using an stm32 microcontroller. The data is sent via a serial connection to a computer to be analyzed for randomness before and after post-processing. 

We will discuss feasability for our setup as a hardware-based random number generator. 

If anyone with a similar setup wants to try out some of the code in the repo, keep in mind that changes had to be made to make uploading the microcontroller code simple on my personal linux setup. If you are on windows you may have to rename, for instance, the src and include directories.
