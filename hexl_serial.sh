#!/bin/bash

# Define constants
iteration_number="10"
thread_numbers="4096,65536,1048576,16777216,268435456"
file_name="serial_result.csv"

# Check if the CSV file exists. If not, create it and add the header.
if [ ! -f $file_name ]; then
  echo "Method,Threads=4096,Threads=65536,Threads=1048576,Threads=16777216,Threads=268435456" > $file_name
fi

# Loop over iterations
  # Run the command and append its output to the CSV file
./build/example $iteration $thread_numbers >> $file_name
