#!/bin/bash

# Command prefix
CMD_PREFIX="./omp_example/build/omp_example"
ITERATIONS=3
THREADS="1,2,4,6,8"

# Array of input sizes
# INPUT_SIZES=( $((2**12)) $((2**16)) $((2**20)) $((2**24)) $((2**28)) )
INPUT_SIZES=( $((2**12)) $((2**16)) $((2**20)))

# INPUT_SIZES=( $((2**24)) )

# Name of the output file
OUTPUT_FILE="hexl_out.csv"

# Clear any previous output file to start fresh
> $OUTPUT_FILE

# Loop through each input size and execute the command
for size in "${INPUT_SIZES[@]}"; do
    echo "Running with input size: $size"
    echo "Input Size = $size" >> $OUTPUT_FILE
    $CMD_PREFIX $ITERATIONS $THREADS $size >> $OUTPUT_FILE
    echo -e "\n" >> $OUTPUT_FILE
done

echo "All runs completed!"
