#!/bin/bash

# Initialize the parameters
iterations=100
thread_numbers="1 2 4 6 8"
# thread_numbers="1 2 4"

method_number=4
# INPUT_SIZES=( $((2**12)) $((2**16)) $((2**20)) $((2**24)) $((2**28)) )
INPUT_SIZES=( $((2**12)) $((2**16)) $((2**20)) $((2**24)))

# Initialize CSV file and write header
csv_file="omp_time.csv"
tmp_file="temp_file.csv"

echo "Method_Number,Thread_Count,Input_Size,Average_Elapsed_Time" > $csv_file

# Loop through the input sizes
for input_size in "${INPUT_SIZES[@]}"; do
    echo "Running test with input_size=${input_size}"

    # Loop through each thread number
    for thread in $thread_numbers; do
        echo "Running with thread_count=${thread}"

        # # Run the command along with the binary and parameters
        # # Capture the output to a temporary file
        ./time_example/build/time_example $iterations $thread $input_size $method_number > $tmp_file

        # Initialize variables to calculate the average
        total_time=0
        count=0
        error_count=0


        # Read each line from the temporary file
        while read -r line; do
            # Extract thread count and elapsed time from the line
            read -r out_thread out_time <<< "$line"

            # Check if thread counts match
            if [ "$out_thread" -ne "$thread" ]; then
                echo "Error: Mismatch in thread count (expected $thread, got $out_thread)"
                error_count=$((error_count + 1))
                continue
            fi

            # Update total time and count
            # echo "$total_time + $out_time" | bc -l
            total_time=$(echo "$total_time + $out_time" | bc -l)
            count=$((count + 1))

        done < $tmp_file

        # Calculate and append the average to the CSV file
        if [ "$count" -ne 0 ]; then
            average_time=$(echo "$total_time / $count" | bc -l)
            echo "$method_number,$thread,$input_size,$average_time" >> $csv_file
        fi

        # Remove the temporary file
        rm -f $tmp_file

        # ./time_example/build/time_example $iterations $thread $input_size $method_number
    done
done

echo "All tests completed."
