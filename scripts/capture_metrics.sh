    #!/bin/bash
    # Capture stabilization metrics from Android device via ADB logcat
    # Usage: ./capture_metrics.sh [output_file.csv]

    OUTPUT_FILE="${1:-metrics_$(date +%Y%m%d_%H%M%S).csv}"

    echo "Capturing metrics to: $OUTPUT_FILE"
    echo "Press Ctrl+C to stop"
    echo ""
    echo "Instructions:"
    echo "1. Connect your Android device via USB"
    echo "2. Enable 'Metrics' button in the app"
    echo "3. This script will capture CSV data to $OUTPUT_FILE"
    echo ""

    # Clear logcat buffer to start fresh
    adb logcat -c

    # Filter for metrics logs and save to file
    # The grep extracts only the CSV lines (header and data)
    adb logcat -s MainActivity:D | grep "METRICS_CSV" | while read -r line; do
        # Extract just the CSV part after "METRICS_CSV_HEADER: " or "METRICS_CSV_DATA: "
        if [[ "$line" =~ METRICS_CSV_HEADER:\ (.+)$ ]]; then
            echo "${BASH_REMATCH[1]}" | tee -a "$OUTPUT_FILE"
        elif [[ "$line" =~ METRICS_CSV_DATA:\ (.+)$ ]]; then
            echo "${BASH_REMATCH[1]}" | tee -a "$OUTPUT_FILE"
        fi
    done
