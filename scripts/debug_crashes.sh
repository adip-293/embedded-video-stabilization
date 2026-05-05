#!/bin/bash
# Capture Android crash logs for debugging
# Usage: ./debug_crashes.sh

echo "========================================"
echo "Android Crash Logger"
echo "========================================"
echo "Monitoring for crashes and errors..."
echo "Press Ctrl+C to stop"
echo ""

# Clear logcat buffer
adb logcat -c

# Monitor for crashes, errors, and exceptions
# Shows crashes, native crashes, ANRs, and Java exceptions
adb logcat -v time "*:E" "AndroidRuntime:E" "DEBUG:I" "MainActivity:D" "System.err:W" | \
    grep --line-buffered -E "(FATAL|AndroidRuntime|Exception|Error|at com.ece420|MainActivity)" | \
    tee crash_log_$(date +%Y%m%d_%H%M%S).txt
