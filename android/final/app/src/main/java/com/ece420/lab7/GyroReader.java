package com.ece420.lab7;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

/**
 * Simplified gyroscope sensor reader based on lab1's SensorReader pattern.
 * Handles gyro registration and provides latest readings.
 */
public class GyroReader implements SensorEventListener {
    private static final String TAG = "GyroReader";

    private final SensorManager mSensorManager;
    private final Sensor mGyroscope;

    private float gyroX = 0f;
    private float gyroY = 0f;
    private float gyroZ = 0f;
    private boolean gyroReady = false;

    public GyroReader(Context context) {
        mSensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        mGyroscope = mSensorManager != null ? mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE) : null;
        
        if (mGyroscope == null) {
            Log.w(TAG, "Gyroscope sensor not available on this device");
        }
    }

    /**
     * Start listening to gyroscope sensor.
     */
    public void register() {
        if (mSensorManager != null && mGyroscope != null) {
            mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_GAME);
            Log.d(TAG, "Gyroscope registered");
        }
    }

    /**
     * Stop listening to gyroscope sensor.
     */
    public void unregister() {
        if (mSensorManager != null) {
            mSensorManager.unregisterListener(this);
            Log.d(TAG, "Gyroscope unregistered");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gyroX = event.values[0];
            gyroY = event.values[1];
            gyroZ = event.values[2];
            gyroReady = true;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not used
    }

    /**
     * Get the latest gyro X value (rad/s).
     */
    public float getGyroX() {
        return gyroX;
    }

    /**
     * Get the latest gyro Y value (rad/s).
     */
    public float getGyroY() {
        return gyroY;
    }

    /**
     * Get the latest gyro Z value (rad/s).
     */
    public float getGyroZ() {
        return gyroZ;
    }

    /**
     * Check if gyroscope is available and has provided at least one reading.
     */
    public boolean isReady() {
        return gyroReady && mGyroscope != null;
    }

    /**
     * Check if gyroscope sensor is available on this device.
     */
    public boolean isAvailable() {
        return mGyroscope != null;
    }
}
