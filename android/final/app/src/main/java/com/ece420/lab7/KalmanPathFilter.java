package com.ece420.lab7;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.ArrayList;

//Kalman filter for camera path stabilization
public final class KalmanPathFilter {
    
    private final int cap;
    private final ArrayList<Mat> histM23 = new ArrayList<>();
    
    //Kalman filter state (tx, ty, theta, log scale)
    private double[] state = new double[4];
    private boolean initialized;
    private double lastTheta;
    
    //Kalman gain (how much to trust new measurements vs. previous estimate)
    //Higher K = more responsive to new measurements, less smoothing
    //Lower K = more smoothing, less responsive
    private static final double KALMAN_GAIN = 0.3;
    
    //Default constructor using PATH_HISTORY_CAP
    public KalmanPathFilter() {
        this(PipelineUtils.PATH_HISTORY_CAP);
    }
    
    //Initialize filter with specified max frame history
    public KalmanPathFilter(int maxFrames) {
        this.cap = Math.max(8, maxFrames);
        this.initialized = false;
        this.lastTheta = 0.0;
        this.state = new double[]{0.0, 0.0, 0.0, 0.0};
    }
    
    //Reset filter state and release all matrices
    public void clear() {
        for (Mat m : histM23) {
            if (m != null) {
                m.release();
            }
        }
        histM23.clear();
        initialized = false;
        lastTheta = 0.0;
        state = new double[]{0.0, 0.0, 0.0, 0.0};
    }
    
    public Mat appendAndComputeWarp(Mat pathR) {
        if (pathR == null || pathR.empty()) {
            return null;
        }
        
        //Extract similarity parameters (tx, ty, theta, log scale)
        double[] measurement = new double[4];
        PipelineUtils.similarityParamsFromR(pathR, measurement);
        
        //Unwrap theta to avoid discontinuities
        measurement[2] = PipelineUtils.unwrapNear(measurement[2], lastTheta);
        lastTheta = measurement[2];
        
        //Initialize on first frame
        if (!initialized) {
            //Set initial state to first measurement
            state[0] = measurement[0];
            state[1] = measurement[1];
            state[2] = measurement[2];
            state[3] = measurement[3];
            initialized = true;
            
            //Store identity warp for first frame
            histM23.add(PipelineUtils.identityAffine23());
            return PipelineUtils.identityAffine23();
        }
        
        //Kalman filter prediction step: x̂_k = x_{k-1}
        //Predicted state is simply the previous state (constant velocity model with zero velocity)
        double[] predictedState = new double[4];
        predictedState[0] = state[0];
        predictedState[1] = state[1];
        predictedState[2] = state[2];
        predictedState[3] = state[3];
        
        //Kalman filter update step: x_k = x_{k-1} + K(z_k - x_{k-1})
        //K is the Kalman gain, z_k is the measurement, (z_k - x_{k-1}) is the residual
        for (int i = 0; i < 4; i++) {
            double residual = measurement[i] - predictedState[i];
            state[i] = predictedState[i] + KALMAN_GAIN * residual;
        }
        
        //Build smoothed transform Q and compute warp M = Q * R^-1
        Mat q3 = PipelineUtils.buildQ3(state[0], state[1], state[2], state[3]);
        Mat warp23 = PipelineUtils.computeWarpM23FromRAndQ(pathR, q3);
        q3.release();
        
        //Store warp matrix for crop history
        if (warp23 != null && !warp23.empty()) {
            Mat c = new Mat();
            warp23.copyTo(c);
            histM23.add(c);
        } else {
            histM23.add(PipelineUtils.identityAffine23());
        }
        
        //Trim history to cap
        trim();
        return warp23;
    }
    
    //Compute stable crop rectangle from warp history
    public Rect clampedCropRect(int frameW, int frameH) {
        //Use larger history window for more stable crop (last 150 frames or all)
        int n = histM23.size();
        if (n == 0) {
            return new Rect(0, 0, frameW, frameH);
        }
        int start = Math.max(0, n - 150);
        ArrayList<Mat> recent = new ArrayList<>();
        for (int i = start; i < n; i++) {
            recent.add(histM23.get(i));
        }
        return PipelineUtils.clampRectToImage(PipelineUtils.cropBoxFromMs(recent, frameW, frameH), frameW, frameH);
    }
    
    //Remove oldest matrices when history exceeds cap
    private void trim() {
        while (histM23.size() > cap) {
            Mat old = histM23.remove(0);
            if (old != null) {
                old.release();
            }
        }
    }
}
