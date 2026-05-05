package com.ece420.lab7;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.ArrayList;

//Uniform moving average (UMA) filter for camera path stabilization (baseline)
//Uses simple arithmetic mean of recent frames with equal weights
public final class UniformMovingAveragePathBuffer {
    
    private final int cap;
    
    //History of cumulative path parameters
    private final ArrayList<Double> histTx = new ArrayList<>();
    private final ArrayList<Double> histTy = new ArrayList<>();
    private final ArrayList<Double> histTheta = new ArrayList<>();
    private final ArrayList<Double> histLogSc = new ArrayList<>();
    private final ArrayList<Mat> histM23 = new ArrayList<>();
    
    //Default constructor using PATH_HISTORY_CAP
    public UniformMovingAveragePathBuffer() {
        this(PipelineUtils.PATH_HISTORY_CAP);
    }
    
    //Initialize filter with specified max frame history
    public UniformMovingAveragePathBuffer(int maxFrames) {
        this.cap = Math.max(8, maxFrames);
    }
    
    //Reset filter state and release all matrices
    public void clear() {
        histTx.clear();
        histTy.clear();
        histTheta.clear();
        histLogSc.clear();
        for (Mat m : histM23) {
            if (m != null) {
                m.release();
            }
        }
        histM23.clear();
    }
    
    public Mat appendAndComputeWarp(Mat pathR) {
        if (pathR == null || pathR.empty()) {
            return null;
        }
        
        //Extract similarity parameters (tx, ty, theta, log scale)
        double[] params = new double[4];
        PipelineUtils.similarityParamsFromR(pathR, params);
        
        //Store raw parameters in history
        histTx.add(params[0]);
        histTy.add(params[1]);
        histTheta.add(params[2]);
        histLogSc.add(params[3]);
        
        //Trim if exceeded capacity
        if (histTx.size() > cap) {
            histTx.remove(0);
            histTy.remove(0);
            histTheta.remove(0);
            histLogSc.remove(0);
        }
        
        //Apply uniform moving average smoothing to trajectory history
        int n = histTx.size();
        double[] tx = new double[n];
        double[] ty = new double[n];
        double[] th = new double[n];
        double[] lg = new double[n];
        
        for (int i = 0; i < n; i++) {
            tx[i] = histTx.get(i);
            ty[i] = histTy.get(i);
            th[i] = histTheta.get(i);
            lg[i] = histLogSc.get(i);
        }
        
        //Unwrap angles before smoothing
        PipelineUtils.unwrapSeriesInPlace(th, n);
        
        //Apply uniform (rectangular window) smoothing
        //Use smaller window (5 frames) for baseline - less aggressive smoothing
        int windowSize = Math.min(5, n);
        tx = uniformSmooth(tx, n, windowSize);
        ty = uniformSmooth(ty, n, windowSize);
        th = uniformSmooth(th, n, windowSize);
        lg = uniformSmooth(lg, n, windowSize);
        
        //Build smoothed transform Q from last smoothed value
        int last = n - 1;
        Mat q3 = PipelineUtils.buildQ3(tx[last], ty[last], th[last], lg[last]);
        
        //Compute stabilization warp: M = Q * R^-1
        Mat warp23 = PipelineUtils.computeWarpM23FromRAndQ(pathR, q3);
        q3.release();
        
        //Store warp in history
        histM23.add(warp23.clone());
        if (histM23.size() > cap) {
            Mat old = histM23.remove(0);
            if (old != null) old.release();
        }
        
        return warp23;
    }
    
    //Apply uniform moving average smoothing to 1D signal
    //Uses rectangular window (equal weights for all values in window)
    private double[] uniformSmooth(double[] signal, int len, int windowSize) {
        double[] smoothed = new double[len];
        int halfWindow = windowSize / 2;
        
        for (int i = 0; i < len; i++) {
            //Determine window bounds
            int start = Math.max(0, i - halfWindow);
            int end = Math.min(len - 1, i + halfWindow);
            
            //Compute mean over window
            double sum = 0.0;
            int count = 0;
            for (int j = start; j <= end; j++) {
                sum += signal[j];
                count++;
            }
            
            smoothed[i] = sum / count;
        }
        
        return smoothed;
    }
    
    //Compute stable crop rectangle from warp history
    public Rect clampedCropRect(int frameW, int frameH) {
        //Use recent warp history to compute crop region
        int n = histM23.size();
        if (n == 0) {
            return new Rect(0, 0, frameW, frameH);
        }
        
        //Use last 150 frames or all available (increased for stability)
        int start = Math.max(0, n - 150);
        ArrayList<Mat> recent = new ArrayList<>();
        for (int i = start; i < n; i++) {
            recent.add(histM23.get(i));
        }
        
        return PipelineUtils.cropBoxFromMs(recent, frameW, frameH);
    }
    
    //Remove oldest frame from history
    public void trim() {
        if (!histTx.isEmpty()) {
            histTx.remove(0);
            histTy.remove(0);
            histTheta.remove(0);
            histLogSc.remove(0);
        }
        if (!histM23.isEmpty()) {
            Mat old = histM23.remove(0);
            if (old != null) old.release();
        }
    }
}
