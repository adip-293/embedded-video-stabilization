package com.ece420.lab7;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.ArrayList;

//Exponential moving average (EMA) filter for camera path stabilization
//Uses exponentially weighted average with more weight on recent frames
public final class ExponentialMovingAveragePathBuffer {
    
    private final int cap;
    private final double alpha; //Smoothing factor (0 < alpha <= 1)
    
    //History of cumulative path parameters
    private final ArrayList<Double> histTx = new ArrayList<>();
    private final ArrayList<Double> histTy = new ArrayList<>();
    private final ArrayList<Double> histTheta = new ArrayList<>();
    private final ArrayList<Double> histLogSc = new ArrayList<>();
    private final ArrayList<Mat> histM23 = new ArrayList<>();
    
    //Default constructor using PATH_HISTORY_CAP and alpha=0.5 (medium smoothing)
    public ExponentialMovingAveragePathBuffer() {
        this(PipelineUtils.PATH_HISTORY_CAP, 0.5);
    }
    
    //Initialize filter with specified max frame history and smoothing factor
    //alpha: 0.1 = heavy smoothing, 0.5 = medium, 0.9 = light smoothing (more responsive)
    public ExponentialMovingAveragePathBuffer(int maxFrames, double smoothingFactor) {
        this.cap = Math.max(8, maxFrames);
        this.alpha = Math.max(0.05, Math.min(smoothingFactor, 1.0));
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
        
        //Apply exponential smoothing to trajectory history
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
        
        //Apply exponential smoothing (forward pass)
        tx = exponentialSmooth(tx, n, alpha);
        ty = exponentialSmooth(ty, n, alpha);
        th = exponentialSmooth(th, n, alpha);
        lg = exponentialSmooth(lg, n, alpha);
        
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
    
    //Apply exponential smoothing to 1D signal
    //Uses exponentially decaying weights favoring recent values
    private double[] exponentialSmooth(double[] signal, int len, double alpha) {
        double[] smoothed = new double[len];
        
        if (len == 0) return smoothed;
        
        //Initialize with first value
        smoothed[0] = signal[0];
        
        //Apply EMA: s[i] = alpha * signal[i] + (1 - alpha) * s[i-1]
        for (int i = 1; i < len; i++) {
            smoothed[i] = alpha * signal[i] + (1.0 - alpha) * smoothed[i - 1];
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
