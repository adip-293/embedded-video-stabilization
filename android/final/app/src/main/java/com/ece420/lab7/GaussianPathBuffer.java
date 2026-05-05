package com.ece420.lab7;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.List;

//Gaussian smoothing buffer for camera path stabilization
public final class GaussianPathBuffer {

    private final int cap;
    private final ArrayList<Double> histTx = new ArrayList<>();
    private final ArrayList<Double> histTy = new ArrayList<>();
    private final ArrayList<Double> histTheta = new ArrayList<>();
    private final ArrayList<Double> histLogSc = new ArrayList<>();
    private final ArrayList<Mat> histM23 = new ArrayList<>();

    //Default constructor using PATH_HISTORY_CAP
    public GaussianPathBuffer() {
        this(PipelineUtils.PATH_HISTORY_CAP);
    }

    //Initialize buffer with specified max frame history
    public GaussianPathBuffer(int maxFrames) {
        this.cap = Math.max(8, maxFrames);
    }

    //Reset buffer and release all stored matrices
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
        double[] p = new double[4];
        PipelineUtils.similarityParamsFromR(pathR, p);
        histTx.add(p[0]);
        histTy.add(p[1]);
        histTheta.add(p[2]);
        histLogSc.add(p[3]);

        //Compute smoothed warp from history
        Mat warp23 = computeWarpFromHistory(pathR);
        
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

    //Remove oldest entries when history exceeds cap
    private void trim() {
        while (histTx.size() > cap) {
            histTx.remove(0);
            histTy.remove(0);
            histTheta.remove(0);
            histLogSc.remove(0);
            Mat old = histM23.remove(0);
            if (old != null) {
                old.release();
            }
        }
    }

    private Mat computeWarpFromHistory(Mat pathR) {
        int n = histTx.size();
        if (n == 0) {
            return null;
        }
        
        //Copy history to arrays for smoothing
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
        
        //Unwrap angles and apply Gaussian smoothing
        PipelineUtils.unwrapSeriesInPlace(th, n);
        double sigma = PipelineUtils.PATH_GAUSSIAN_SIGMA;
        
        //Reduce sigma if history is short to avoid edge padding issues
        //Kernel radius is ~3*sigma, so when n < 2*radius, reduce sigma proportionally
        int kernelRadius = (int)(3 * sigma);
        if (n < 2 * kernelRadius) {
            sigma = Math.max(1.0, n / 6.0);
        }
        
        tx = PipelineUtils.gaussianSmooth1d(tx, n, sigma);
        ty = PipelineUtils.gaussianSmooth1d(ty, n, sigma);
        th = PipelineUtils.gaussianSmooth1d(th, n, sigma);
        lg = PipelineUtils.gaussianSmooth1d(lg, n, sigma);
        
        //Build smoothed transform Q and compute warp M = Q * R^-1
        int last = n - 1;
        Mat q3 = PipelineUtils.buildQ3(tx[last], ty[last], th[last], lg[last]);
        Mat m23 = PipelineUtils.computeWarpM23FromRAndQ(pathR, q3);
        q3.release();
        return m23;
    }
}
