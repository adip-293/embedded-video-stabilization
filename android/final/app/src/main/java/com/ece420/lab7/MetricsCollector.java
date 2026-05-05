package com.ece420.lab7;

import java.util.ArrayList;
import java.util.Locale;

public class MetricsCollector {
    
    //Frame-level metrics
    private int frameCount = 0;
    private int totalFeatures = 0;
    private int trackedFeatures = 0;
    private int ransacInliers = 0;
    private int ransacOutliers = 0;
    
    //Incremental motion tracking (frame-to-frame motion magnitude)
    private double rawMotionEnergy = 0.0;
    private double smoothMotionEnergy = 0.0;
    
    //Previous smoothed path parameters (to compute incremental smoothed motion)
    private double prevSmoothTx = 0.0;
    private double prevSmoothTy = 0.0;
    private boolean hasPrevSmooth = false;
    
    //Filter variance tracking - store incremental motion history
    private ArrayList<Double> rawIncTxHistory = new ArrayList<>();
    private ArrayList<Double> rawIncTyHistory = new ArrayList<>();
    private ArrayList<Double> smoothIncTxHistory = new ArrayList<>();
    private ArrayList<Double> smoothIncTyHistory = new ArrayList<>();
    private static final int VARIANCE_WINDOW = 30; //Track last 30 frames for variance
    
    //Motion gating stats
    private int gatedFrames = 0;
    
    //Crop efficiency
    private double cropRatio = 1.0;
    
    //Current filter in use
    private String currentFilter = "Kalman";
    
    public MetricsCollector() {
        reset();
    }
    
    public void reset() {
        frameCount = 0;
        totalFeatures = 0;
        trackedFeatures = 0;
        ransacInliers = 0;
        ransacOutliers = 0;
        rawMotionEnergy = 0.0;
        smoothMotionEnergy = 0.0;
        prevSmoothTx = 0.0;
        prevSmoothTy = 0.0;
        hasPrevSmooth = false;
        rawIncTxHistory.clear();
        rawIncTyHistory.clear();
        smoothIncTxHistory.clear();
        smoothIncTyHistory.clear();
        gatedFrames = 0;
        cropRatio = 1.0;
    }
    
    //Start a new frame
    public void startFrame() {
        totalFeatures = 0;
        trackedFeatures = 0;
        ransacInliers = 0;
        ransacOutliers = 0;
        cropRatio = 1.0;
    }
    
    //Record feature tracking results
    public void recordFeatureTracking(int tracked, int total) {
        this.trackedFeatures = tracked;
        this.totalFeatures = total;
    }
    
    //Record RANSAC results
    public void recordRANSAC(int inliers, int outliers) {
        this.ransacInliers = inliers;
        this.ransacOutliers = outliers;
    }
    
    //Record incremental motion from raw transform and smoothed path parameters
    public void recordMotionMetrics(double rawIncTx, double rawIncTy, double smoothPathTx, double smoothPathTy) {
        //Compute incremental smoothed motion (change from previous smoothed position)
        double smoothIncTx = 0.0;
        double smoothIncTy = 0.0;
        
        if (hasPrevSmooth) {
            smoothIncTx = smoothPathTx - prevSmoothTx;
            smoothIncTy = smoothPathTy - prevSmoothTy;
        }
        
        //Accumulate motion energy (squared magnitude of incremental motion)
        double rawMagnitude = Math.sqrt(rawIncTx * rawIncTx + rawIncTy * rawIncTy);
        double smoothMagnitude = hasPrevSmooth ? Math.sqrt(smoothIncTx * smoothIncTx + smoothIncTy * smoothIncTy) : 0.0;
        
        rawMotionEnergy += rawMagnitude;
        if (hasPrevSmooth) {
            smoothMotionEnergy += smoothMagnitude;
        }
        
        //Store incremental motions for variance calculation
        rawIncTxHistory.add(rawIncTx);
        rawIncTyHistory.add(rawIncTy);
        if (hasPrevSmooth) {
            smoothIncTxHistory.add(smoothIncTx);
            smoothIncTyHistory.add(smoothIncTy);
        }
        
        //Trim history to window size
        if (rawIncTxHistory.size() > VARIANCE_WINDOW) {
            rawIncTxHistory.remove(0);
            rawIncTyHistory.remove(0);
        }
        if (smoothIncTxHistory.size() > VARIANCE_WINDOW) {
            smoothIncTxHistory.remove(0);
            smoothIncTyHistory.remove(0);
        }
        
        //Update previous smoothed position for next frame
        prevSmoothTx = smoothPathTx;
        prevSmoothTy = smoothPathTy;
        hasPrevSmooth = true;
    }
    
    //Record current filter name
    public void recordFilter(String filterName) {
        this.currentFilter = filterName;
    }
    
    //Record crop efficiency
    public void recordCrop(double ratio) {
        this.cropRatio = ratio;
    }
    
    //Record motion gating event
    public void recordGating(boolean gated) {
        if (gated) {
            gatedFrames++;
        }
    }
    
    //Complete frame
    public void endFrame() {
        frameCount++;
    }
    
    //Compute variance of a list
    private double computeVariance(ArrayList<Double> data) {
        if (data.size() < 2) return 0.0;
        
        double mean = 0.0;
        for (double v : data) {
            mean += v;
        }
        mean /= data.size();
        
        double variance = 0.0;
        for (double v : data) {
            double diff = v - mean;
            variance += diff * diff;
        }
        variance /= data.size();
        
        return variance;
    }
    
    //Generate live display string
    public String getLiveSummary() {
        StringBuilder sb = new StringBuilder();
        
        //Feature tracking
        double trackingRate = totalFeatures > 0 ? (100.0 * trackedFeatures / totalFeatures) : 0.0;
        sb.append(String.format(Locale.US, "Track: %d/%d (%.0f%%)\n", 
                trackedFeatures, totalFeatures, trackingRate));
        
        //RANSAC
        int totalPoints = ransacInliers + ransacOutliers;
        double inlierRate = totalPoints > 0 ? (100.0 * ransacInliers / totalPoints) : 0.0;
        sb.append(String.format(Locale.US, "Inlier: %d/%d (%.0f%%)\n", 
                ransacInliers, totalPoints, inlierRate));
        
        //Motion energy (accumulated frame-to-frame translation magnitude)
        double reductionPercent = rawMotionEnergy > 0 ? (100.0 * (1.0 - smoothMotionEnergy / rawMotionEnergy)) : 0.0;
        sb.append(String.format(Locale.US, "Shake: %.1f → %.1f (%.0f%% ↓)\n", 
                rawMotionEnergy, smoothMotionEnergy, reductionPercent));
        
        //Filter variance reduction (if enough data)
        if (rawIncTxHistory.size() >= 10 && smoothIncTxHistory.size() >= 10) {
            //Compute variance of incremental tx motions
            double rawVar = computeVariance(rawIncTxHistory);
            double smoothVar = computeVariance(smoothIncTxHistory);
            double varReduction = rawVar > 0 ? (100.0 * (1.0 - smoothVar / rawVar)) : 0.0;
            sb.append(String.format(Locale.US, "%s: σ²=%.2f → %.2f (%.0f%% ↓)\n",
                    currentFilter, rawVar, smoothVar, varReduction));
        }
        
        //Crop efficiency
        sb.append(String.format(Locale.US, "Crop: %.2f%%", cropRatio * 100.0));
        
        return sb.toString();
    }
    
    //Generate CSV row for data export (logs to logcat)
    public String getCSVRow() {
        //Calculate derived metrics
        double trackingRate = totalFeatures > 0 ? (100.0 * trackedFeatures / totalFeatures) : 0.0;
        int totalPoints = ransacInliers + ransacOutliers;
        double inlierRate = totalPoints > 0 ? (100.0 * ransacInliers / totalPoints) : 0.0;
        double shakeReduction = rawMotionEnergy > 0 ? (100.0 * (1.0 - smoothMotionEnergy / rawMotionEnergy)) : 0.0;
        
        double rawVar = computeVariance(rawIncTxHistory);
        double smoothVar = computeVariance(smoothIncTxHistory);
        double varReduction = rawVar > 0 ? (100.0 * (1.0 - smoothVar / rawVar)) : 0.0;
        
        //CSV format: frame,filter,tracked,total,trackRate,inliers,totalPts,inlierRate,rawShake,smoothShake,shakeReduct,rawVar,smoothVar,varReduct,cropPct
        return String.format(Locale.US, "%d,%s,%d,%d,%.2f,%d,%d,%.2f,%.4f,%.4f,%.2f,%.6f,%.6f,%.2f,%.4f",
                frameCount, currentFilter,
                trackedFeatures, totalFeatures, trackingRate,
                ransacInliers, totalPoints, inlierRate,
                rawMotionEnergy, smoothMotionEnergy, shakeReduction,
                rawVar, smoothVar, varReduction,
                cropRatio * 100.0);
    }
}
