package com.ece420.lab7;

import android.app.Activity;
import android.Manifest;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.Locale;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    /** Max stage index for seek bar (inclusive). */
    private static final int STAGE_MAX = 6;

    /** Default seek position on launch (1 = gyro HUD). */
    private static final int DEFAULT_STAGE_INDEX = 1;

    // UI components
    private TextView pipelineTitle;
    private SeekBar pipelineSeekBar;
    private String[] pipelineStages;
    private CameraBridgeViewBase mOpenCvCameraView;
    private Button liveButton;
    private Button motionGatingButton;
    private Button metricsButton;
    private Spinner filterSpinner;
    private TextView metricsTextView;
    private boolean isLive = false;
    private boolean motionGatingEnabled = false;
    private boolean metricsEnabled = false;
    private boolean csvHeaderLogged = false;
    private ImageView processedImageView;

    // OpenCV matrices
    private Mat mRgba;
    private Mat mProcessed;
    private Bitmap processedBitmap;

    // Pipeline state
    private Mat prevGray;
    private MatOfPoint2f prevPts;
    private Mat pathR;
    private final UniformMovingAveragePathBuffer uniformFilter = new UniformMovingAveragePathBuffer();
    private final ExponentialMovingAveragePathBuffer exponentialFilter = new ExponentialMovingAveragePathBuffer();
    private final GaussianPathBuffer gaussianFilter = new GaussianPathBuffer();
    private final KalmanPathFilter kalmanFilter = new KalmanPathFilter();
    private final MetricsCollector metrics = new MetricsCollector();
    private Mat warpFull;

    // Gyro sensor (simplified using lab1 pattern)
    private GyroReader gyroReader;

    /** Monotonic time of last temporal pipeline frame (for gyro integration). */
    private long lastPipelineFrameNs;

    private volatile int currentStageIndex = DEFAULT_STAGE_INDEX;

    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    if (isLive && mOpenCvCameraView != null) {
                        mOpenCvCameraView.enableView();
                    }
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        setContentView(R.layout.activity_main);

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCVLoader.initDebug() failed");
        } else {
            Log.d(TAG, "OpenCVLoader.initDebug() OK");
        }

        // Request camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] { Manifest.permission.CAMERA }, 1);
        }

        // Initialize gyro sensor using lab1 pattern
        gyroReader = new GyroReader(this);

        // Setup UI components
        pipelineTitle = findViewById(R.id.pipelineTitle);
        pipelineSeekBar = findViewById(R.id.pipelineSeekBar);
        liveButton = findViewById(R.id.liveButton);
        motionGatingButton = findViewById(R.id.motionGatingButton);
        metricsButton = findViewById(R.id.metricsButton);
        filterSpinner = findViewById(R.id.filterSpinner);
        metricsTextView = findViewById(R.id.metricsTextView);
        mOpenCvCameraView = findViewById(R.id.opencv_camera_preview);
        processedImageView = findViewById(R.id.processedImageView);

        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.setCameraIndex(0);
            mOpenCvCameraView.setCvCameraViewListener(this);
        }

        // Setup pipeline stage names
        pipelineStages = new String[] {
                getString(R.string.stage_raw),
                getString(R.string.stage_gyro),
                getString(R.string.stage_feature),
                getString(R.string.stage_lk),
                getString(R.string.stage_ransac_affine),
                getString(R.string.stage_kalman),
                getString(R.string.stage_warp),
        };

        pipelineSeekBar.setMax(STAGE_MAX);
        pipelineSeekBar.setProgress(DEFAULT_STAGE_INDEX);

        pipelineSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (progress >= 0 && progress <= STAGE_MAX) {
                    if (currentStageIndex != progress) {
                        resetTrackingState();
                    }
                    pipelineTitle.setText(pipelineStages[progress]);
                    currentStageIndex = progress;
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        syncPipelineTitleToSeekBar();

        liveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                isLive = true;
                //Flush pipeline state to remove stale tracking data
                resetTrackingState();
                if (mOpenCvCameraView != null) {
                    mOpenCvCameraView.enableView();
                }
            }
        });
        
        //Setup filter spinner with dropdown options
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
                R.array.filter_options, R.layout.spinner_item);
        adapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
        filterSpinner.setAdapter(adapter);
        filterSpinner.setSelection(0); //Default to UMA (baseline)
        
        //Motion gating button click listener toggles state and color
        motionGatingButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                motionGatingEnabled = !motionGatingEnabled;
                if (motionGatingEnabled) {
                    motionGatingButton.setBackgroundColor(getResources().getColor(R.color.catppuccin_orange));
                    motionGatingButton.setTextColor(getResources().getColor(R.color.catppuccin_background));
                } else {
                    motionGatingButton.setBackgroundColor(getResources().getColor(R.color.catppuccin_surface));
                    motionGatingButton.setTextColor(getResources().getColor(R.color.catppuccin_text_muted));
                }
            }
        });
        
        //Metrics button click listener toggles state and color
        metricsButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                metricsEnabled = !metricsEnabled;
                if (metricsEnabled) {
                    metricsButton.setBackgroundColor(getResources().getColor(R.color.catppuccin_blue));
                    metricsButton.setTextColor(getResources().getColor(R.color.catppuccin_background));
                    metricsTextView.setVisibility(View.VISIBLE);
                    metrics.reset();
                    csvHeaderLogged = false; //Reset for new session
                } else {
                    metricsButton.setBackgroundColor(getResources().getColor(R.color.catppuccin_surface));
                    metricsButton.setTextColor(getResources().getColor(R.color.catppuccin_text_muted));
                    metricsTextView.setVisibility(View.GONE);
                }
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        gyroReader.register();
        
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV initAsync (Manager)");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV package library OK");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Unregister gyro sensor
        gyroReader.unregister();
        
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    private void syncPipelineTitleToSeekBar() {
        int p = pipelineSeekBar.getProgress();
        if (p < 0 || p > STAGE_MAX) {
            p = DEFAULT_STAGE_INDEX;
            pipelineSeekBar.setProgress(p);
        }
        pipelineTitle.setText(pipelineStages[p]);
        currentStageIndex = p;
    }

    private void resetTrackingState() {
        if (prevGray != null) prevGray.release();
        prevGray = new Mat();
        if (prevPts != null) prevPts.release();
        prevPts = new MatOfPoint2f();
        if (pathR != null) pathR.release();
        pathR = Mat.eye(3, 3, CvType.CV_64F);
        uniformFilter.clear();
        exponentialFilter.clear();
        gaussianFilter.clear();
        kalmanFilter.clear();
        lastPipelineFrameNs = 0;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.d(TAG, "onCameraViewStarted " + width + "x" + height);
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mProcessed = new Mat(height, width, CvType.CV_8UC4);
        processedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        warpFull = new Mat(height, width, CvType.CV_8UC4);
        prevGray = new Mat();
        prevPts = new MatOfPoint2f();
        if (pathR != null) {
            pathR.release();
        }
        pathR = Mat.eye(3, 3, CvType.CV_64F);
        uniformFilter.clear();
        exponentialFilter.clear();
        gaussianFilter.clear();
        kalmanFilter.clear();
        lastPipelineFrameNs = 0;
    }

    @Override
    public void onCameraViewStopped() {
        Log.d(TAG, "onCameraViewStopped");
        if (mRgba != null) {
            mRgba.release();
            mRgba = null;
        }
        if (mProcessed != null) {
            mProcessed.release();
            mProcessed = null;
        }
        processedBitmap = null;
        if (prevGray != null) {
            prevGray.release();
            prevGray = null;
        }
        if (prevPts != null) {
            prevPts.release();
            prevPts = null;
        }
        if (pathR != null) {
            pathR.release();
            pathR = null;
        }
        if (warpFull != null) {
            warpFull.release();
            warpFull = null;
        }
        uniformFilter.clear();
        exponentialFilter.clear();
        gaussianFilter.clear();
        kalmanFilter.clear();
        lastPipelineFrameNs = 0;
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if (mRgba == null || mProcessed == null) {
            return inputFrame.rgba();
        }
        mRgba = inputFrame.rgba();
        runPipeline(mRgba, mProcessed, currentStageIndex);
        
        if (processedBitmap != null && processedImageView != null) {
            Utils.matToBitmap(mProcessed, processedBitmap);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    processedImageView.setImageBitmap(processedBitmap);
                }
            });
        }
        return mRgba;
    }

    private void runPipeline(Mat input, Mat output, int stage) {
        // ---------------------------------------------------------------------
        // Stage 0: Raw Passthrough
        // ---------------------------------------------------------------------
        if (stage == 0) {
            //Copy input to output
            input.copyTo(output);
            return;
        }

        // ---------------------------------------------------------------------
        // Stage 1: Gyro Motion Gating
        // ---------------------------------------------------------------------
        if (stage == 1) {
            //Copy input to output
            input.copyTo(output);

            //Harvest gyro data
            float gx = gyroReader.getGyroX();
            float gy = gyroReader.getGyroY();
            float gz = gyroReader.getGyroZ();
            boolean ready = gyroReader.isReady();
            
            String printLine;

            //Display gyro status and values on output frame
            if (!gyroReader.isAvailable()) {
                printLine = "Gyro: not available";
            } else if (!ready) {
                printLine = "Gyro: waiting…";
            } else {
                float norm = PipelineUtils.gyroNormRadS(gx, gy, gz);
                boolean gate = norm > PipelineUtils.GYRO_GATE_NORM_RAD_S;
                printLine = String.format(Locale.US, "|w|=%.2f rad/s  gate %s",
                        norm, gate ? "ON" : "off");
            }
            Imgproc.putText(output, printLine, new Point(16, 44), Core.FONT_HERSHEY_SIMPLEX, 0.6,
                    new Scalar(0, 255, 255, 255), 2, Core.LINE_AA, false);
            Imgproc.putText(output, "Stages 3–6: pre-warp + gate when |w| high", new Point(16, 84),
                    Core.FONT_HERSHEY_SIMPLEX, 0.45, new Scalar(220, 220, 220, 255), 1, Core.LINE_AA, false);
            return;
        }

        // ---------------------------------------------------------------------
        // Stage 2: Shi-Tomasi Corners
        // ---------------------------------------------------------------------
        if (stage == 2) {
            //Create matrix for grayscale image
            Mat gray = new Mat();

            //Convert input frame to grayscale
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY);

            //Detect corners via Shi-Tomasi & copy input frame to output
            MatOfPoint2f corners = PipelineUtils.goodFeaturesToPoint2f(gray);
            input.copyTo(output);

            //Use Shi-Tomasi Corners to draw green circles on output frame
            for (Point p : corners.toArray()) {
                Imgproc.circle(output, p, 4, new Scalar(0, 255, 0, 255), 2);
            }

            //Free temporary matrices and return
            gray.release();
            corners.release();
            return;
        }

        // =====================================================================
        // Stages 3–6: Temporal Chain with Motion Gating
        // =====================================================================
        
        //Start metrics collection for this frame
        if (metricsEnabled) {
            metrics.startFrame();
        }
        
        //Create matrix for grayscale image
        Mat gray = new Mat();
        
        //Convert input frame to grayscale
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY);

        //Initialize on first frame (let first frame be junk)
        if (prevGray.empty()) {
            //Save current frame as previous for next iteration
            gray.copyTo(prevGray);
            
            //Detect initial feature points
            prevPts = PipelineUtils.goodFeaturesToPoint2f(gray);
            
            //Copy input to output and return
            input.copyTo(output);
            gray.release();
            return;
        }

        //Check gyro sensor to determine if motion gating is needed
        float gyroNorm = 0f;
        if (gyroReader.isReady()) {
            //Harvest gyro data
            float gx = gyroReader.getGyroX();
            float gy = gyroReader.getGyroY();
            float gz = gyroReader.getGyroZ();
            
            //Compute gyro norm (magnitude of angular velocity)
            gyroNorm = PipelineUtils.gyroNormRadS(gx, gy, gz);
        }
        
        //Activate motion gate if gyro norm exceeds threshold
        boolean motionGate = gyroNorm > PipelineUtils.GYRO_GATE_NORM_RAD_S;
        
        //Record gating event
        if (metricsEnabled) {
            metrics.recordGating(motionGate);
        }
        
        //Check if motion gating is enabled and we should skip processing
        if (motionGatingEnabled && motionGate) {
            //Motion gate active and enabled: skip all processing, just pass through
            input.copyTo(output);
            gray.release();
            return;
        }

        //Check if we have features to track (prevPts might be empty if no features detected)
        if (prevPts.empty() || prevPts.toArray().length == 0) {
            //No features to track, re-detect and return
            prevPts.release();
            prevPts = PipelineUtils.goodFeaturesToPoint2f(gray);
            prevGray.release();
            gray.copyTo(prevGray);
            input.copyTo(output);
            gray.release();
            return;
        }

        //Run Lucas-Kanade optical flow to track features from previous frame to current
        MatOfPoint2f currPts = new MatOfPoint2f();
        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();
        Video.calcOpticalFlowPyrLK(prevGray, gray, prevPts, currPts, status, err);

        //Extract points and status arrays from OpenCV matrices
        Point[] p0 = prevPts.toArray();
        Point[] p1 = currPts.toArray();
        byte[] st = status.toArray();

        //Check if optical flow tracking returned valid results
        if (st.length == 0 || p0.length == 0 || p1.length == 0) {
            //No features to track, reset and return passthrough
            status.release();
            err.release();
            currPts.release();
            gray.release();
            resetTrackingState();
            input.copyTo(output);
            return;
        }

        //Filter to keep only successfully tracked points
        ArrayList<Point> lkPrev = new ArrayList<>();
        ArrayList<Point> lkNext = new ArrayList<>();
        for (int i = 0; i < st.length; i++) {
            //Status byte of 1 means point was successfully tracked
            if (st[i] == 1) {
                lkPrev.add(p0[i]);
                lkNext.add(p1[i]);
            }
        }

        //Count tracked points
        int n = lkPrev.size();
        
        //Record feature tracking metrics
        if (metricsEnabled) {
            metrics.recordFeatureTracking(n, p0.length);
        }
        
        Mat inc23 = null;
        Mat inliers = null;

        //Estimate incremental affine transform using RANSAC (needs at least 4 points)
        if (n >= 4) {
            //Convert point lists to OpenCV matrices
            MatOfPoint2f fromM = new MatOfPoint2f(lkPrev.toArray(new Point[0]));
            MatOfPoint2f toM = new MatOfPoint2f(lkNext.toArray(new Point[0]));
            
            //Run RANSAC to find best similarity transform (tx, ty, rotation, scale)
            inliers = new Mat();
            inc23 = Calib3d.estimateAffinePartial2D(fromM, toM, inliers, Calib3d.RANSAC,
                    PipelineUtils.RANSAC_REPROJ_THRESH_PX);
            
            //Free temporary matrices
            fromM.release();
            toM.release();
            
            //Check if estimation failed
            if (inc23.empty()) {
                inc23.release();
                inc23 = null;
            } else {
                //Hybrid rigid/similarity model: check if scale is close to 1.0
                //Extract scale from similarity transform
                double a = inc23.get(0, 0)[0];
                double b = inc23.get(0, 1)[0];
                double scale = Math.sqrt(a * a + b * b);
                
                //If scale is negligible, constrain to rigid (rotation only, no scale)
                final double SCALE_THRESHOLD = 0.02;
                if (Math.abs(scale - 1.0) < SCALE_THRESHOLD) {
                    //Convert to rigid transform by normalizing rotation components
                    double theta = Math.atan2(b, a);
                    double cosTheta = Math.cos(theta);
                    double sinTheta = Math.sin(theta);
                    
                    //Rebuild transform with unit scale
                    inc23.put(0, 0, cosTheta);
                    inc23.put(0, 1, -sinTheta);
                    inc23.put(1, 0, sinTheta);
                    inc23.put(1, 1, cosTheta);
                    //Translation components (0,2) and (1,2) remain unchanged
                }
            }
        }
        
        //Record RANSAC metrics
        if (metricsEnabled && inliers != null && !inliers.empty()) {
            int inlierCount = 0;
            for (int i = 0; i < inliers.rows(); i++) {
                byte[] bit = new byte[1];
                inliers.get(i, 0, bit);
                if (bit[0] != 0) inlierCount++;
            }
            metrics.recordRANSAC(inlierCount, n - inlierCount);
        }

        //Update cumulative camera path and compute stabilization warp
        Mat warp23;
        if (motionGatingEnabled && motionGate) {
            //Motion gate active: use identity transform (no stabilization during fast motion)
            warp23 = PipelineUtils.identityAffine23();
        } else {
            //Convert incremental affine to 3x3 homogeneous matrix
            Mat A3 = PipelineUtils.partialAffineTo3x3(inc23);
            
            //Update cumulative path: pathR = pathR * A3
            Mat Rnew = new Mat(3, 3, CvType.CV_64F);
            Core.gemm(pathR, A3, 1.0, new Mat(), 0, Rnew, 0);
            pathR.release();
            pathR = Rnew;
            A3.release();
            
            //Compute smoothed warp using selected filter
            int filterIndex = filterSpinner != null ? filterSpinner.getSelectedItemPosition() : 0;
            if (filterIndex == 0) {
                warp23 = uniformFilter.appendAndComputeWarp(pathR);
            } else if (filterIndex == 1) {
                warp23 = exponentialFilter.appendAndComputeWarp(pathR);
            } else if (filterIndex == 2) {
                warp23 = gaussianFilter.appendAndComputeWarp(pathR);
            } else {
                warp23 = kalmanFilter.appendAndComputeWarp(pathR);
            }
            
            //Fallback to identity if smoothing failed
            if (warp23 == null || warp23.empty()) {
                warp23 = PipelineUtils.identityAffine23();
            }
        }
        
        //Record metrics for this frame
        if (metricsEnabled && pathR != null && !pathR.empty()) {
            //Record current filter
            int filterIndex = filterSpinner != null ? filterSpinner.getSelectedItemPosition() : 0;
            String filterName;
            if (filterIndex == 0) {
                filterName = "UMA";
            } else if (filterIndex == 1) {
                filterName = "EMA";
            } else if (filterIndex == 2) {
                filterName = "Gaussian";
            } else {
                filterName = "Kalman";
            }
            metrics.recordFilter(filterName);
            
            //Extract raw incremental motion from inc23 (camera motion this frame)
            double rawIncTx = 0.0;
            double rawIncTy = 0.0;
            if (inc23 != null && !inc23.empty()) {
                rawIncTx = inc23.get(0, 2)[0];
                rawIncTy = inc23.get(1, 2)[0];
            }
            
            //Compute smoothed cumulative path: Q = warp23 * pathR
            Mat warp3 = PipelineUtils.partialAffineTo3x3(warp23);
            Mat smoothPathR = new Mat();
            Core.gemm(warp3, pathR, 1.0, new Mat(), 0, smoothPathR, 0);
            
            //Extract translation from smoothed path
            double[] smoothParams = new double[4];
            PipelineUtils.similarityParamsFromR(smoothPathR, smoothParams);
            double smoothPathTx = smoothParams[0];
            double smoothPathTy = smoothParams[1];
            
            smoothPathR.release();
            warp3.release();
            
            //Record motion metrics (collector will compute incremental smoothed motion internally)
            metrics.recordMotionMetrics(rawIncTx, rawIncTy, smoothPathTx, smoothPathTy);
        }

        //Define colors for visualization
        final Scalar green = new Scalar(0, 255, 0, 255);
        final Scalar red = new Scalar(255, 0, 0, 255);

        // ---------------------------------------------------------------------
        // Stage 3: Lucas-Kanade Optical Flow Visualization
        // ---------------------------------------------------------------------
        if (stage == 3) {
            //Copy input to output
            input.copyTo(output);
            
            //Draw all successfully tracked features as green arrows
            for (int i = 0; i < n; i++) {
                //Compute arrow tip with display scaling
                Point tip = PipelineUtils.lkArrowTipForDisplay(lkPrev.get(i), lkNext.get(i));
                if (tip != null) {
                    //Draw green arrow from previous position to current position
                    Imgproc.arrowedLine(output, lkPrev.get(i), tip, green, PipelineUtils.LK_ARROW_THICKNESS);
                }
            }
        } 

        // ---------------------------------------------------------------------
        // Stage 4: RANSAC Inlier/Outlier Visualization
        // ---------------------------------------------------------------------
        else if (stage == 4) {
            //Copy input to output
            input.copyTo(output);
            
            //Check if RANSAC estimation succeeded
            if (inc23 == null || inliers == null || inliers.empty() || inliers.rows() < n) {
                //RANSAC failed: draw all tracks as green
                for (int i = 0; i < n; i++) {
                    Point tip = PipelineUtils.lkArrowTipForDisplay(lkPrev.get(i), lkNext.get(i));
                    if (tip != null) {
                        Imgproc.arrowedLine(output, lkPrev.get(i), tip, green, PipelineUtils.LK_ARROW_THICKNESS);
                    }
                }
            } else {
                //RANSAC succeeded: color tracks by inlier status
                for (int i = 0; i < n; i++) {
                    Point tip = PipelineUtils.lkArrowTipForDisplay(lkPrev.get(i), lkNext.get(i));
                    if (tip != null) {
                        //Read inlier bit from RANSAC mask
                        byte[] bit = new byte[1];
                        inliers.get(i, 0, bit);
                        
                        //Green for inliers, red for outliers
                        Scalar c = (bit[0] != 0) ? green : red;
                        Imgproc.arrowedLine(output, lkPrev.get(i), tip, c, PipelineUtils.LK_ARROW_THICKNESS);
                    }
                }
            }
        } 
         // ---------------------------------------------------------------------
        // Stage 5: Motion Grid Visualization
        // ---------------------------------------------------------------------
        else if (stage == 5) {
            //Copy input to output
            input.copyTo(output);
            
            //Draw uniform grid of arrows showing stabilization warp field
            if (warp23 != null && !warp23.empty()) {
                //Draw yellow arrows on grid showing motion compensation
                PipelineUtils.drawEvenGridMotionArrows(output, warp23);
            }
           
        } 
         // ---------------------------------------------------------------------
        // Stage 6: Final Stabilized Output
        // ---------------------------------------------------------------------
        else {
            //Apply stabilization warp if available
            if (warp23 != null && !warp23.empty() && warpFull != null) {
                //Warp input frame using smoothed affine transform
                Imgproc.warpAffine(input, warpFull, warp23, input.size(),
                        Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);
                
                //Compute stable crop region to remove black borders
                int filterIndex = filterSpinner != null ? filterSpinner.getSelectedItemPosition() : 0;
                Rect roi;
                if (filterIndex == 0) {
                    roi = uniformFilter.clampedCropRect(input.cols(), input.rows());
                } else if (filterIndex == 1) {
                    roi = exponentialFilter.clampedCropRect(input.cols(), input.rows());
                } else if (filterIndex == 2) {
                    roi = gaussianFilter.clampedCropRect(input.cols(), input.rows());
                } else {
                    roi = kalmanFilter.clampedCropRect(input.cols(), input.rows());
                }
                
                //Record crop efficiency metric
                if (metricsEnabled) {
                    double frameArea = input.cols() * input.rows();
                    double cropArea = roi.width * roi.height;
                    metrics.recordCrop(cropArea / frameArea);
                }
                
                //Apply crop and resize to output
                if (roi.width > 2 && roi.height > 2) {
                    //Extract cropped region
                    Mat patch = warpFull.submat(roi);
                    
                    //Resize to output dimensions
                    Imgproc.resize(patch, output, output.size());
                    patch.release();
                } else {
                    //Crop region too small: just resize full warped frame
                    Imgproc.resize(warpFull, output, output.size());
                }
            } else {
                //Warp unavailable: pass through input
                input.copyTo(output);
            }
        }

        //Free temporary matrices used for this frame
        if (inc23 != null) {
            inc23.release();
        }
        if (inliers != null) {
            inliers.release();
        }
        if (warp23 != null) {
            warp23.release();
        }

        //Update tracking state for next frame
        
        //Save current gray frame as previous
        gray.copyTo(prevGray);
        
        //Update feature points for next iteration
        if (prevPts != null) prevPts.release();
        if (lkNext.size() < 80) {
            //Too few tracked points: re-detect features
            prevPts = PipelineUtils.goodFeaturesToPoint2f(gray);
        } else {
            //Sufficient tracked points: use them for next frame
            prevPts = new MatOfPoint2f(lkNext.toArray(new Point[0]));
        }

        //Free remaining temporary matrices
        gray.release();
        status.release();
        err.release();
        
        //Complete metrics collection and update display
        if (metricsEnabled) {
            metrics.endFrame();
            
            //Log CSV header once
            if (!csvHeaderLogged) {
                Log.d(TAG, "METRICS_CSV_HEADER: frame,filter,tracked,total,trackRate,inliers,totalPts,inlierRate,rawShake,smoothShake,shakeReduct,rawVar,smoothVar,varReduct,cropPct");
                csvHeaderLogged = true;
            }
            
            //Log CSV data row each frame
            Log.d(TAG, "METRICS_CSV_DATA: " + metrics.getCSVRow());
            
            final String summary = metrics.getLiveSummary();
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    metricsTextView.setText(summary);
                }
            });
        }
    }

}
