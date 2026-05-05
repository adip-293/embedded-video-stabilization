package com.ece420.lab7;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

//Math and drawing utilities for video stabilization pipeline
public final class PipelineUtils {

    private PipelineUtils() {
    }

    //Feature detection parameters
    public static final int MAX_CORNERS = 200;
    public static final double CORNER_QUALITY = 0.01;
    public static final int CORNER_MIN_DISTANCE = 10;
    public static final double CORNER_ROI_MARGIN_RATIO = 0.10; //Exclude outer 10% of frame

    //Path smoothing parameters
    public static final double PATH_GAUSSIAN_SIGMA = 12.0;
    public static final int PATH_HISTORY_CAP = 400;

    //Motion grid visualization
    public static final int GRID_STEP_PX = 200;
    public static final double GRID_DISP_SCALE = 2.0;
    public static final int GRID_ARROW_THICKNESS = 6;
    public static final double GRID_ARROW_TIP = 0.42;

    //Optical flow parameters
    public static final int LK_MIN_GOOD_FEATURES = 80;
    public static final double RANSAC_REPROJ_THRESH_PX = 3.0;

    //Gyro sensor parameters
    public static final float GYRO_GATE_NORM_RAD_S = 0.5f;
    public static final float GYRO_PREWARP_GAIN = 0.9f;
    public static final int GYRO_PREWARP_AXIS = 2;
    public static final float GYRO_MAX_DT_SEC = 0.18f;

    //Arrow visualization parameters
    public static final double LK_ARROW_DISP_SCALE = 1.0;
    public static final double LK_ARROW_MIN_LEN_PX = 6.0;
    public static final double LK_ARROW_MAX_LEN_PX = 100.0;
    public static final int LK_ARROW_THICKNESS = 6;

    //Convert 2x3 affine to 3x3 homogeneous matrix (identity if null)
    public static Mat partialAffineTo3x3(Mat aff23) {
        Mat m = Mat.eye(3, 3, CvType.CV_64F);
        if (aff23 != null && !aff23.empty()) {
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 3; c++) {
                    m.put(r, c, aff23.get(r, c)[0]);
                }
            }
        }
        return m;
    }

    //Unwrap single angle to be near reference value (avoid 2π jumps)
    public static double unwrapNear(double th, double ref) {
        double t = th;
        while (t - ref > Math.PI) {
            t -= 2 * Math.PI;
        }
        while (t - ref < -Math.PI) {
            t += 2 * Math.PI;
        }
        return t;
    }

    //Unwrap angle series in-place to remove discontinuities
    public static void unwrapSeriesInPlace(double[] th, int n) {
        for (int i = 1; i < n; i++) {
            double d = th[i] - th[i - 1];
            while (d > Math.PI) {
                th[i] -= 2 * Math.PI;
                d = th[i] - th[i - 1];
            }
            while (d < -Math.PI) {
                th[i] += 2 * Math.PI;
                d = th[i] - th[i - 1];
            }
        }
    }

    //1D Gaussian smoothing with edge padding
    public static double[] gaussianSmooth1d(double[] x, int n, double sigma) {
        if (n <= 0) {
            return new double[0];
        }
        if (n == 1) {
            return new double[]{x[0]};
        }
        
        //Build Gaussian kernel
        int r = Math.max(1, (int) (3 * sigma));
        int klen = 2 * r + 1;
        double[] k = new double[klen];
        double sumk = 0;
        for (int i = 0; i < klen; i++) {
            int t = i - r;
            k[i] = Math.exp(-(t * t) / (2 * sigma * sigma));
            sumk += k[i];
        }
        for (int i = 0; i < klen; i++) {
            k[i] /= sumk;
        }

        //Pad edges with first and last values
        int nPad = n + 2 * r;
        double[] padded = new double[nPad];
        for (int i = 0; i < r; i++) {
            padded[i] = x[0];
        }
        System.arraycopy(x, 0, padded, r, n);
        for (int i = 0; i < r; i++) {
            padded[r + n + i] = x[n - 1];
        }

        //Apply convolution
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double acc = 0;
            for (int j = 0; j < klen; j++) {
                acc += padded[i + j] * k[j];
            }
            out[i] = acc;
        }
        return out;
    }

    //Extract tx, ty, theta, log(scale) from 3x3 cumulative transform
    public static void similarityParamsFromR(Mat pathR, double[] out4) {
        double r00 = pathR.get(0, 0)[0];
        double r10 = pathR.get(1, 0)[0];
        out4[0] = pathR.get(0, 2)[0];
        out4[1] = pathR.get(1, 2)[0];
        out4[2] = Math.atan2(r10, r00);
        double scale = Math.hypot(r00, r10);
        if (scale < 1e-9) {
            scale = 1e-9;
        }
        out4[3] = Math.log(scale);
    }

    //Build 3x3 similarity transform from smoothed parameters (tx, ty, theta, log scale)
    public static Mat buildQ3(double smTx, double smTy, double smTheta, double smLogScale) {
        double sc = Math.exp(smLogScale);
        double c = Math.cos(smTheta);
        double s = Math.sin(smTheta);
        Mat q = Mat.eye(3, 3, CvType.CV_64F);
        q.put(0, 0, sc * c);
        q.put(0, 1, -sc * s);
        q.put(0, 2, smTx);
        q.put(1, 0, sc * s);
        q.put(1, 1, sc * c);
        q.put(1, 2, smTy);
        return q;
    }

    //Compute warp M = Q * R^-1 as 2x3 float matrix
    public static Mat computeWarpM23FromRAndQ(Mat pathR, Mat q3) {
        if (pathR == null || pathR.empty() || q3 == null || q3.empty()) {
            return null;
        }
        Mat rInv = new Mat(3, 3, CvType.CV_64F);
        if (Math.abs(Core.determinant(pathR)) < 1e-12) {
            rInv.release();
            return null;
        }
        Core.invert(pathR, rInv, Core.DECOMP_LU);
        Mat m3 = new Mat(3, 3, CvType.CV_64F);
        Core.gemm(q3, rInv, 1.0, new Mat(), 0, m3, 0);
        rInv.release();

        Mat out = new Mat(2, 3, CvType.CV_32F);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                out.put(r, c, (float) m3.get(r, c)[0]);
            }
        }
        m3.release();
        return out;
    }

    //Create 2x3 identity affine transform
    public static Mat identityAffine23() {
        Mat id = new Mat(2, 3, CvType.CV_32F);
        id.put(0, 0, 1f);
        id.put(0, 1, 0f);
        id.put(0, 2, 0f);
        id.put(1, 0, 0f);
        id.put(1, 1, 1f);
        id.put(1, 2, 0f);
        return id;
    }

    //Create 2x3 rotation affine about image center
    public static Mat rotationAffine23AboutCenter(double thetaRad, double cx, double cy) {
        float c = (float) Math.cos(thetaRad);
        float s = (float) Math.sin(thetaRad);
        float fx = (float) cx;
        float fy = (float) cy;
        Mat m = new Mat(2, 3, CvType.CV_32F);
        m.put(0, 0, c);
        m.put(0, 1, -s);
        m.put(0, 2, fx - c * fx + s * fy);
        m.put(1, 0, s);
        m.put(1, 1, c);
        m.put(1, 2, fy - s * fx - c * fy);
        return m;
    }

    //Compute L2 norm of gyro sample (rad/s)
    public static float gyroNormRadS(float gx, float gy, float gz) {
        return (float) Math.sqrt(gx * gx + gy * gy + gz * gz);
    }

    //Compute stable crop box from history of warp matrices
    public static Rect cropBoxFromMs(List<Mat> ms23, int w, int h) {
        if (ms23 == null || ms23.isEmpty()) {
            return new Rect(0, 0, w, h);
        }
        double xmin = Double.NEGATIVE_INFINITY;
        double xmax = Double.POSITIVE_INFINITY;
        double ymin = Double.NEGATIVE_INFINITY;
        double ymax = Double.POSITIVE_INFINITY;

        Mat m33 = Mat.eye(3, 3, CvType.CV_64F);
        Mat inv = new Mat(3, 3, CvType.CV_64F);
        Mat corners = new Mat(3, 4, CvType.CV_64F);
        double[] cornerX = {0, w, w, 0};
        double[] cornerY = {0, 0, h, h};
        for (int col = 0; col < 4; col++) {
            corners.put(0, col, cornerX[col]);
            corners.put(1, col, cornerY[col]);
            corners.put(2, col, 1.0);
        }

        Mat prod = new Mat(3, 4, CvType.CV_64F);

        for (Mat ms : ms23) {
            if (ms == null || ms.empty()) {
                continue;
            }
            //Defensive check: verify Mat dimensions and type before accessing
            if (ms.rows() < 2 || ms.cols() < 3) {
                continue;
            }
            boolean validMat = true;
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 3; c++) {
                    double[] val = ms.get(r, c);
                    if (val == null || val.length == 0) {
                        validMat = false;
                        break;
                    }
                    m33.put(r, c, val[0]);
                }
                if (!validMat) break;
            }
            if (!validMat) {
                continue;
            }
            m33.put(2, 0, 0.0);
            m33.put(2, 1, 0.0);
            m33.put(2, 2, 1.0);
            if (Math.abs(Core.determinant(m33)) < 1e-12) {
                continue;
            }
            Core.invert(m33, inv, Core.DECOMP_LU);
            Core.gemm(inv, corners, 1.0, new Mat(), 0, prod, 0);

            for (int col = 0; col < 4; col++) {
                double xh = prod.get(0, col)[0];
                double yh = prod.get(1, col)[0];
                double zh = prod.get(2, col)[0];
                if (Math.abs(zh) < 1e-12) {
                    continue;
                }
                double xs = xh / zh;
                double ys = yh / zh;
                xmin = Math.max(xmin, xs);
                xmax = Math.min(xmax, xs);
                ymin = Math.max(ymin, ys);
                ymax = Math.min(ymax, ys);
            }
        }

        prod.release();
        corners.release();
        inv.release();
        m33.release();

        if (!Double.isFinite(xmin) || !Double.isFinite(xmax)
                || !Double.isFinite(ymin) || !Double.isFinite(ymax)) {
            return new Rect(0, 0, w, h);
        }

        int inset = Math.max(1, (int) (0.005 * Math.min(w, h)));
        int x1 = (int) Math.ceil(xmin) + inset;
        int y1 = (int) Math.ceil(ymin) + inset;
        int x2 = (int) Math.floor(xmax) - inset;
        int y2 = (int) Math.floor(ymax) - inset;

        double m = 0.05;
        int fx1 = (int) (w * m);
        int fy1 = (int) (h * m);
        int fx2 = (int) (w * (1 - m));
        int fy2 = (int) (h * (1 - m));

        if (x2 <= x1 || y2 <= y1 || (long) (x2 - x1) * (y2 - y1) < (long) (0.72 * w * h)) {
            return new Rect(fx1, fy1, Math.max(1, fx2 - fx1), Math.max(1, fy2 - fy1));
        }
        int cx1 = Math.max(x1, fx1);
        int cy1 = Math.max(y1, fy1);
        int cx2 = Math.min(x2, fx2);
        int cy2 = Math.min(y2, fy2);
        int rw = Math.max(1, cx2 - cx1);
        int rh = Math.max(1, cy2 - cy1);
        return new Rect(cx1, cy1, rw, rh);
    }

    //Clamp rectangle to image bounds
    public static Rect clampRectToImage(Rect r, int cols, int rows) {
        int x1 = Math.max(0, Math.min(r.x, cols - 1));
        int y1 = Math.max(0, Math.min(r.y, rows - 1));
        int x2 = Math.max(x1 + 1, Math.min(r.x + r.width, cols));
        int y2 = Math.max(y1 + 1, Math.min(r.y + r.height, rows));
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    private static final double MOTION_ARROW_SKIP_BELOW_PX = 0.5;

    //Compute arrow tip for motion visualization with length scaling and clamping
    public static Point motionArrowTip(double ox, double oy, double vx, double vy) {
        double len = Math.hypot(vx, vy);
        if (len < MOTION_ARROW_SKIP_BELOW_PX) {
            return null;
        }
        double visLen = len * LK_ARROW_DISP_SCALE;
        visLen = Math.max(visLen, LK_ARROW_MIN_LEN_PX);
        visLen = Math.min(visLen, LK_ARROW_MAX_LEN_PX);
        double ux = vx / len;
        double uy = vy / len;
        return new Point(ox + ux * visLen, oy + uy * visLen);
    }

    //Compute shortened arrow for LK optical flow display
    public static Point lkArrowTipForDisplay(Point prev, Point next) {
        return motionArrowTip(prev.x, prev.y, next.x - prev.x, next.y - prev.y);
    }

    //Draw uniform grid of motion arrows showing warp field
    public static void drawEvenGridMotionArrows(Mat rgbaOut, Mat warp23) {
        int h = rgbaOut.rows();
        int w = rgbaOut.cols();
        Scalar yellow = new Scalar(255, 255, 0, 255);
        int countX = Math.max(2, 1 + (w - 1) / GRID_STEP_PX);
        int countY = Math.max(2, 1 + (h - 1) / GRID_STEP_PX);

        double m00 = warp23.get(0, 0)[0];
        double m01 = warp23.get(0, 1)[0];
        double m02 = warp23.get(0, 2)[0];
        double m10 = warp23.get(1, 0)[0];
        double m11 = warp23.get(1, 1)[0];
        double m12 = warp23.get(1, 2)[0];

        for (int iy = 0; iy < countY; iy++) {
            double gy = (countY <= 1) ? 0 : iy * (h - 1.0) / (countY - 1);
            for (int ix = 0; ix < countX; ix++) {
                double gx = (countX <= 1) ? 0 : ix * (w - 1.0) / (countX - 1);
                double qx = m00 * gx + m01 * gy + m02;
                double qy = m10 * gx + m11 * gy + m12;
                double vx = (qx - gx) * GRID_DISP_SCALE;
                double vy = (qy - gy) * GRID_DISP_SCALE;
                Point tip = motionArrowTip(gx, gy, vx, vy);
                if (tip == null) {
                    continue;
                }
                Point tail = new Point(Math.round(gx), Math.round(gy));
                Imgproc.arrowedLine(rgbaOut, tail, tip, yellow,
                        GRID_ARROW_THICKNESS, Imgproc.LINE_AA, 0, GRID_ARROW_TIP);
            }
        }
    }

    //Detect Shi-Tomasi corners with reduced ROI to avoid edge features
    public static MatOfPoint2f goodFeaturesToPoint2f(Mat gray) {
        //Calculate margin to exclude edge regions where features may leave frame
        int marginX = (int)(gray.cols() * CORNER_ROI_MARGIN_RATIO);
        int marginY = (int)(gray.rows() * CORNER_ROI_MARGIN_RATIO);
        
        //Define inner ROI (excluding margins)
        int roiWidth = gray.cols() - 2 * marginX;
        int roiHeight = gray.rows() - 2 * marginY;
        
        //Ensure ROI is valid
        if (roiWidth <= 0 || roiHeight <= 0) {
            //Fallback to full frame if margins are too large
            MatOfPoint corners = new MatOfPoint();
            Imgproc.goodFeaturesToTrack(gray, corners, MAX_CORNERS, CORNER_QUALITY, CORNER_MIN_DISTANCE);
            MatOfPoint2f pts = new MatOfPoint2f(corners.toArray());
            corners.release();
            return pts;
        }
        
        Rect roi = new Rect(marginX, marginY, roiWidth, roiHeight);
        Mat roiGray = gray.submat(roi);
        
        //Detect corners in ROI
        MatOfPoint corners = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(roiGray, corners, MAX_CORNERS, CORNER_QUALITY, CORNER_MIN_DISTANCE);
        
        //Offset corner coordinates back to full frame coordinates
        Point[] pts = corners.toArray();
        for (int i = 0; i < pts.length; i++) {
            pts[i].x += marginX;
            pts[i].y += marginY;
        }
        
        roiGray.release();
        corners.release();
        return new MatOfPoint2f(pts);
    }
}
