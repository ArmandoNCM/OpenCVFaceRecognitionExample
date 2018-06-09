package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class ImageConversion {

    /**
     * Converts an Android Bitmap to an OpenCV Matrix (Mat)
     * @param bitmap Android Bitmap to be converted
     * @return OpenCV Matrix (Mat)
     */
    public static Mat convertBitmapToMatrix(Bitmap bitmap){
        Mat matrix = new Mat();
        Utils.bitmapToMat(bitmap, matrix);
        return matrix;
    }

    /**
     * Converts an OpenCV Matrix to an Android Bitmap
     * @param matrix OpenCV Matrix to be converted
     * @return Android Bitmap
     */
    public static Bitmap convertMatrixToBitmap(Mat matrix){
        Bitmap bitmap = Bitmap.createBitmap(matrix.cols(), matrix.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matrix, bitmap);
        return bitmap;
    }
}
