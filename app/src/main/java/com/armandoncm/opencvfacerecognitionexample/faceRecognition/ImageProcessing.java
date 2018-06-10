package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * This class is responsible for converting images from Android Bitmap objects to
 *  OpenCV Matrix (Mat) objects and vice versa
 *
 * @author ArmandoNCM
 */
public class ImageProcessing {

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

    /**
     * This function scales an image to the desired width while maintaining
     * the aspect ratio
     * @param image Image to be scaled
     * @param desiredWidth Desired width of the scaled image
     * @return Scaled image with the same aspect ratio
     */
    public static Mat scaleImage(Mat image, double desiredWidth){

        Size size = image.size();
        double aspectRatio = size.width / size.height;

        double desiredHeight = desiredWidth / aspectRatio;

        Size newSize = new Size(desiredWidth, desiredHeight);
        Mat resizedImage = new Mat();
        Imgproc.resize(image, resizedImage, newSize);

        return resizedImage;
    }
}
