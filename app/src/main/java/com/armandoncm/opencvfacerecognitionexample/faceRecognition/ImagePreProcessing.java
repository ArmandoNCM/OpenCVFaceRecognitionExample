package com.armandoncm.opencvfacerecognitionexample.faceRecognition;


import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.media.ExifInterface;

import com.armandoncm.opencvfacerecognitionexample.ApplicationCore;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;

/**
 * This class is responsible for pre-processing images in preparation for face recognition with OpenCV
 */
public class ImagePreProcessing {

    /**
     * Removes the Color information from an OpenCV Matrix
     * @param matrix Original OpenCV Matrix to filter
     * @return Gray Scale OpenCV Matrix
     */
    public static Mat removeColorInformation(Mat matrix){
        Mat grayScaleMatrix = new Mat();
        Imgproc.cvtColor(matrix, grayScaleMatrix, Imgproc.COLOR_RGB2GRAY);
        return grayScaleMatrix;
    }

    /**
     * Rotates the given Android Bitmap to the correct orientation to display on an ImageView
     * @param bitmap Bitmap whose orientation is to be corrected
     * @param orientation Orientation attribute obtained by the ExifInterface EXIF tag reading capabilities
     * @return Correctly oriented Android Bitmap
     */
    private static Bitmap rotateBitmap(Bitmap bitmap, int orientation) {

        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_NORMAL:
                return bitmap;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setRotate(180);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90);
                break;
            default:
                return bitmap;
        }
        try {
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }
        catch (OutOfMemoryError e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Retrieves an Android Bitmap given its Uri
     * @param uri Uri of the content to be fetched
     * @return Android Bitmap
     * @throws IOException If there was a problem retrieving the content
     */
    public static Bitmap loadBitmap(Uri uri) throws IOException {

        Bitmap bitmap = MediaStore.Images.Media.getBitmap(ApplicationCore.getContext().getContentResolver(), uri);

        try {
            InputStream inputStream = ApplicationCore.getContext().getContentResolver().openInputStream(uri);
            assert inputStream != null;
            ExifInterface exifInterface = new ExifInterface(inputStream);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);
            return rotateBitmap(bitmap, orientation);
        } catch (IOException e){
            e.printStackTrace();
            // If the orientation correction attempt failed, return the bitmap as is
            return bitmap;
        }
    }


}
