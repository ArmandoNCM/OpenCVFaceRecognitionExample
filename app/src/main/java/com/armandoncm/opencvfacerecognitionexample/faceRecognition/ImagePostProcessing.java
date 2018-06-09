package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ImagePostProcessing {

    public static Mat upscaleImage(Mat image){

        Size size = image.size();
        double aspectRatio = size.width / size.height;

        double newWidth = 1000;
        double newHeight = newWidth / aspectRatio;

        Size newSize = new Size(newWidth, newHeight);

        Mat resizedImage = new Mat();

        Imgproc.resize(image, resizedImage, newSize);

        return resizedImage;
    }
}
