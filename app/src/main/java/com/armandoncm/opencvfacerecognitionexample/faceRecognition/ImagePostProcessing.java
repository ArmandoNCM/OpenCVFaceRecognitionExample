package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * This class is responsible for the post-processing of the images
 * such post-processing is most commonly comprehended as upscaling of images
 * that have been downscaled during the face recognition process
 *
 * @author ArmandoNCM
 */
public class ImagePostProcessing {

    /**
     * This function upscales an image to the desired width while maintaining
     * the aspect ratio
     * @param image Image to be upscaled
     * @param desiredWidth Desired width of the upscaled image
     * @return Upscaled image with the same aspect ratio
     */
    public static Mat upscaleImage(Mat image, double desiredWidth){

        Size size = image.size();
        double aspectRatio = size.width / size.height;

        double newWidth = desiredWidth;
        double newHeight = newWidth / aspectRatio;

        Size newSize = new Size(newWidth, newHeight);
        Mat resizedImage = new Mat();
        Imgproc.resize(image, resizedImage, newSize);

        return resizedImage;
    }
}
