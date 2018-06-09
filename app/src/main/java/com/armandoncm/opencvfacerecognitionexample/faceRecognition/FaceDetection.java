package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import android.content.Context;
import android.util.Log;

import com.armandoncm.opencvfacerecognitionexample.ApplicationCore;

import org.apache.commons.io.IOUtils;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * This class is responsible solely for the face detection in images with OpenCV
 * Keep in mind the difference between detection and recognition formal definitions when
 * using this class
 *
 * @author ArmandoNCM
 */
public class FaceDetection {

    private static final String PRE_TRAINED_DATA_FILENAME = "opencv_data.xml";

    private static FaceDetection instance;

    private CascadeClassifier cascadeClassifier;

    private FaceDetection(){

        Context context = ApplicationCore.getContext();

        try {
            // The pre-trained data for face detection will be copied into a caches directory from the RAW resources
            File parentDirectory = context.getCacheDir();
            File trainedDataFile = new File(parentDirectory, PRE_TRAINED_DATA_FILENAME);
            // Apache IOUtils is used to copy from an InputStream to an OutputStream
            FileOutputStream fileOutputStream = new FileOutputStream(trainedDataFile);
            InputStream inputStream = context.getResources().openRawResource(org.opencv.R.raw.haarcascade_frontalface_default);
            IOUtils.copy(inputStream, fileOutputStream);
            // Close all streams
            inputStream.close();
            fileOutputStream.flush();
            fileOutputStream.close();
            // Get the canonical path to pass it on to the initialization of the Cascade Classifier
            String canonicalPath = trainedDataFile.getCanonicalPath();
            cascadeClassifier = new CascadeClassifier(canonicalPath);
            cascadeClassifier.load(canonicalPath); // This code line is required in addition to passing the data in the constructor for the correct initialization of the Classifier

            Log.d("CLASSIFIER", "Classifier Correctly Initialized: " + !cascadeClassifier.empty());

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    /**
     * Detect faces present in an OpenCV Matrix (Mat) representation of an image
     * @param image Image in the form of an OpenCV Matrix (Mat)
     * @return Array of OpenCV Rectangle objects representing ROI's (Regions of Interest)
     */
    public Rect[] detectFaces(Mat image){

        // The detection of faces writes the ROI's to a Matrix of Rectangles
        MatOfRect matOfRect = new MatOfRect();
        // Minimum and Maximum size of detected objects (faces) in pixels
        Size minSize = new Size(50,50);
        Size maxSize = new Size(2000, 2000);
        /*
         * scaleFactor: image is scaled down by 30%
         * minNeighbors: The higher the lower chance of detection but higher quality of the
         *      detection themselves, recommended values range from 3 to 6
         * flags: unused by the new implementation of cascade classifier
         */
        cascadeClassifier.detectMultiScale(image, matOfRect, 1.3, 4, 0, minSize, maxSize);

        // Array of ROI's
        Rect[] rectangles = matOfRect.toArray();

        Log.d("CLASSIFIER", "Number of Faces Detected: " + rectangles.length);

        return rectangles;
    }

    /**
     * Crops a subsection of the image determined by the given rectangle
     * @param image Image to be cropped
     * @param rectangle ROI in which the Face was detected
     * @return Subsection of the image containing solely the face delimited by the given rectangle
     */
    public Mat cropFace(Mat image, Rect rectangle){

        return image.submat(rectangle);
    }

    /**
     * Singleton pattern instantiation
     * @return Instance of FaceDetection
     */
    public static FaceDetection getInstance(){

        if (instance == null){
            instance = new FaceDetection();
        }

        return instance;
    }


}
