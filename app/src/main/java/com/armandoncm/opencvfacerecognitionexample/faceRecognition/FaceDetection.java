package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import android.content.Context;
import android.util.Log;

import com.armandoncm.opencvfacerecognitionexample.ApplicationCore;

import org.apache.commons.io.IOUtils;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
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

    // Cache file names
    private static final String PRE_TRAINED_FACE_DATA_FILENAME = "opencv_face_data.xml";
    private static final String PRE_TRAINED_LEFT_EYE_DATA_FILENAME = "opencv_left_eye_data.xml";
    private static final String PRE_TRAINED_RIGHT_EYE_DATA_FILENAME = "opencv_right_eye_data.xml";

    // Pre-trained data resource ID's
    private static final int PRE_TRAINED_FACE_DATA_RESOURCE_ID = org.opencv.R.raw.haarcascade_frontalface_default;
    private static final int PRE_TRAINED_LEFT_EYE_DATA_RESOURCE_ID = org.opencv.R.raw.haarcascade_lefteye_2splits;
    private static final int PRE_TRAINED_RIGHT_EYE_DATA_RESOURCE_ID = org.opencv.R.raw.haarcascade_righteye_2splits;

    // Minimum and Maximum size of detected objects (faces) in pixels
    private static final Size MINIMUM_OBJECT_DETECTION_SIZE = new Size(30,30);
    private static final Size MAXIMUM_OBJECT_DETECTION_SIZE = new Size(ImagePreProcessing.DOWNSCALED_IMAGE_WIDTH, ImagePreProcessing.DOWNSCALED_IMAGE_WIDTH * 2);

    // The values are recommended values from the book referenced in the README
    // Eye Area Width and Height
    private static final double EYE_AREA_WIDTH = 0.37;
    private static final double EYE_AREA_HEIGHT = 0.36;
    // Left Eye Area Position
    private static final double LEFT_EYE_AREA_X = 0.12;
    private static final double LEFT_EYE_AREA_Y = 0.17;
    // Right Eye Area Position
    private static final double RIGHT_EYE_AREA_X = 1.0 - LEFT_EYE_AREA_X - EYE_AREA_WIDTH;
    private static final double RIGHT_EYE_AREA_Y = LEFT_EYE_AREA_Y;

    // Singleton pattern instance holder
    private static FaceDetection instance;

    // Object Detection Classifiers
    private CascadeClassifier faceClassifier;
    private CascadeClassifier leftEyeClassifier;
    private CascadeClassifier rightEyeClassifier;

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

    /**
     * Creates a new instance of the FaceDetection class
     */
    private FaceDetection(){

        initializeFaceClassifier();

        initializeLeftEyeClassifier();

        initializeRightEyeClassifier();
    }

    /**
     * Copies the pre-trained data from the application resources to the cache directory
     * @param filename Filename of the cache file to be created
     * @param resourceId ID of the resource to be cached
     * @return Canonical path to the created cache file
     */
    private String copyResourceToCache(String filename, int resourceId){

        Context context = ApplicationCore.getContext();

        try {
            // The pre-trained data for face detection will be copied into a caches directory from the RAW resources
            File parentDirectory = context.getCacheDir();
            File trainedDataFile = new File(parentDirectory, filename);
            // Apache IOUtils is used to copy from an InputStream to an OutputStream
            FileOutputStream fileOutputStream = new FileOutputStream(trainedDataFile);
            InputStream inputStream = context.getResources().openRawResource(resourceId);
            IOUtils.copy(inputStream, fileOutputStream);
            // Close all streams
            inputStream.close();
            fileOutputStream.flush();
            fileOutputStream.close();
            // Get the canonical path to pass it on to the initialization of the Cascade Classifier
            return trainedDataFile.getCanonicalPath();

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Initializes the Face Classifier
     */
    private void initializeFaceClassifier(){

        String filePath = copyResourceToCache(PRE_TRAINED_FACE_DATA_FILENAME, PRE_TRAINED_FACE_DATA_RESOURCE_ID);
        faceClassifier = new CascadeClassifier(filePath);
        faceClassifier.load(filePath); // This code line is required in addition to passing the data in the constructor for the correct initialization of the Classifier
        Log.d("CLASSIFIER", "Face Classifier Correctly Initialized: " + !faceClassifier.empty());
    }

    /**
     * Initializes the Left Eye Classifier
     */
    private void initializeLeftEyeClassifier(){

        String filePath = copyResourceToCache(PRE_TRAINED_LEFT_EYE_DATA_FILENAME, PRE_TRAINED_LEFT_EYE_DATA_RESOURCE_ID);
        leftEyeClassifier = new CascadeClassifier(filePath);
        leftEyeClassifier.load(filePath); // This code line is required in addition to passing the data in the constructor for the correct initialization of the Classifier
        Log.d("CLASSIFIER", "Left Eye Classifier Correctly Initialized: " + !leftEyeClassifier.empty());
    }

    /**
     * Initializes the Right Eye Classifier
     */
    private void initializeRightEyeClassifier(){

        String filePath = copyResourceToCache(PRE_TRAINED_RIGHT_EYE_DATA_FILENAME, PRE_TRAINED_RIGHT_EYE_DATA_RESOURCE_ID);
        rightEyeClassifier = new CascadeClassifier(filePath);
        rightEyeClassifier.load(filePath); // This code line is required in addition to passing the data in the constructor for the correct initialization of the Classifier
        Log.d("CLASSIFIER", "Right Eye Classifier Correctly Initialized: " + !rightEyeClassifier.empty());
    }

    /**
     * Detect faces present in an OpenCV Matrix (Mat) representation of an image
     * @param image Image in the form of an OpenCV Matrix (Mat)
     * @return Array of OpenCV Rectangle objects representing ROI's (Regions of Interest)
     */
    public Rect[] detectFaces(Mat image){

        // The detection of faces writes the ROI's to a Matrix of Rectangles
        MatOfRect matOfRect = new MatOfRect();

        /*
         * scaleFactor: image is scaled down by 30%
         * minNeighbors: The higher the lower chance of detection but higher quality of the
         *      detection themselves, recommended values range from 3 to 6
         * flags: unused by the new implementation of cascade classifier
         */
        faceClassifier.detectMultiScale(image, matOfRect, 1.3, 4, 0, MINIMUM_OBJECT_DETECTION_SIZE, MAXIMUM_OBJECT_DETECTION_SIZE);

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
    public Mat cropImage(Mat image, Rect rectangle){

        return image.submat(rectangle);
    }

    /**
     * Crops the left eye region of the face
     * @param faceImage Image of the whole face
     * @return Region of the image containing th left eye
     */
    public Mat cropLeftEye(Mat faceImage){

        Size imageSize = faceImage.size();
        Point eyeAreaCenterPoint = new Point(LEFT_EYE_AREA_X * imageSize.width, LEFT_EYE_AREA_Y * imageSize.height);
        Size eyeAreaSize = new Size(EYE_AREA_WIDTH * imageSize.width, EYE_AREA_HEIGHT * imageSize.height);

        Rect leftEyeArea = new Rect(eyeAreaCenterPoint, eyeAreaSize);
        return cropImage(faceImage, leftEyeArea);
    }

    /**
     * Crops the right eye region of the face
     * @param faceImage Image of the whole face
     * @return Region of the image containing the right eye
     */
    public Mat cropRightEye(Mat faceImage){

        Size imageSize = faceImage.size();
        Point eyeAreaCenterPoint = new Point(RIGHT_EYE_AREA_X * imageSize.width, RIGHT_EYE_AREA_Y * imageSize.height);
        Size eyeAreaSize = new Size(EYE_AREA_WIDTH * imageSize.width, EYE_AREA_HEIGHT * imageSize.height);

        Rect rightEyeArea = new Rect(eyeAreaCenterPoint, eyeAreaSize);
        return cropImage(faceImage, rightEyeArea);
    }

    /**
     * Picks the largest ROI out of a Matrix
     * @param matrix Matrix where the largest ROI is going to be obtained from
     * @param rectangles List of ROI's where the largest one is going to be picked from
     * @return Largest ROI obtained from the image or the image unchanged if no ROI's were found
     */
    private Mat pickTheLargestArea(Mat matrix, Rect[] rectangles){
        if (rectangles.length == 0){
            return matrix;
        } else if (rectangles.length == 1){
            return matrix.submat(rectangles[0]);
        } else {
            Rect largestRect = rectangles[0];
            for (int i = 1; i < rectangles.length; i++){
                if (rectangles[i].area() > largestRect.area()){
                    largestRect = rectangles[i];
                }
            }
            return matrix.submat(largestRect);
        }
    }


}
