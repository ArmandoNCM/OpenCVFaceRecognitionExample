package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import android.content.Context;
import android.util.Log;

import com.armandoncm.opencvfacerecognitionexample.ApplicationCore;

import org.apache.commons.io.IOUtils;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

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
    // Eye side indicator constants
    private static final int LEFT_EYE = 1;
    private static final int RIGHT_EYE = 2;
    // Desired left eye position
    private static final double DESIRED_LEFT_EYE_X = 0.16;
    private static final double DESIRED_LEFT_EYE_Y = 0.14;
    // Desired right eye position
    private static final double DESIRED_RIGHT_EYE_X = 1.0 - DESIRED_LEFT_EYE_X;
    private static final double DESIRED_RIGHT_EYE_Y = DESIRED_LEFT_EYE_Y;

    // Desired face widht and height
    private static final int DESIRED_FACE_WIDTH = 70;
    private static final int DESIRED_FACE_HEIGHT = 70;

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
        faceClassifier.detectMultiScale(image, matOfRect, 1.3, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT, MINIMUM_OBJECT_DETECTION_SIZE, MAXIMUM_OBJECT_DETECTION_SIZE);

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
    private Mat cropLeftEye(Mat faceImage){

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
    private Mat cropRightEye(Mat faceImage){

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

    /**
     * Detects an eye in the given region
     * @param eyeRegion Eye region
     * @param whichEye Indicator of whether it's the left or right eye
     * @return Center of the detected eye
     */
    private Point detectEye(Mat eyeRegion, int whichEye){

        MatOfRect detectedEyes = new MatOfRect();
        Size regionSize = eyeRegion.size();
        Size minSize = new Size(50, 50);



        switch (whichEye) {
            case LEFT_EYE:

                Log.d("EYE-DETECTION", "Detecting Left Eye");
                leftEyeClassifier.detectMultiScale(eyeRegion, detectedEyes, 1.05, 3, Objdetect.CASCADE_FIND_BIGGEST_OBJECT, minSize, regionSize);
                break;

            case RIGHT_EYE:

                Log.d("EYE-DETECTION", "Detecting Right Eye");
                rightEyeClassifier.detectMultiScale(eyeRegion, detectedEyes, 1.05, 3, Objdetect.CASCADE_FIND_BIGGEST_OBJECT, minSize, regionSize);
                break;
        }

        Rect[] rectangles = detectedEyes.toArray();

        if (rectangles.length > 0){
            Rect eyeROI = rectangles[0];
            Log.d("EYE-DETECTION", "Eye Detected");
            return new Point(eyeROI.x + (eyeROI.width / 2), eyeROI.y + (eyeROI.height / 2));
        }
        Log.d("EYE-DETECTION", "Eye NOT Detected");
        return null;
    }

    public Mat alignEyes(Mat face){

        // Cropping of eye regions
        Mat leftEyeRegion = cropLeftEye(face);
        Mat rightEyeRegion = cropRightEye(face);
        // Detecting position of eyes
        Point leftEyePosition = detectEye(leftEyeRegion, LEFT_EYE);
        Point rightEyePosition = detectEye(rightEyeRegion, RIGHT_EYE);
        // Calculating center of eyes

        if (leftEyePosition == null || rightEyePosition == null){
            Log.w("EYE-DETECTION", "One or both eyes were not detected");
            return face;
        }
        Point eyesCenter = new Point((leftEyePosition.x + rightEyePosition.x) * 0.5, (leftEyePosition.y + rightEyePosition.y) * 0.5);

        double dx = rightEyePosition.x - leftEyePosition.x;
        double dy = rightEyePosition.y - leftEyePosition.y;

        double distance = Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));

        double angle = Math.toDegrees(Math.atan2(dy, dx));

        double desiredDistance = DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X;

        double scale = desiredDistance * DESIRED_FACE_WIDTH / distance;

        Mat rotationMatrix = Imgproc.getRotationMatrix2D(eyesCenter, angle, scale);

        double ex = DESIRED_FACE_WIDTH * 0.5 - eyesCenter.x;
        double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;

        ex += rotationMatrix.get(0, 2)[0];
        ey += rotationMatrix.get(1, 2)[0];

        rotationMatrix.put(0,2, ex);
        rotationMatrix.put(1,2, ey);

        Mat warped = new Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CvType.CV_8U, new Scalar(128));

        Imgproc.warpAffine(face, warped, rotationMatrix, warped.size());

        return warped;
    }


}
