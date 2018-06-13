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
    private static final String PRE_TRAINED_LEFT_EYE_DATA_FILENAME = "haarcascade_eye.xml";

    // Pre-trained data resource ID's
    private static final int PRE_TRAINED_FACE_DATA_RESOURCE_ID = org.opencv.R.raw.haarcascade_frontalface_default;
    private static final int PRE_TRAINED_LEFT_EYE_DATA_RESOURCE_ID = org.opencv.R.raw.haarcascade_eye;

    // Minimum and Maximum size of detected objects (faces) in pixels
    private static final Size MINIMUM_OBJECT_DETECTION_SIZE = new Size(30,30);
    private static final Size MAXIMUM_OBJECT_DETECTION_SIZE = new Size(ImagePreProcessing.DOWNSCALED_IMAGE_WIDTH, ImagePreProcessing.DOWNSCALED_IMAGE_WIDTH * 2);

    // The values are recommended values from the book referenced in the README
    // Eye Area Width and Height
    private static final double EYE_AREA_WIDTH = 0.3;
    private static final double EYE_AREA_HEIGHT = 0.3;
    // Left Eye Area Position
    private static final double LEFT_EYE_AREA_X = 0.16;
    private static final double LEFT_EYE_AREA_Y = 0.22;
    // Right Eye Area Position
    private static final double RIGHT_EYE_AREA_X = 1.0 - LEFT_EYE_AREA_X - EYE_AREA_WIDTH + 0.03;
    private static final double RIGHT_EYE_AREA_Y = LEFT_EYE_AREA_Y;
    // Desired left eye position
    private static final double DESIRED_LEFT_EYE_X = 0.16;
    private static final double DESIRED_LEFT_EYE_Y = 0.14;
    // Desired right eye position
    private static final double DESIRED_RIGHT_EYE_X = 1.0 - DESIRED_LEFT_EYE_X;

    // Desired face width and height
    private static final int DESIRED_FACE_WIDTH = 320;
    private static final int DESIRED_FACE_HEIGHT = 320;

    // Singleton pattern instance holder
    private static FaceDetection instance;

    // Object Detection Classifiers
    private CascadeClassifier faceClassifier;
    private CascadeClassifier eyeClassifier;

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

        initializeEyeClassifier();
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
    private void initializeEyeClassifier(){

        String filePath = copyResourceToCache(PRE_TRAINED_LEFT_EYE_DATA_FILENAME, PRE_TRAINED_LEFT_EYE_DATA_RESOURCE_ID);
        eyeClassifier = new CascadeClassifier(filePath);
        eyeClassifier.load(filePath); // This code line is required in addition to passing the data in the constructor for the correct initialization of the Classifier
        Log.d("CLASSIFIER", "Left Eye Classifier Correctly Initialized: " + !eyeClassifier.empty());
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
         * scaleFactor: image is scaled down by 5%
         * minNeighbors: The higher the lower chance of detection but higher quality of the
         *      detection themselves, recommended values range from 3 to 6
         * flags: unused by the new implementation of cascade classifier
         */
        faceClassifier.detectMultiScale(image, matOfRect, 1.05, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT, MINIMUM_OBJECT_DETECTION_SIZE, MAXIMUM_OBJECT_DETECTION_SIZE);

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
     * Calculates the left eye region of the face
     * @param faceImage Image of the whole face
     * @return Region of the image containing th left eye
     */
    private Rect getLeftEyeRegion(Mat faceImage){

        Size imageSize = faceImage.size();
        Point eyeAreaTopLeftPoint = new Point(LEFT_EYE_AREA_X * imageSize.width, LEFT_EYE_AREA_Y * imageSize.height);
        Size eyeAreaSize = new Size(EYE_AREA_WIDTH * imageSize.width, EYE_AREA_HEIGHT * imageSize.height);

        return new Rect(eyeAreaTopLeftPoint, eyeAreaSize);
    }

    /**
     * Calculates the right eye region of the face
     * @param faceImage Image of the whole face
     * @return Region of the image containing the right eye
     */
    private Rect getRightEyeRegion(Mat faceImage){

        Size imageSize = faceImage.size();
        Point eyeAreaTopLeftPoint = new Point(RIGHT_EYE_AREA_X * imageSize.width, RIGHT_EYE_AREA_Y * imageSize.height);
        Size eyeAreaSize = new Size(EYE_AREA_WIDTH * imageSize.width, EYE_AREA_HEIGHT * imageSize.height);

        return new Rect(eyeAreaTopLeftPoint, eyeAreaSize);
    }

    /**
     * Picks the largest rectangle out of an array
     * @param rectangles List of rectangles
     * @return Largest rectangle
     */
    public Rect pickTheLargestArea(Rect[] rectangles){
        if (rectangles.length == 0){
            return null;
        } else if (rectangles.length == 1){
            return rectangles[0];
        } else {
            Rect largestRect = rectangles[0];
            for (int i = 1; i < rectangles.length; i++){
                if (rectangles[i].area() > largestRect.area()){
                    largestRect = rectangles[i];
                }
            }
            return largestRect;
        }
    }

    /**
     * Detects an eye in the given region
     * @param eyeRegion Eye region
     * @return Center of the detected eye
     */
    private Point detectEye(Mat eyeRegion){

        MatOfRect detectedEyes = new MatOfRect();
        Size regionSize = eyeRegion.size();
        Size minSize = new Size(20, 20);

        eyeClassifier.detectMultiScale(eyeRegion, detectedEyes, 1.05, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT, minSize, regionSize);

        Rect[] rectangles = detectedEyes.toArray();
        if (rectangles.length > 0){
            Rect eyeROI = pickTheLargestArea(rectangles);
            Log.d("EYE-DETECTION", "Eye Detected");
            return new Point(eyeROI.x + (eyeROI.width / 2), eyeROI.y + (eyeROI.height / 2));
        } else {
            Log.w("EYE-DETECTION", "NO Eye Detected");
            return null;
        }
    }

    /**
     * Rotates and warps the face to improve face recognition
     * @param face Face image
     * @return Rotated and warped image
     */
    public Mat alignEyes(Mat face){

        // Calculating eye regions
        Rect leftEyeRegionRectangle = getLeftEyeRegion(face);
        Rect rightEyeRegionRectangle = getRightEyeRegion(face);
        // Cropping of eye regions
        Mat leftEyeRegion = cropImage(face, leftEyeRegionRectangle);
        Mat rightEyeRegion = cropImage(face, rightEyeRegionRectangle);
        // Detecting position of eyes
        Point leftEyePosition = detectEye(leftEyeRegion);
        Point rightEyePosition = detectEye(rightEyeRegion);

        if (leftEyePosition == null || rightEyePosition == null){
            Log.w("EYE-DETECTION", "One or both eyes were not detected");
            return face;
        }

        leftEyePosition = new Point(leftEyePosition.x + leftEyeRegionRectangle.x, leftEyePosition.y + leftEyeRegionRectangle.y);
        rightEyePosition = new Point(rightEyePosition.x + rightEyeRegionRectangle.x, rightEyePosition.y + rightEyeRegionRectangle.y);

        // Calculating center of eyes
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

    /**
     * Uses separate histogram equalization to balance lighting conditions on both sides
     * of the face
     * @param image Image to be equalized
     * @return Equalized Image
     */
    public Mat equalizeLight(Mat image){

        double width = image.cols();
        double height = image.rows();

        double midX = width / 2;

        Mat wholeFace = new Mat();
        Imgproc.equalizeHist(image, wholeFace);

        Mat leftSide = wholeFace.submat(new Rect(0, 0, (int) midX, (int) height));
        Mat rightSide = wholeFace.submat(new Rect((int) midX, 0, (int) (width-midX), (int) height));

        Imgproc.equalizeHist(leftSide, leftSide);
        Imgproc.equalizeHist(rightSide, rightSide);

        // Code exctracted and translated from C++ example in the book
        // "Mastering OpenCV with Practical Computer Vision Projects" by Daniel Lelis Baggio

        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                double v;
                if (x < width/4) {
                    // Left 25%: just use the left face.
                    v = leftSide.get(y, x)[0]; //leftSide.at<uchar>(y,x);
                }
                else if (x < width * 2/4) {
                    // Mid-left 25%: blend the left face & whole face.
                    double lv = leftSide.get(y,x)[0]; //leftSide.at<uchar>(y,x);
                    double wv = wholeFace.get(y,x)[0]; //wholeFace.at<uchar>(y,x);
                    // Blend more of the whole face as it moves
                    // further right along the face.
                    double f = (x -  width * 1/4) / (width / 4);
                    v =  Math.round((1.0f - f) * wv + (f) * lv);
                }
                else if (x < width * 3/4) {
                    // Mid-right 25%: blend right face & whole face.
                    double rv = rightSide.get(y, (int) (x-midX))[0]; //rightSide.at<uchar>(y,x-midX);
                    double wv = wholeFace.get(y,x)[0]; //wholeFace.at<uchar>(y,x);
                    // Blend more of the right-side face as it moves
                    // further right along the face.
                    double f = (x -  width *2/4) / (width / 4);
                    v = Math.round((1.0f - f) * wv + (f) * rv);
                }
                else {
                    // Right 25%: just use the right face.
                    v = rightSide.get(y, (int) (x-midX))[0]; //rightSide.at<uchar>(y,x-midX);
                }

                image.put(y, x, v); //faceImg.at<uchar>(y,x) = v;
            }// end x loop
        }//end y loop


        // Bilateral filtering to reduce noise
        Mat filtered = new Mat(image.size(), CvType.CV_8U);
        Imgproc.bilateralFilter(image, filtered, 0, 20.0, 2.0);

        return filtered;
    }

}
