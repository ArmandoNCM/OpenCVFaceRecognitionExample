package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import android.annotation.SuppressLint;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Range;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.StatModel;
import org.opencv.ml.TrainData;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * This class is responsible for the PCA Computation from the training faces
 *
 * @author ArmandoNCM
 */
public class ModelTraining {

    /**
     * Array of faces
     */
    private List<Mat> faces;

    /**
     * Mean face obtained from training
     */
    private Mat mean;

    /**
     * Eigen-vectors obtained from training
     */
    private Mat eigenVectors;

    /**
     * Model used in facial recognition
     */
    private StatModel statModel;

    /**
     * Creates an instance of ModelTraining with an empty faces array
     */
    public ModelTraining(){

        faces = new ArrayList<>();
    }

    /**
     * Obtains the number of added faces
     * @return Number of faces that have been added
     */
    public int getNumberOfAddedFaces(){
        return faces.size();
    }

    /**
     * Obtains the mean face
     * @return Mean face
     */
    public Mat getMean(){
        return mean;
    }

    /**
     * Obtains the eigen vectors obtained from the PCA computation
     * @return Eigenvectors
     */
    public Mat getEigenVectors() {
        return eigenVectors;
    }

    /**
     * Reshape images so that each image is just a single row representing
     * a point in an N dimensional space
     *
     * The reshaped images form part of a single matrix as individual rows
     * @param images Array of images to reshape
     * @return Matrix of training set
     */
    private Mat createTrainingMatrix(Mat[] images){

        if (images.length > 0){


            int numberOfColumns = FaceDetection.DESIRED_FACE_HEIGHT * FaceDetection.DESIRED_FACE_WIDTH;

            Mat trainingSet = new Mat(images.length, numberOfColumns, CvType.CV_8U);

            for (int i = 0; i < images.length; i++){

                Mat image = images[i];

                // Reshape image to represent a single data point in an N dimensional space
                // where N equals to the number of pixel values (Width * Height) of the images
                // All images should be the same size, this is guaranteed by the preprocessing
                image = image.reshape(0, 1);

                // Add the image to the i'th row of the training set matrix
                image.copyTo(trainingSet.row(i));
            }

            return trainingSet;
        } else {
            // There were no images in the array
            return null;
        }
    }

    /**
     * This method uses the array of cropped faces in the form of OpenCV Matrix (Mat)
     * to train the face recognition model of a given person
     * @param trainingSet Training set
     */
    private void trainModel(Mat trainingSet){

        // Initializing matrices
        mean = new Mat();
        eigenVectors = new Mat();
        // Computing the mean, the eigenVectors and the eigenValues
        Core.PCACompute(trainingSet, mean, eigenVectors);

        // Projection of the
        Mat projection = new Mat();
        Core.PCAProject(trainingSet, mean, eigenVectors, projection);
        inspectMat(projection);

        // Creation of the Support Vector Machine
        SVM svm = SVM.create();
        svm.setType(SVM.ONE_CLASS);
        svm.setKernel(SVM.RBF);
        svm.setGamma(3);
        svm.setC(5);
        svm.setNu(0.5);

        Mat responses = new Mat(0,0,CvType.CV_32S);
        svm.trainAuto(trainingSet, Ml.ROW_SAMPLE, responses);

        statModel = svm;

    }

    public void trainModel() throws Exception{

        if (faces.size() == 0) {
            throw new Exception("No faces have been added");
        }

        Mat trainingSet = createTrainingMatrix(faces.toArray(new Mat[0]));

        trainModel(trainingSet);
    }

    /**
     * Adds the given face to the face list
     * @param image Face to add
     */
    public void addFace(Mat image) {
        faces.add(image);
    }

    /**
     * This method is used for debugging purposes
     * It inspects an OpenCV Matrix (Mat) object
     * If you put a breakpoint on it, you can see the
     * contents of a given Matrix
     * @param mat Matrix to inspect
     */
    private static void inspectMat(Mat mat){

        int rows = mat.rows();
        int columns = mat.cols();
        Log.d("INSPCETION", "Dimensions: " + rows + ", " + columns);

        double value;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                value = mat.get(i,j)[0];
                //noinspection UnusedAssignment
                value = value * 1;
            }
        }
    }
}
