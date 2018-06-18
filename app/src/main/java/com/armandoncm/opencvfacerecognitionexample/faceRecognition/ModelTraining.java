package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

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
    private Mat createTrainingSet(Mat[] images){

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
        Core.PCACompute(trainingSet, mean, eigenVectors, 10);
    }

    public void trainModel() throws Exception{

        if (faces.size() < 2) {
            throw new Exception("No faces have been added");
        }

        Mat trainingSet = createTrainingSet(faces.toArray(new Mat[0]));

        trainModel(trainingSet);
    }

    /**
     * Adds the given face to the face list
     * @param image Face to add
     */
    public void addFace(Mat image) {
        faces.add(image);
    }

}
