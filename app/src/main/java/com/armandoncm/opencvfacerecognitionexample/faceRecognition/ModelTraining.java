package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

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

import java.util.ArrayList;
import java.util.List;

/**
 * This class is responsible for the PCA Computation from the training faces
 *
 * @author ArmandoNCM
 */
public class ModelTraining {

    /**
     * Last face added
     */
    private Mat lastFace;

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
        Core.PCACompute(trainingSet, mean, eigenVectors, 300);

        inspectMat(eigenVectors);

//        Mat result = new Mat();
//        Core.PCAProject(trainingSet, mean, eigenVectors, result);
//        inspectMat(result);

        Mat responses = new Mat(eigenVectors.rows(),1,CvType.CV_32S);

        TrainData trainData = TrainData.create(eigenVectors, Ml.ROW_SAMPLE, responses);

        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);

//        svm.trainAuto(eigenVectors, Ml.ROW_SAMPLE, responses);

        statModel = svm;
        statModel.train(trainData);

    }

    public void trainModel() throws Exception{

        if (faces.size() < 2) {
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
        lastFace = image;
    }

    private static void inspectMat(Mat mat){

        int rows = mat.rows();
        int columns = mat.cols();
        Log.d("INSPCETION", "Dimensions: " + rows + ", " + columns);

        double value;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                value = mat.get(i,j)[0];
                value = value * 1;
            }
        }
    }

    public void test(){

        Mat testData = createTrainingMatrix(new Mat[]{lastFace});

        Mat eigenVectors = new Mat();
        Core.PCACompute(testData, mean, eigenVectors, 10);

        inspectMat(eigenVectors);

//        Mat projected = new Mat();
//        Core.PCAProject(testData, mean, eigenVectors, projected);
//        inspectMat(projected);

        Mat results = new Mat();
        float prediction = statModel.predict(eigenVectors, results, StatModel.RAW_OUTPUT);
//        float prediction = statModel.predict(eigenVectors);
        inspectMat(results);

        Log.d("PREDICTION", "Prediction: " + prediction);

    }


    public Mat reconstruct(){

        Mat image = lastFace;
        image = image.reshape(1, 1);
        inspectMat(image);

        int numberOfComponents = 50;
        Mat truncatedEigenVectors = new Mat(eigenVectors, Range.all(), new Range(0, numberOfComponents));
        inspectMat(truncatedEigenVectors);

        Mat projection = new Mat();
        Core.PCAProject(image, mean, truncatedEigenVectors, projection);

        inspectMat(projection);

        Mat reconstruction = new Mat();
        Core.PCABackProject(projection, mean, truncatedEigenVectors, reconstruction);

        reconstruction = reconstruction.reshape(1, FaceDetection.DESIRED_FACE_HEIGHT);

        Core.normalize(reconstruction, reconstruction, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);

        return reconstruction;

    }

}
