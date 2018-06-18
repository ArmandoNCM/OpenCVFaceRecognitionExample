package com.armandoncm.opencvfacerecognitionexample.faceRecognition;

import android.content.Context;

import com.armandoncm.opencvfacerecognitionexample.ApplicationCore;

import org.apache.commons.lang3.SerializationUtils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * This class is responsible for exportation and importation of trained data (Eigen-vectors)
 *
 * @author ArmandoNCM
 */
public class DataManagement {

    /**
     * Serializes the Eigen-vectors data to the file system
     * @param identifier File identifier or name
     * @param matrixData Data to serialize
     * @throws Exception If something goes wrong while serializing data
     */
    public static void exportData(String identifier, Mat matrixData) throws Exception{

        Context context = ApplicationCore.getContext();
        File parentDirectory = context.getExternalFilesDir(null);

        identifier = identifier.concat(".data");
        File dataFile = new File(parentDirectory, identifier);

        int rows = matrixData.rows();
        int cols = matrixData.cols();

        double[][] data = new double[rows][cols];

        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                data[i][j] = matrixData.get(i,j)[0];
            }
        }

        OutputStream outputStream = new FileOutputStream(dataFile);

        SerializationUtils.serialize(data, outputStream);

    }

    /**
     * Deserializes matrix data with the given identifier
     * @param identifier Identifier of the data to deserialize
     * @return OpenCV Matrix (Mat)
     * @throws Exception If something goes wrong while deserializing data
     */
    public static Mat deserializeData(String identifier) throws Exception{

        Context context = ApplicationCore.getContext();
        File parentDirectory = context.getExternalFilesDir(null);

        identifier = identifier.concat(".data");
        File dataFile = new File(parentDirectory, identifier);

        InputStream inputStream = new FileInputStream(dataFile);

        double[][] data = SerializationUtils.deserialize(inputStream);

        int rows = data.length;
        int cols = data[0].length;

        Mat result = new Mat(rows, cols, CvType.CV_32FC1);

        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                result.put(i,j, data[i][j]);
            }
        }

        return result;
    }
}
