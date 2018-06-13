package com.armandoncm.opencvfacerecognitionexample;

import android.app.Application;
import android.content.Context;

import com.armandoncm.opencvfacerecognitionexample.faceRecognition.FaceDetection;

import org.opencv.android.InstallCallbackInterface;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.contrib.FaceRecognizer;
import org.opencv.core.Algorithm;

public class ApplicationCore extends Application {

    private static ApplicationCore instance;

    private static boolean openCVLoaded = false;

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
        loadOpenCV();
    }

    /**
     * Loads the OpenCV Library
     * If the OpenCV Manager is not present in the phone, an install request is shown to the user
     */
    public static void loadOpenCV(){

        // If the OpenCV library has already been loaded, do nothing and exit
        if (openCVLoaded){
            return;
        }

        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, instance.getApplicationContext(), new LoaderCallbackInterface() {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                        // The Library was successfully loaded
                        openCVLoaded = true;
                        FaceDetection.getInstance();

                        break;

                    default:


                        break;
                }
            }

            @Override
            public void onPackageInstall(int operation, InstallCallbackInterface callback) {

                switch (operation){
                    case InstallCallbackInterface.NEW_INSTALLATION:
                        // The user is required to install OpenCV Manager
                        callback.install();
                        break;
                }

            }
        });
    }

    public static Context getContext(){
        return instance.getApplicationContext();
    }
}
