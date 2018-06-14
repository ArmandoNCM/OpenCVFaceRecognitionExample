package com.armandoncm.opencvfacerecognitionexample;

import android.app.Application;
import android.content.Context;
import android.util.Log;

import org.opencv.android.OpenCVLoader;

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

        new Thread(new Runnable() {
            @Override
            public void run() {

                openCVLoaded = OpenCVLoader.initDebug();
                Log.d("NATIVE-LIBRARY", "Loaded Debug: " + openCVLoaded);
            }
        }).start();

    }

    public static Context getContext(){
        return instance.getApplicationContext();
    }
}
