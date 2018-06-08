package com.armandoncm.opencvfacerecognitionexample;

import android.app.Application;
import android.content.Context;

import org.opencv.android.InstallCallbackInterface;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

public class ApplicationCore extends Application {

    private static ApplicationCore instance;

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;

        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, getApplicationContext(), new LoaderCallbackInterface() {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:

                        break;

                     default:


                         break;
                }
            }

            @Override
            public void onPackageInstall(int operation, InstallCallbackInterface callback) {

                switch (operation){
                    case InstallCallbackInterface.NEW_INSTALLATION:
                        callback.install();
                }

            }
        });
    }

    public static Context getContext(){
        return instance.getApplicationContext();
    }
}
