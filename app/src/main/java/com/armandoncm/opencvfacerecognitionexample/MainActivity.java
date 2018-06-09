package com.armandoncm.opencvfacerecognitionexample;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.armandoncm.opencvfacerecognitionexample.faceRecognition.FaceDetection;
import com.armandoncm.opencvfacerecognitionexample.faceRecognition.ImageConversion;
import com.armandoncm.opencvfacerecognitionexample.faceRecognition.ImagePostProcessing;
import com.armandoncm.opencvfacerecognitionexample.faceRecognition.ImagePreProcessing;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity {

    private static final int REQUEST_TAKE_PHOTO = 1;
    private static final int REQUEST_SELECT_PHOTO = 2;
    private static final int PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 100;

    private ImageView imageView;

    private Uri photoURI;

    boolean permissionGranted;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonSelectImage = findViewById(R.id.btnSelectImage);
        buttonSelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (permissionGranted) {

                    showSourceSelectionDialog();
                }
            }
        });

        imageView = findViewById(R.id.imageView);

        checkPermissions();

    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);

        return File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                // Error occurred while creating the File
                ex.printStackTrace();
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                photoURI = FileProvider.getUriForFile(this,
                        "com.armandoncm.opencvfacerecognitionexample",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
            }
        }
    }

    private void dispatchSelectPictureIntent() {

        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(intent, REQUEST_SELECT_PHOTO);
    }

    private void showSourceSelectionDialog() {

        String[] options = new String[2];

        final int optionGallery = 0;
        final int optionCamera = 1;

        options[optionGallery] = getResources().getString(R.string.option_select_from_gallery);
        options[optionCamera] = getResources().getString(R.string.option_capture_from_camera);


        DialogInterface.OnClickListener onClickListener = new DialogInterface.OnClickListener() {

            @Override
            public void onClick(DialogInterface dialog, int which) {

                switch (which) {
                    case optionCamera:
                        dispatchTakePictureIntent();
                        break;
                    case optionGallery:
                        dispatchSelectPictureIntent();
                }

            }
        };

        new AlertDialog.Builder(this)
                .setTitle(R.string.title_select_image_source)
                .setItems(options, onClickListener)
                .show();
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Uri uri;
        switch (requestCode) {
            case REQUEST_SELECT_PHOTO:
                switch (resultCode) {
                    case RESULT_OK:

                        uri = data.getData();

                        new ProcessImageTask().execute(uri);
                        break;
                    case RESULT_CANCELED:
                        Toast.makeText(this, R.string.msg_image_selection_cancelled, Toast.LENGTH_LONG).show();
                        break;
                }
                break;
            case REQUEST_TAKE_PHOTO:

                switch (resultCode) {
                    case RESULT_OK:

                        uri = photoURI;

                        new ProcessImageTask().execute(uri);
                        break;
                    case RESULT_CANCELED:
                        Toast.makeText(this, R.string.msg_image_selection_cancelled, Toast.LENGTH_LONG).show();
                        break;
                }
                break;
        }
    }

    @SuppressLint("StaticFieldLeak")
    private class ProcessImageTask extends AsyncTask<Uri, Void, Bitmap> {

        @Override
        protected Bitmap doInBackground(Uri... uris) {

            try {
                Bitmap bitmap = ImagePreProcessing.loadBitmap(uris[0]);
                Mat matrix = ImageConversion.convertBitmapToMatrix(bitmap);
                matrix = ImagePreProcessing.removeColorInformation(matrix);
                FaceDetection faceDetection = FaceDetection.getInstance();
                Rect[] detectedFaceRectangles = faceDetection.detectFaces(matrix);

                final int numberOfDetectedFaces = detectedFaceRectangles.length;
                MainActivity.this.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        String message = MainActivity.this.getResources().getString(R.string.msg_number_of_detected_faces);
                        message = message.concat(": " + numberOfDetectedFaces);
                        Toast.makeText(MainActivity.this, message, Toast.LENGTH_LONG).show();
                    }
                });
                if (numberOfDetectedFaces > 0) {

                    matrix = faceDetection.cropFace(matrix, detectedFaceRectangles[0]);
                    matrix = ImagePostProcessing.upscaleImage(matrix, 1000);
                }
                return ImageConversion.convertMatrixToBitmap(matrix);
            } catch (IOException e) {
                e.printStackTrace();
            }

            return null;
        }

        @Override
        protected void onPostExecute(Bitmap bitmap) {
            super.onPostExecute(bitmap);

            imageView.setImageBitmap(bitmap);

        }
    }


    public void checkPermissions() {
        // Here, thisActivity is the current activity
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {

            // No explanation needed; request the permission
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);

            // PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE is an
            // app-defined int constant. The callback method gets the
            // result of the request.
        } else {
            // Permission has already been granted
            ApplicationCore.loadOpenCV();
            permissionGranted = true;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String permissions[], @NonNull int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                    ApplicationCore.loadOpenCV();
                    permissionGranted = true;
                } else {
                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                    permissionGranted = false;
                }
            }
            // other 'case' lines to check for other
            // permissions this app might request.
        }
    }
}
