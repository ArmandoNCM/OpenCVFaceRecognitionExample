package com.armandoncm.opencvfacerecognitionexample;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.armandoncm.opencvfacerecognitionexample.faceRecognition.FaceDetection;
import com.armandoncm.opencvfacerecognitionexample.faceRecognition.ImageProcessing;
import com.armandoncm.opencvfacerecognitionexample.faceRecognition.ImagePreProcessing;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity {

    private static final int REQUEST_TAKE_PHOTO = 1;
    private static final int REQUEST_SELECT_PHOTO = 2;

    private ImageView imageView;

    private Uri photoURI;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonSelectImage = findViewById(R.id.btnSelectImage);
        buttonSelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showSourceSelectionDialog();
            }
        });

        imageView = findViewById(R.id.imageView);

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
                // Image loading
                Bitmap bitmap = ImagePreProcessing.loadBitmap(uris[0]);

                // Image conversion to OpenCV Matrix (Mat)
                Mat image = ImageProcessing.convertBitmapToMatrix(bitmap);

                // Image Pre-Processing
                image = ImagePreProcessing.removeColorInformation(image);
                image = ImagePreProcessing.equalizeHistogram(image);
                Mat downscaledImage = ImagePreProcessing.downscaleImage(image);

                double scale = image.size().width / downscaledImage.size().width;

                // Face Detection
                FaceDetection faceDetection = FaceDetection.getInstance();
                Rect[] detectedFaceRectangles = faceDetection.detectFaces(downscaledImage);

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
                    // Scale ROI to compensate for downscaling at the time of face detection
                    Rect faceROI = detectedFaceRectangles[0];
                    Size currentROISize = faceROI.size();
                    faceROI = new Rect(new Point(faceROI.x * scale, faceROI.y * scale), new Size(currentROISize.width * scale, currentROISize.height * scale));
                    // Cropping of the original image
                    image = faceDetection.cropImage(image, faceROI);

                    image = faceDetection.alignEyes(image);

                    image = ImageProcessing.scaleImage(image, 1000);

                }
                return ImageProcessing.convertMatrixToBitmap(image);
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
}
