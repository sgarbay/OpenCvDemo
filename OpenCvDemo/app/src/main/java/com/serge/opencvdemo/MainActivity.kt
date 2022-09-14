package com.serge.opencvdemo

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCamera2View
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Moments

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private val TAG = "OpenCvDemo"
    private val RECORD_REQUEST_CODE = 101

    // view
    private val viewFinder by lazy { findViewById<JavaCamera2View>(R.id.cameraView) }
    // image storage
    lateinit var imageMat: Mat

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        setupPermissions()
        checkOpenCV(this)

        viewFinder.setCameraPermissionGranted()
        viewFinder.visibility = SurfaceView.VISIBLE
        viewFinder.setCameraIndex(CameraCharacteristics.LENS_FACING_FRONT)
        viewFinder.setCvCameraViewListener(this)
    }

    override fun onResume() {
        super.onResume()
        supportActionBar?.hide()
        viewFinder?.let { viewFinder.enableView() }
    }

    override fun onPause() {
        super.onPause()
        viewFinder?.let { viewFinder.disableView() }
    }

    override fun onDestroy() {
        super.onDestroy()
        viewFinder?.let { viewFinder.disableView() }
    }

    companion object {
        fun shortMsg(context: Context, s: String) =
            Toast.makeText(context, s, Toast.LENGTH_SHORT).show()

        // messages:
        private const val OPENCV_SUCCESSFUL = "OpenCV Loaded Successfully!"
        private const val OPENCV_FAIL = "Could not load OpenCV!!!"
    }

    private fun checkOpenCV(context: Context) =
        if (OpenCVLoader.initDebug()){
            shortMsg(context, OPENCV_SUCCESSFUL)
        } else {
            shortMsg(context, OPENCV_FAIL)
        }

    private fun setupPermissions() {
        val permission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)

        if (permission != PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission to record denied")
            makeRequest()
        }
    }

    private fun makeRequest() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), RECORD_REQUEST_CODE)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        when (requestCode) {
            RECORD_REQUEST_CODE -> {
                if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    Log.i(TAG, "Permission has been denied by user")
                } else {
                    Log.i(TAG, "Permission has been granted by user")
                }
            }
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        imageMat = Mat(width, height, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        imageMat.release()
    }

    // Acutal logic of the hand tracking is inside onCameraFrame
    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        imageMat = inputFrame!!.rgba()

        val ratio = 1.0 // Shrink the image for performance boost

        val resizedMat = Mat()
        Imgproc.resize(imageMat, resizedMat, Size(ratio*imageMat.width(), ratio*imageMat.height()))

        // Removes or fuses small objects
        Imgproc.blur(resizedMat, resizedMat, Size(5.0, 5.0))

        // Mask color of skin TODO Calibrate at start
        val mask = Mat(resizedMat.height(), resizedMat.width(), CvType.CV_8UC4)
        Core.inRange(resizedMat, Scalar(160.0, 100.0, 80.0, 0.0), Scalar(180.0, 120.0, 100.0, 255.0), mask)

        // Assume hand is in view. Then the area of the hand in the mask will grow.
        val kernelEllipse = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(11.0, 11.0))
        val kernelSquare = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.dilate(mask, mask, kernelEllipse, Point(-1.0,-1.0), 1)
        Imgproc.erode(mask, mask, kernelSquare, Point(-1.0,-1.0), 3)
        Imgproc.dilate(mask, mask, kernelEllipse, Point(-1.0,-1.0), 1)
        Imgproc.medianBlur(mask, mask, 5)
        Imgproc.dilate(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(8.0, 8.0)))
        Imgproc.medianBlur(mask, mask, 5)
        Imgproc.dilate(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
        Imgproc.threshold(mask, mask, 127.0,255.0,0)

        // Get contours of all blobs
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

        //val dstMat = Mat(imageMat.height(), imageMat.width(), CvType.CV_8UC4)
        //Imgproc.drawContours(dstMat, contours, -1, Scalar(255.0, 0.0, 0.0, 255.0))

        // Get largest contour and use it as the hand
        var maxI = 0
        var maxArea = 0.0
        for(i in contours.indices){
            val area = Imgproc.contourArea(contours[i])
            if(area > maxArea){
                maxArea = area
                maxI = i
            }
        }

        val dstMat = Mat(imageMat.height(), imageMat.width(), CvType.CV_8UC4)
        dstMat.setTo(Scalar(0.0, 0.0, 0.0, 0.0))

        if(contours.size > 0){
            val moments: Moments = Imgproc.moments(contours[maxI])
            val cx = moments._m10/(ratio*moments._m00)
            val cy = moments._m01/(ratio*moments._m00)

            Imgproc.circle(dstMat, Point(cx, cy), 7, Scalar(0.0, 255.0, 0.0, 255.0), Imgproc.FILLED)
        }

//        val dstMat = Mat()
//        Imgproc.resize(mask, dstMat, Size(imageMat.width().toDouble(), imageMat.height().toDouble()))

        return dstMat
    }
}