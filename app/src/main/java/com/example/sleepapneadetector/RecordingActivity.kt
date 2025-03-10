package com.example.sleepapneadetector

import android.content.Context
import android.content.Intent
import android.database.Cursor
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.TimeUnit
import android.animation.ObjectAnimator
import android.animation.ValueAnimator

class RecordingActivity : AppCompatActivity() {

    private var isRecording = false
    private var mediaRecorder: MediaRecorder? = null
    private lateinit var audioFile: File
    private lateinit var timerText: TextView
    private lateinit var pulseAnimator: ObjectAnimator
    private var recordingStartTime: Long = 0L
    private val timerHandler = Handler(Looper.getMainLooper())

    // ActivityResultLauncher for file upload
    private lateinit var uploadLauncher: ActivityResultLauncher<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recording)

        val btnStop: Button = findViewById(R.id.btnStopRecording)
        val btnSubmit: Button = findViewById(R.id.btnSubmit)
        val btnUpload: Button = findViewById(R.id.btn_upload_recording)
        val circleView: ImageView = findViewById(R.id.circle_animation)
        timerText = findViewById(R.id.recordingTimer)

        btnSubmit.visibility = Button.GONE // Hide submit initially

        // Set up the pulse animation
        pulseAnimator = ObjectAnimator.ofFloat(circleView, "scaleX", 1f, 1.5f).apply {
            duration = 800
            repeatMode = ValueAnimator.REVERSE
            repeatCount = ValueAnimator.INFINITE
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener {
                circleView.scaleY = it.animatedValue as Float
            }
        }

        // Register the file picker launcher
        uploadLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) {
                // Copy the selected file to the cache directory and get its absolute path.
                val filePath = getRealPathFromURI(this, uri)
                if (!filePath.isNullOrEmpty()) {
                    audioFile = File(filePath)
                    Toast.makeText(this, "File selected: ${audioFile.name}", Toast.LENGTH_SHORT).show()
                    btnSubmit.visibility = Button.VISIBLE
                } else {
                    Toast.makeText(this, "Failed to process selected file", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this, "No file selected", Toast.LENGTH_SHORT).show()
            }
        }

        // Start recording automatically and begin animation.
        startRecording()
        pulseAnimator.start()

        btnStop.setOnClickListener {
            if (isRecording) {
                stopRecording()
                pulseAnimator.cancel()
                circleView.scaleX = 1f
                circleView.scaleY = 1f
                btnSubmit.visibility = Button.VISIBLE
                btnStop.text = "Recording Stopped"
            } else {
                Toast.makeText(this, "No active recording", Toast.LENGTH_SHORT).show()
            }
        }

        // Upload button click launches the file picker.
        btnUpload.setOnClickListener {
            // Launch file picker for audio files.
            uploadLauncher.launch("audio/*")
        }

        // Submit button click – send the audio for further processing.
        btnSubmit.setOnClickListener {
            if (::audioFile.isInitialized) {
                val intent = Intent(this, ResultsActivity::class.java).apply {
                    putExtra("AUDIO_FILE_PATH", audioFile.absolutePath)
                }
                startActivity(intent)
                finish()
            } else {
                Toast.makeText(this, "No recording to submit", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun startRecording() {
        try {
            val outputDir = File(filesDir, "recordings")
            if (!outputDir.exists()) outputDir.mkdir()
            audioFile = File(outputDir, "recording_${System.currentTimeMillis()}.mp3")

            mediaRecorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setOutputFile(audioFile.absolutePath)
                prepare()
                start()
            }

            recordingStartTime = System.currentTimeMillis()
            timerHandler.post(timerRunnable)

            isRecording = true
            Toast.makeText(this, "Recording started", Toast.LENGTH_SHORT).show()
        } catch (e: IOException) {
            Toast.makeText(this, "Failed to start recording: ${e.message}", Toast.LENGTH_LONG).show()
        } catch (e: IllegalStateException) {
            Toast.makeText(this, "Recording error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null
            isRecording = false
            timerHandler.removeCallbacks(timerRunnable)
            Toast.makeText(this, "Recording saved to: ${audioFile.absolutePath}", Toast.LENGTH_SHORT).show()
        } catch (e: IllegalStateException) {
            Toast.makeText(this, "Failed to stop recording: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private val timerRunnable = object : Runnable {
        override fun run() {
            val elapsedMillis = System.currentTimeMillis() - recordingStartTime
            val minutes = TimeUnit.MILLISECONDS.toMinutes(elapsedMillis)
            val seconds = TimeUnit.MILLISECONDS.toSeconds(elapsedMillis) % 60
            timerText.text = String.format(Locale.getDefault(), "%02d:%02d", minutes, seconds)
            timerHandler.postDelayed(this, 1000)
        }
    }

    /**
     * Copies the content from the provided URI into a temporary file in the cache directory.
     * Returns the absolute path of the temporary file.
     */
    private fun getRealPathFromURI(context: Context, uri: Uri): String? {
        var filePath: String? = null
        val projection = arrayOf(MediaStore.Audio.Media.DATA)
        var cursor: Cursor? = null
        try {
            cursor = context.contentResolver.query(uri, projection, null, null, null)
            if (cursor != null && cursor.moveToFirst()) {
                val columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Audio.Media.DATA)
                filePath = cursor.getString(columnIndex)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            cursor?.close()
        }
        if (filePath == null || filePath.isEmpty()) {
            try {
                val fileName = uri.lastPathSegment ?: "temp_audio.wav"
                val tempFile = File(context.cacheDir, fileName)
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    tempFile.outputStream().use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
                filePath = tempFile.absolutePath
            } catch (e: Exception) {
                e.printStackTrace()
                return null
            }
        }
        return filePath
    }
}
