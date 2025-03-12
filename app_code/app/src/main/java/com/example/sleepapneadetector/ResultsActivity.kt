package com.example.sleepapneadetector

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class ResultsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_results)

        val resultTextView: TextView = findViewById(R.id.report_text_view)

        // Retrieve the audio file path passed from the previous activity.
        val audioFilePath = intent.getStringExtra("AUDIO_FILE_PATH")
        if (audioFilePath.isNullOrEmpty()) {
            resultTextView.text = "No audio file provided."
            return
        }

        // Create an instance of your API client.
        val apiClient = EnsembleModelAPI()

        // Run inference by sending the audio file to the API.
        // This is asynchronous—when the API call completes, the callback updates the UI.
        apiClient.runInference(File(audioFilePath)) { result ->
            runOnUiThread {
                if (result != null) {
                    // Display the result returned by your API.
                    resultTextView.text = result
                } else {
                    resultTextView.text = "Error: Failed to get prediction from API."
                }
            }
        }
    }
}
