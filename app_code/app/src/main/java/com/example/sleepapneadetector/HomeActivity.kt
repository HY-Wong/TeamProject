package com.example.sleepapneadetector

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class HomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        val btnStartRecording: Button = findViewById(R.id.btn_start_recording)
        val btnViewRecordings: Button = findViewById(R.id.btn_view_recordings)
        val btnViewReports: Button = findViewById(R.id.btn_view_reports)

        // Navigate to RecordingActivity
        btnStartRecording.setOnClickListener {
            val intent = Intent(this, RecordingActivity::class.java)
            startActivity(intent)
        }

        // Navigate directly to RecordingsActivity
        btnViewRecordings.setOnClickListener {
            val intent = Intent(this, RecordingsListActivity::class.java)
            startActivity(intent)
        }


        // Navigate to ReportsActivity
        btnViewReports.setOnClickListener {
            val intent = Intent(this, ReportsActivity::class.java)
            startActivity(intent)
        }
    }
}
