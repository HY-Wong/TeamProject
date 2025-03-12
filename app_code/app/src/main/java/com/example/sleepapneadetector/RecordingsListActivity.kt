package com.example.sleepapneadetector

import android.media.MediaPlayer
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class RecordingsListActivity : AppCompatActivity() {

    private lateinit var recordingsList: ListView
    private lateinit var noRecordingsText: TextView
    private lateinit var recordings: List<File>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recordings_list)

        recordingsList = findViewById(R.id.lv_recordings)
        noRecordingsText = findViewById(R.id.tv_no_recordings)

        // Load recordings from the directory
        val recordingsDir = File(filesDir, "recordings")
        recordings = recordingsDir.listFiles()?.toList() ?: emptyList()

        if (recordings.isEmpty()) {
            noRecordingsText.visibility = View.VISIBLE
            recordingsList.visibility = View.GONE
        } else {
            noRecordingsText.visibility = View.GONE
            recordingsList.visibility = View.VISIBLE

            // Populate ListView
            val adapter = ArrayAdapter(
                this,
                android.R.layout.simple_list_item_1,
                recordings.map { it.name }
            )
            recordingsList.adapter = adapter

            // Handle item clicks to play recordings
            recordingsList.onItemClickListener =
                AdapterView.OnItemClickListener { _, _, position, _ ->
                    playRecording(recordings[position])
                }

            // Handle long clicks to delete recordings
            recordingsList.onItemLongClickListener =
                AdapterView.OnItemLongClickListener { _, _, position, _ ->
                    showDeleteDialog(recordings[position], adapter)
                    true
                }
        }
    }

    private fun playRecording(file: File) {
        val mediaPlayer = MediaPlayer()
        try {
            mediaPlayer.setDataSource(file.absolutePath)
            mediaPlayer.prepare()
            mediaPlayer.start()
            Toast.makeText(this, "Playing: ${file.name}", Toast.LENGTH_SHORT).show()

            mediaPlayer.setOnCompletionListener {
                Toast.makeText(this, "Playback completed", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error playing file: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun showDeleteDialog(file: File, adapter: ArrayAdapter<String>) {
        AlertDialog.Builder(this)
            .setTitle("Delete Recording")
            .setMessage("Are you sure you want to delete ${file.name}?")
            .setPositiveButton("Yes") { _, _ ->
                if (file.delete()) {
                    recordings = recordings.filter { it != file }
                    adapter.clear()
                    adapter.addAll(recordings.map { it.name })
                    adapter.notifyDataSetChanged()

                    if (recordings.isEmpty()) {
                        noRecordingsText.visibility = View.VISIBLE
                        recordingsList.visibility = View.GONE
                    }

                    Toast.makeText(this, "${file.name} deleted", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Failed to delete ${file.name}", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("No", null)
            .show()
    }
}
