package com.example.sleepapneadetector

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class ReportsActivity : AppCompatActivity() {

    private lateinit var adapter: ArrayAdapter<String>
    private lateinit var reportFiles: Array<File>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_reports)

        val listView: ListView = findViewById(R.id.reports_list)
        val noReportsText: TextView = findViewById(R.id.no_reports_text)

        val reportsDir = File(filesDir, "reports")
        if (!reportsDir.exists()) reportsDir.mkdir()

        // Fetch report files from the directory
        reportFiles = reportsDir.listFiles() ?: emptyArray()
        val reportNames = reportFiles.map { it.name }.toMutableList()

        if (reportNames.isEmpty()) {
            noReportsText.visibility = View.VISIBLE
            listView.visibility = View.GONE
        } else {
            noReportsText.visibility = View.GONE
            listView.visibility = View.VISIBLE

            // Populate the ListView with report file names
            adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, reportNames)
            listView.adapter = adapter

            // Open the report when an item is clicked
            listView.onItemClickListener = AdapterView.OnItemClickListener { _, _, position, _ ->
                val selectedFile = reportFiles[position]
                val reportContent = selectedFile.readText()

                // Navigate to ResultsActivity and display the selected report
                val intent = Intent(this, ResultsActivity::class.java).apply {
                    putExtra("REPORT_CONTENT", reportContent)
                }
                startActivity(intent)
            }

            // Handle long-click for deleting a report
            listView.onItemLongClickListener =
                AdapterView.OnItemLongClickListener { _, _, position, _ ->
                    val selectedFile = reportFiles[position]

                    // Show a confirmation dialog
                    AlertDialog.Builder(this)
                        .setTitle("Delete Report")
                        .setMessage("Are you sure you want to delete this report?")
                        .setPositiveButton("Yes") { _, _ ->
                            if (selectedFile.delete()) {
                                Toast.makeText(this, "Report deleted", Toast.LENGTH_SHORT).show()
                                refreshReportList(reportsDir)
                            } else {
                                Toast.makeText(this, "Failed to delete report", Toast.LENGTH_SHORT)
                                    .show()
                            }
                        }
                        .setNegativeButton("No", null)
                        .show()

                    true
                }
        }
    }

    private fun refreshReportList(reportsDir: File) {
        // Refresh the list of reports after deleting one
        reportFiles = reportsDir.listFiles() ?: emptyArray()
        val reportNames = reportFiles.map { it.name }
        adapter.clear()
        adapter.addAll(reportNames)
        adapter.notifyDataSetChanged()

        // Handle case when all reports are deleted
        val noReportsText: TextView = findViewById(R.id.no_reports_text)
        val listView: ListView = findViewById(R.id.reports_list)

        if (reportNames.isEmpty()) {
            noReportsText.visibility = View.VISIBLE
            listView.visibility = View.GONE
        } else {
            noReportsText.visibility = View.GONE
            listView.visibility = View.VISIBLE
        }
    }
}
