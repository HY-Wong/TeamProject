package com.example.sleepapneadetector

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Callback
import okhttp3.Response
import java.io.File
import java.io.IOException

class EnsembleModelAPI {

    private val client = OkHttpClient()

    /**
     * Sends an inference request to your API with the given audio file.
     *
     * @param audioFile The WAV file to send.
     * @param callback A callback that will be invoked with the API response as a String,
     *                 or null if an error occurred.
     */
    fun runInference(audioFile: File, callback: (result: String?) -> Unit) {
        val mediaType = "audio/wav".toMediaTypeOrNull()
        // Create a RequestBody for the audio file.
        val fileBody = RequestBody.create(mediaType, audioFile)

        // Build a multipart form request.
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("audio_file", audioFile.name, fileBody)
            .build()

        // Replace the URL below with your actual API endpoint.
        val request = Request.Builder()
            .url("https://fastapi-service-801935418303.us-central1.run.app/predict")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: okhttp3.Call, e: IOException) {
                callback(null)
            }

            override fun onResponse(call: okhttp3.Call, response: Response) {
                response.use {
                    if (!response.isSuccessful) {
                        callback(null)
                    } else {
                        callback(response.body?.string())
                    }
                }
            }
        })
    }
}
