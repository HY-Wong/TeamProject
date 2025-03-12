package com.example.sleepapneadetector

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity

class SplashActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Use the splash layout
        setContentView(R.layout.activity_splash)

        // Delay for 2 seconds, then navigate to HomeActivity
        Handler(Looper.getMainLooper()).postDelayed({
            startActivity(Intent(this, HomeActivity::class.java))
            finish() // Prevent the user from returning to the splash screen
        }, 2000)
    }
}
