const express = require('express');
const admin = require('firebase-admin');
const fetch = require('node-fetch');
const path = require('path');
const cors = require('cors'); // WAJIB: Untuk mengizinkan koneksi dari frontend

const app = express();
const port = process.env.PORT || 8080;

// --- KONFIGURASI ---
const serviceAccount = require('./rf-bioflok-firebase-adminsdk-fbsvc-617560ca39.json');
const PREDICTION_API_URL = 'http://127.0.0.1:8081'; 

// WAJIB: Izinkan koneksi dari domain frontend Anda (Firebase Hosting)
// Ganti dengan URL .web.app Anda jika berbeda
app.use(cors({
    origin: 'https://rf-bioflok.web.app' 
}));

// ==========================================================
// ===         KONFIGURASI NOTIFIKASI TELEGRAM            ===
// ==========================================================
const TELEGRAM_BOT_TOKEN = '7402821846:AAEa2KG03G0d95o0Gf5EOYaqVJKD74i31r0';
const TELEGRAM_CHAT_ID = '1083305963';
// ==========================================================

// --- PENYIMPANAN STATUS TERAKHIR (UNTUK MENCEGAH SPAM) ---
// CATATAN: Variabel ini akan reset jika server restart.
// Untuk produksi jangka panjang, disarankan menyimpan ini di database.
let lastKnownStatus = 'Baik'; 

// --- INISIALISASI DATABASE ---
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: 'https://rf-bioflok-default-rtdb.asia-southeast1.firebasedatabase.app/'
});
const db = admin.database();

// --- FUNGSI BANTUAN ---
function parseDateString(dateString) {
  if (!dateString || !dateString.includes(' ')) return new Date().toISOString();
  const [datePart, timePart] = dateString.split(' ');
  const [day, month, year] = datePart.split('/');
  return `${year}-${month}-${day}T${timePart}`;
}

// --- RUTE API (Logika Inti Tidak Berubah) ---
app.get('/firebase-data', async (req, res) => {
  try {
    const sensorRef = db.ref('sensor_readings');
    const snapshot = await sensorRef.orderByKey().limitToLast(1).once('value');
    const data = snapshot.val();
    if (!data) return res.status(404).json({ error: 'Data sensor tidak ditemukan.' });

    const latestKey = Object.keys(data)[0];
    const latestData = data[latestKey];
    const latestSensorTimestamp = latestData.timestamp;

    const predictionResponseForUI = await fetch(PREDICTION_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(latestData)
    });
    if (!predictionResponseForUI.ok) throw new Error(`API Prediksi gagal untuk UI.`);
    const predictionResultForUI = await predictionResponseForUI.json();
    const currentStatus = predictionResultForUI.label;

    if (currentStatus === 'Buruk' && lastKnownStatus !== 'Buruk') {
        console.log("Kondisi buruk terdeteksi, mengirim notifikasi Telegram...");
        const message = `⚠️ *PERINGATAN KUALITAS AIR!* ⚠️\n\nStatus kolam saat ini: *${currentStatus}*\n\nMohon segera diperiksa.\n- Suhu: ${latestData.temperature.toFixed(1)}°C\n- pH: ${latestData.pH.toFixed(2)}\n- Kekeruhan: ${latestData.turbidity.toFixed(0)} NTU`;
        const telegramApiUrl = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;
        
        fetch(telegramApiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                chat_id: TELEGRAM_CHAT_ID,
                text: message,
                parse_mode: 'Markdown'
            })
        }).catch(err => console.error("Gagal mengirim notifikasi Telegram:", err));
    }
    lastKnownStatus = currentStatus;

    const logRef = db.ref('logs'); 
    const lastLogSnapshot = await logRef.orderByChild('original_timestamp').limitToLast(1).once('value');
    const lastLogData = lastLogSnapshot.val();
    let lastLoggedTimestamp = lastLogData ? Object.values(lastLogData)[0].original_timestamp : null;

    if (latestSensorTimestamp !== lastLoggedTimestamp) {
        console.log(`Data baru (${latestSensorTimestamp}) terdeteksi, menyimpan ke /logs...`);
        const newLogEntry = {
            date: latestSensorTimestamp.split(' ')[0],
            time: latestSensorTimestamp.split(' ')[1],
            temperature: latestData.temperature,
            pH: latestData.pH,
            turbidity: latestData.turbidity,
            status: currentStatus,
            original_timestamp: latestSensorTimestamp,
            server_timestamp: admin.database.ServerValue.TIMESTAMP
        };
        logRef.push(newLogEntry);
    } else {
        console.log(`Data duplikat (${latestSensorTimestamp}) terdeteksi, skip penyimpanan log.`);
    }
    
    res.json({
      raw_data: latestData,
      classification: currentStatus,
      timestamp: parseDateString(latestSensorTimestamp)
    });

  } catch (error) {
    console.error('Error di /firebase-data:', error);
    res.status(500).json({ error: 'Gagal memproses permintaan data.' });
  }
});

app.get('/firebase-logs', async (req, res) => {
    try {
        const logRef = db.ref('logs');
        const logSnapshot = await logRef.orderByChild('server_timestamp').limitToLast(50).once('value');
        const logData = logSnapshot.val() || {};
        const logsArray = Object.values(logData).sort((a, b) => b.server_timestamp - a.server_timestamp);
        res.json(logsArray);
    } catch (error) {
        console.error('Error di /firebase-logs:', error);
        res.status(500).json({ error: 'Gagal mengambil data log.' });
    }
});

// --- SERVER START ---
app.listen(port, () => console.log(`Server utama berjalan di http://localhost:${port}`));
