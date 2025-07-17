// Memuat environment variables dari file .env di paling atas
require('dotenv').config();

const express = require('express');
const admin = require('firebase-admin');
const fetch = require('node-fetch');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 8080;

// ==========================================================
// ===   BAGIAN 1: KONFIGURASI AMAN (dari file .env)      ===
// ==========================================================
// Mengambil konfigurasi dari environment variables untuk keamanan
const PREDICTION_API_URL = process.env.PREDICTION_API_URL;
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;
const FIREBASE_DATABASE_URL = process.env.FIREBASE_DATABASE_URL;

// Middleware CORS, hanya izinkan koneksi dari frontend Anda
// Daftar semua domain yang diizinkan untuk mengakses server ini
const allowedOrigins = [
  'https://rf-bioflok.web.app', // Frontend Anda di Firebase
  'https://bioflok-api.trycloudflare.com' // Ganti dengan URL Cloudflare Anda
];

app.use(cors({
  origin: function (origin, callback) {
    // Izinkan jika origin ada di dalam daftar 'allowedOrigins'
    if (!origin || allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Akses diblokir oleh kebijakan CORS'));
    }
  }
}));

// --- INISIALISASI FIREBASE ---
// Firebase Admin SDK akan otomatis menggunakan GOOGLE_APPLICATION_CREDENTIALS dari .env
admin.initializeApp({
  databaseURL: FIREBASE_DATABASE_URL
});
const db = admin.database();

// --- BANTUAN FORMAT TANGGAL ---
// Fungsi ini tetap berguna untuk frontend
function parseDateString(dateString) {
  if (!dateString || !dateString.includes(' ')) return new Date().toISOString();
  const [datePart, timePart] = dateString.split(' ');
  const [day, month, year] = datePart.split('/');
  return `${year}-${month}-${day}T${timePart}`;
}

// === ENDPOINT UTAMA ===
app.get('/firebase-data', async (req, res) => {
  try {
    // 1. Ambil data sensor terakhir
    const sensorRef = db.ref('sensor_readings');
    const snapshot = await sensorRef.orderByKey().limitToLast(1).once('value');
    const data = snapshot.val();
    if (!data) return res.status(404).json({ error: 'Data sensor tidak ditemukan.' });

    const latestKey = Object.keys(data)[0];
    const latestData = data[latestKey];
    const latestSensorTimestamp = latestData.timestamp;

    // 2. Panggil API prediksi untuk mendapatkan status saat ini
    const predictionResponseForUI = await fetch(PREDICTION_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(latestData)
    });
    if (!predictionResponseForUI.ok) throw new Error(`API Prediksi gagal untuk UI.`);
    const predictionResultForUI = await predictionResponseForUI.json();
    const currentStatus = predictionResultForUI.label;

    // ===================================================================
    // === BAGIAN 3: STATUS PERSISTEN UNTUK NOTIFIKASI YANG ANDAL      ===
    // ===================================================================
    // Ambil status terakhir yang tersimpan di database, bukan dari memori
    const statusRef = db.ref('server_status/lastKnown');
    const statusSnapshot = await statusRef.once('value');
    const lastKnownStatus = statusSnapshot.val() || 'Baik'; // Default 'Baik' jika belum ada

    // Kirim notifikasi HANYA jika status berubah menjadi 'Buruk'
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
        
        // Simpan status baru ke database agar tidak kirim notif lagi
        await statusRef.set('Buruk'); 
    } else if (currentStatus !== lastKnownStatus) {
        // Jika status berubah (misal dari Buruk ke Baik), update juga di DB
        await statusRef.set(currentStatus);
    }
    // ===================================================================

    // 4. Simpan data ke log jika merupakan data baru
    const logRef = db.ref('logs'); 
    const lastLogSnapshot = await logRef.orderByChild('original_timestamp').limitToLast(1).once('value');
    const lastLogData = lastLogSnapshot.val();
    let lastLoggedTimestamp = lastLogData ? Object.values(lastLogData)[0].original_timestamp : null;

    if (latestSensorTimestamp !== lastLoggedTimestamp) {
        console.log(`Data baru (${latestSensorTimestamp}) terdeteksi, menyimpan ke /logs...`);
        
        // Format tanggal YYYY-MM-DD untuk optimasi query
        const [datePart, timePart] = latestSensorTimestamp.split(' ');
        const [day, month, year] = datePart.split('/');
        const formattedDateForQuery = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;

        const newLogEntry = {
            date: formattedDateForQuery, // Format YYYY-MM-DD
            time: timePart,
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
    
    // 5. Kirim respons ke frontend
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

// ==========================================================
// === BAGIAN 2: ENDPOINT LOG YANG EFISIEN & TEROPTIMASI  ===
// ==========================================================
app.get('/firebase-logs', async (req, res) => {
    try {
        const { range, date } = req.query;
        // Gunakan query Firebase untuk memfilter data di sisi database, bukan di server
        let query = db.ref('logs').orderByChild('date');

        if (date) {
            // Filter berdasarkan tanggal spesifik (format YYYY-MM-DD)
            query = query.equalTo(date);
        } else if (range && range !== 'all') {
            // Filter berdasarkan rentang waktu
            const now = new Date();
            let startDate = new Date();
            if (range === '1d') startDate.setDate(now.getDate() - 1);
            if (range === '7d') startDate.setDate(now.getDate() - 7);
            if (range === '30d') startDate.setDate(now.getDate() - 30);
            
            const formattedStartDate = startDate.toISOString().split('T')[0]; // Konversi ke YYYY-MM-DD
            query = query.startAt(formattedStartDate);
        }

        const snapshot = await query.once('value');
        const logData = snapshot.val() || {};

        // Ubah object hasil query menjadi array dan urutkan di server (karena data sudah sedikit)
        const logsArray = Object.values(logData).sort((a, b) => {
            const aTime = new Date(`${a.date}T${a.time}`).getTime();
            const bTime = new Date(`${b.date}T${b.time}`).getTime();
            return bTime - aTime; // Urutkan dari yang terbaru ke terlama
        });

        res.json(logsArray);

    } catch (error) {
        console.error('Error di /firebase-logs:', error);
        res.status(500).json({ error: 'Gagal mengambil data log.' });
    }
});

// === START SERVER ===
app.listen(port, () => {
    console.log(`Server utama berjalan di http://localhost:${port}`);
    console.log(`Konfigurasi API Prediksi: ${PREDICTION_API_URL}`);
});
