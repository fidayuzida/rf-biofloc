// Load environment variables dari file .env
require('dotenv').config();

const express = require('express');
const admin = require('firebase-admin');
const fetch = require('node-fetch');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 8080;

// === Load konfigurasi dari file .env ===
const PREDICTION_API_URL = process.env.PREDICTION_API_URL;
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;
const FIREBASE_DATABASE_URL = process.env.FIREBASE_DATABASE_URL;

// === CORS config ===
const allowedOrigins = [
  'https://rf-bioflok.web.app',
];

app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Akses diblokir oleh kebijakan CORS'));
    }
  }
}));

// === Inisialisasi Firebase dengan credential ===
const serviceAccount = require('./rf-bioflok-firebase-adminsdk-fbsvc-1310b6e533.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: FIREBASE_DATABASE_URL
});

const db = admin.database();

// === Parse tanggal untuk log frontend ===
function parseDateString(dateString) {
  if (!dateString || !dateString.includes(' ')) return new Date().toISOString();
  const [datePart, timePart] = dateString.split(' ');
  const [day, month, year] = datePart.split('/');
  return `${year}-${month}-${day}T${timePart}`;
}



// === Endpoint utama ===
app.get('/firebase-data', async (req, res) => {
  try {
    const sensorRef = db.ref('sensor_readings');
    const snapshot = await sensorRef.orderByKey().limitToLast(1).once('value');
    const data = snapshot.val();
    if (!data) return res.status(404).json({ error: 'Data sensor tidak ditemukan.' });

    const latestKey = Object.keys(data)[0];
    const latestData = data[latestKey];
    const latestSensorTimestamp = latestData.timestamp;

    // server.js - Kode BARU yang benar
    console.log(`LOG: Menghubungi API Prediksi di ${PREDICTION_API_URL} dengan metode GET...`);

    const predictionResponseForUI = await fetch(PREDICTION_API_URL, {
    method: 'GET' // Ubah ke GET, tidak perlu headers dan body
    });
    if (!predictionResponseForUI.ok) throw new Error('API Prediksi gagal.');

    const predictionResultForUI = await predictionResponseForUI.json();
    const currentStatus = predictionResultForUI.label;

    const statusRef = db.ref('server_status/lastKnown');
    const statusSnapshot = await statusRef.once('value');
    const lastKnownStatus = statusSnapshot.val() || 'Baik';

    if (currentStatus === 'Buruk' && lastKnownStatus !== 'Buruk') {
      console.log("Kondisi Buruk! Kirim notifikasi...");
      const message = `⚠️ *PERINGATAN KUALITAS AIR!* ⚠️\n\nStatus kolam: *${currentStatus}*\n- Suhu: ${latestData.temperature.toFixed(1)}°C\n- pH: ${latestData.pH.toFixed(2)}\n- Kekeruhan: ${latestData.turbidity.toFixed(0)} NTU`;
      const telegramApiUrl = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;

      fetch(telegramApiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: TELEGRAM_CHAT_ID,
          text: message,
          parse_mode: 'Markdown'
        })
      }).catch(err => console.error("Gagal kirim notifikasi Telegram:", err));

      await statusRef.set('Buruk');
    } else if (currentStatus !== lastKnownStatus) {
      await statusRef.set(currentStatus);
    }

    const logRef = db.ref('logs');
    const lastLogSnapshot = await logRef.orderByChild('original_timestamp').limitToLast(1).once('value');
    const lastLogData = lastLogSnapshot.val();
    const lastLoggedTimestamp = lastLogData ? Object.values(lastLogData)[0].original_timestamp : null;

    if (latestSensorTimestamp !== lastLoggedTimestamp) {
      console.log(`Data baru (${latestSensorTimestamp}) terdeteksi, simpan log...`);
      const [datePart, timePart] = latestSensorTimestamp.split(' ');
      const [day, month, year] = datePart.split('/');
      const formattedDate = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;

      const newLogEntry = {
        date: formattedDate,
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
      console.log(`Data duplikat (${latestSensorTimestamp}), lewati log.`);
    }

    res.json({
      raw_data: latestData,
      classification: currentStatus,
      timestamp: parseDateString(latestSensorTimestamp)
    });

  } catch (error) {
    console.error('Error di /firebase-data:', error);
    res.status(500).json({ error: 'Gagal memproses permintaan.' });
  }
});

// === Endpoint log ===
app.get('/firebase-logs', async (req, res) => {
  try {
    const { range, date } = req.query;
    let query = db.ref('logs').orderByChild('date');

    if (date) {
      query = query.equalTo(date);
    } else if (range && range !== 'all') {
      const now = new Date();
      let startDate = new Date();
      if (range === '1d') startDate.setDate(now.getDate() - 1);
      if (range === '7d') startDate.setDate(now.getDate() - 7);
      if (range === '30d') startDate.setDate(now.getDate() - 30);
      const formattedStart = startDate.toISOString().split('T')[0];
      query = query.startAt(formattedStart);
    }

    const snapshot = await query.once('value');
    const logData = snapshot.val() || {};
    const logsArray = Object.values(logData).sort((a, b) => {
      const aTime = new Date(`${a.date}T${a.time}`).getTime();
      const bTime = new Date(`${b.date}T${b.time}`).getTime();
      return bTime - aTime;
    });

    res.json(logsArray);
  } catch (error) {
    console.error('Error di /firebase-logs:', error);
    res.status(500).json({ error: 'Gagal mengambil data log.' });
  }
});

// === Start server ===
app.listen(port, () => {
  console.log(`Server utama berjalan di http://localhost:${port}`);
  console.log(`Konfigurasi API Prediksi: ${PREDICTION_API_URL}`);
});
