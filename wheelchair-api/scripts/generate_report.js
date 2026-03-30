const admin = require('firebase-admin');
const fs = require('fs');
const path = require('path');

// --- MANUAL ENV LOADER (Failsafe for local Node scripts) ---
function loadEnv() {
    const envPath = path.resolve(process.cwd(), '.env.local');
    if (fs.existsSync(envPath)) {
        const content = fs.readFileSync(envPath, 'utf8');
        content.split('\n').forEach(line => {
            const match = line.match(/^\s*([\w.-]+)\s*=\s*(.*)$/);
            if (match) {
                let [_, key, value] = match;
                // Remove existing outer quotes if present
                value = value.trim();
                if (value.startsWith('"') && value.endsWith('"')) value = value.slice(1, -1);
                if (value.startsWith("'") && value.endsWith("'")) value = value.slice(1, -1);
                // Set it in process.env if not already set
                if (!process.env[key]) process.env[key] = value;
            }
        });
    }
}
loadEnv();
// -----------------------------------------------------------

// Initialize Firebase Admin (Using cert for local/prod compatibility)
if (!admin.apps.length) {
    const projectId = process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID?.trim();
    const clientEmail = process.env.FIREBASE_CLIENT_EMAIL?.trim();
    const databaseURL = process.env.NEXT_PUBLIC_FIREBASE_DATABASE_URL?.trim();
    let privateKey = process.env.FIREBASE_PRIVATE_KEY || '';
    
    // Handle Vercel-style escaping and double-quotes
    privateKey = privateKey.replace(/\\n/g, '\n').trim();
    while (privateKey.startsWith('"') && privateKey.endsWith('"')) {
        privateKey = privateKey.slice(1, -1).trim();
    }

    if (!projectId || !clientEmail || !privateKey || !databaseURL) {
        console.warn("⚠️ Warning: Missing Firebase environment variables:");
        if (!projectId) console.warn("  - NEXT_PUBLIC_FIREBASE_PROJECT_ID is missing");
        if (!clientEmail) console.warn("  - FIREBASE_CLIENT_EMAIL is missing");
        if (!privateKey) console.warn("  - FIREBASE_PRIVATE_KEY is missing");
        if (!databaseURL) console.warn("  - NEXT_PUBLIC_FIREBASE_DATABASE_URL is missing");
        
        console.warn("\nAttempting default credentials...");
        admin.initializeApp({
            credential: admin.credential.applicationDefault(),
            databaseURL: databaseURL || "https://smart-wheelchair-5517b-default-rtdb.asia-southeast1.firebasedatabase.app"
        });
    } else {
        admin.initializeApp({
            credential: admin.credential.cert({ projectId, clientEmail, privateKey }),
            databaseURL
        });
    }
}

const db = admin.database();

async function generateReport() {
    console.log("\n📊 GENERATING SMART WHEELCHAIR SYSTEM REPORT (MERGE VIEW)");
    console.log("==========================================================");

    try {
        // 1. Fetch all users (from Authentication and Sync node)
        const usersSnapshot = await db.ref('users').once('value');
        const users = usersSnapshot.val() || {};

        // 2. Fetch all devices
        const devicesSnapshot = await db.ref('devices').once('value');
        const devices = devicesSnapshot.val() || {};

        const report = [];

        for (const deviceId in devices) {
            const device = devices[deviceId];
            const metadata = device.metadata || {};
            
            // Find patient/guardian linked to this device in sessions or users
            let patientName = "Unknown";
            let guardianName = "Unknown";

            // Check sessions for currently connected users
            const sessions = device.sessions || {};
            for (const sessionId in sessions) {
                const session = sessions[sessionId];
                if (session.role === 'patient') patientName = session.name || patientName;
                if (session.role === 'guardian') guardianName = session.name || guardianName;
            }

            report.push({
                'Device ID': deviceId,
                'Patient Email': metadata.patientEmail || 'N/A',
                'Patient Pwd': metadata.patientPassword || metadata.password || 'N/A',
                'Guardian Email': metadata.guardianEmail || 'N/A',
                'Guardian Pwd': metadata.guardianPassword || metadata.password || 'N/A',
                'Last Signal': device.current?.timestamp ? new Date(device.current.timestamp).toLocaleString() : 'No Signal',
                'Reg At': metadata.registeredAt ? new Date(metadata.registeredAt).toLocaleString() : 'N/A'
            });
        }

        if (report.length === 0) {
            console.log("\n❌ No active devices or users found in the database.");
        } else {
            console.table(report);
        }

        console.log("\n✅ Report Complete.");
        process.exit(0);
    } catch (error) {
        console.error("\n❌ Error generating report:", error);
        process.exit(1);
    }
}

generateReport();
