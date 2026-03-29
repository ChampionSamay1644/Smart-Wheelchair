import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';
import { KILLSWITCH_ENABLED, validateSensorData } from '@/lib/config';

export async function POST(request: NextRequest) {
  try {
    if (KILLSWITCH_ENABLED) {
      return NextResponse.json(
        { error: 'Service temporarily disabled', killswitch: true },
        { status: 503 }
      );
    }

    const data = await request.json();
    const { dht11, max30100, motorStatus } = data;
    const deviceId = request.headers.get('X-Device-Id') || data.deviceId;
    const patientPassword = request.headers.get('X-Patient-Password') || 'pat_123';
    const guardianPassword = request.headers.get('X-Guardian-Password') || 'gua_123';

    if (!deviceId) {
      return NextResponse.json(
        { error: 'Device ID is required' },
        { status: 401 }
      );
    }

    if (!validateSensorData(data)) {
      return NextResponse.json(
        { error: 'Invalid sensor data format' },
        { status: 400 }
      );
    }

    // Verify or Register Device Password
    const deviceRef = db.ref(`devices/${deviceId}`);
    const metadataSnapshot = await deviceRef.child('metadata').once('value');
    const metadata = metadataSnapshot.val();

    if (!metadata) {
      // Register new device passwords! 
      await deviceRef.child('metadata').set({
        patientPassword,
        guardianPassword,
        registeredAt: Date.now()
      });
    } else {
      // Validate device-level authorization simply by requiring *either* password to be correct to upload
      // Technically, physical wheelchair just uploads, but this ensures a stranger doesn't maliciously spam the DB.
      if (metadata.patientPassword !== patientPassword && metadata.guardianPassword !== guardianPassword && metadata.password !== request.headers.get('X-Device-Password')) {
         console.warn(`Unauthorized upload attempt for ${deviceId}`);
         return NextResponse.json({ error: 'Unauthorized: Incorrect device credentials' }, { status: 401 });
      }
    }

    // Add server timestamp
    const payload = {
      ...data,
      deviceId,
      serverTimestamp: Date.now(),
    };

    // Update current sensor data
    console.log(`📡 Incoming from ${deviceId}: Pulse: ${max30100?.ir / 100}, SpO2: ${max30100?.red / 100}`);
    await deviceRef.child('current').update(payload);

    // Add to history (keep last 100 entries)
    const historyRef = deviceRef.child('history');
    await historyRef.push(payload);

    // Cleanup old history entries
    const snapshot = await historyRef.orderByChild('timestamp').limitToLast(101).once('value');
    if (snapshot.numChildren() > 100) {
      const firstKey = Object.keys(snapshot.val() || {})[0];
      if (firstKey) await historyRef.child(firstKey).remove();
    }

    // --- MONITORING ---
    let alertMessage = "";

    // 1. Health Monitoring
    if (max30100) {
      const pulse = max30100.ir / 100;
      const spo2 = max30100.red / 100;
      
      if (pulse > 120) alertMessage = `Critical High Pulse: ${pulse.toFixed(0)} BPM`;
      else if (pulse < 50 && pulse > 0) alertMessage = `Critical Low Pulse: ${pulse.toFixed(0)} BPM`;
      else if (spo2 < 90 && spo2 > 0) alertMessage = `Critical Low SpO2: ${spo2.toFixed(0)}%`;
    }

    if (!alertMessage && dht11) {
      if (dht11.temperature > 38) alertMessage = `High Body Temperature: ${dht11.temperature.toFixed(1)}°C`;
    }

    // 2. Battery Monitoring
    if (!alertMessage && motorStatus && motorStatus.battery) {
      if (motorStatus.battery < 15) alertMessage = `Low Battery: ${motorStatus.battery.toFixed(1)}% Remaining`;
    }

    if (alertMessage) {
      console.log(`🚨 HEALTH ALERT for ${deviceId}: ${alertMessage}`);
      
      // 1. Create entry in Database Alert node
      const alertRef = db.ref(`devices/${deviceId}/alerts`);
      const newAlert = {
        id: Date.now().toString(),
        title: "Health Emergency",
        message: alertMessage,
        timestamp: Date.now(),
        read: false,
        severity: "high"
      };
      await alertRef.push(newAlert);

      // 2. Fetch Guardian email to notify
      const sessionSnapshot = await db.ref(`devices/${deviceId}/sessions`).orderByChild('role').equalTo('guardian').once('value');
      const sessions = sessionSnapshot.val();
      
      if (sessions) {
        // Send email to all connected guardians
        for (const key in sessions) {
          const guardian = sessions[key];
          if (guardian.email || (guardian.name && guardian.name.includes('@'))) {
            const targetEmail = guardian.email || guardian.name;
            
            // Trigger internal email API
            fetch(`${request.nextUrl.origin}/api/notifications/email`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                type: 'health_alert',
                targetEmail: targetEmail,
                patientName: metadata?.patientName || deviceId,
                contextEmail: alertMessage
              })
            }).catch(err => console.error("Failed to trigger alert email:", err));
          }
        }
      }
    }
    // --- END MONITORING ---

    return NextResponse.json({ success: true, timestamp: payload.serverTimestamp });
  } catch (error: any) {
    console.error('Error uploading sensor data:', error);
    return NextResponse.json({ error: 'Internal server error', message: error.message }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  try {
    if (KILLSWITCH_ENABLED) {
      return NextResponse.json({ error: 'Service temporarily disabled', killswitch: true }, { status: 503 });
    }

    const { searchParams } = new URL(request.url);
    const deviceId = request.headers.get('X-Device-Id') || searchParams.get('deviceId');
    const password = request.headers.get('X-Device-Password');

    if (!deviceId || !password) {
      return NextResponse.json({ error: 'Device ID and Password required' }, { status: 401 });
    }

    // Verify Password
    const deviceRef = db.ref(`devices/${deviceId}`);
    const metadataSnapshot = await deviceRef.child('metadata').once('value');
    const metadata = metadataSnapshot.val();

    if (!metadata) {
      return NextResponse.json({ error: 'Device not found or not registered' }, { status: 404 });
    }

    if (metadata.password !== password) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const history = searchParams.get('history') === 'true';
    const checkOnly = searchParams.get('checkOnly') === 'true';

    if (checkOnly) {
      return NextResponse.json({ exists: true });
    }

    if (history) {
      const snapshot = await deviceRef.child('history').orderByChild('timestamp').limitToLast(100).once('value');
      const historyData = snapshot.val();
      if (!historyData) return NextResponse.json([]);

      const historyArray = Object.values(historyData).sort((a: any, b: any) =>
        (b.timestamp || 0) - (a.timestamp || 0)
      );
      return NextResponse.json(historyArray);
    }

    const snapshot = await deviceRef.child('current').once('value');
    const data = snapshot.val();
    if (!data) return NextResponse.json({ error: 'No data found' }, { status: 404 });

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error fetching sensor data:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
