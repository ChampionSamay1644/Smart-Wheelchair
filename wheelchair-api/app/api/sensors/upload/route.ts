import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';
import { KILLSWITCH_ENABLED, validateSensorData } from '@/lib/config';

export async function POST(request: NextRequest) {
  try {
    // Check killswitch
    if (KILLSWITCH_ENABLED) {
      return NextResponse.json(
        { error: 'Service temporarily disabled', killswitch: true },
        { status: 503 }
      );
    }

    // Parse and validate request body
    const data = await request.json();
    
    if (!validateSensorData(data)) {
      return NextResponse.json(
        { error: 'Invalid sensor data format' },
        { status: 400 }
      );
    }

    // Add server timestamp
    const payload = {
      ...data,
      serverTimestamp: Date.now(),
    };

    // Store in Firebase Realtime Database
    const deviceRef = db.ref(`devices/${data.deviceId}`);
    
    // Update current sensor data
    await deviceRef.child('current').set(payload);
    
    // Add to history (keep last 100 entries)
    const historyRef = deviceRef.child('history');
    await historyRef.push(payload);
    
    // Cleanup old history entries (keep only last 100)
    const snapshot = await historyRef.orderByChild('timestamp').limitToLast(101).once('value');
    const count = snapshot.numChildren();
    
    if (count > 100) {
      const firstKey = Object.keys(snapshot.val() || {})[0];
      if (firstKey) {
        await historyRef.child(firstKey).remove();
      }
    }

    return NextResponse.json({
      success: true,
      timestamp: payload.serverTimestamp,
    });
  } catch (error: any) {
    console.error('Error uploading sensor data:', error);
    return NextResponse.json(
      { error: 'Internal server error', message: error.message },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    // Check killswitch
    if (KILLSWITCH_ENABLED) {
      return NextResponse.json(
        { error: 'Service temporarily disabled', killswitch: true },
        { status: 503 }
      );
    }

    const { searchParams } = new URL(request.url);
    const deviceId = searchParams.get('deviceId');

    if (!deviceId) {
      return NextResponse.json(
        { error: 'deviceId parameter required' },
        { status: 400 }
      );
    }

    // Fetch latest sensor data
    const snapshot = await db.ref(`devices/${deviceId}/current`).once('value');
    const data = snapshot.val();

    if (!data) {
      return NextResponse.json(
        { error: 'No data found for device' },
        { status: 404 }
      );
    }

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error fetching sensor data:', error);
    return NextResponse.json(
      { error: 'Internal server error', message: error.message },
      { status: 500 }
    );
  }
}
