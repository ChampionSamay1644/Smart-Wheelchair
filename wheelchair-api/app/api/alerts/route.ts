import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const { deviceId, title, message } = data;

    if (!deviceId) {
      return NextResponse.json({ error: 'Device ID required' }, { status: 400 });
    }

    console.log(`🚨 [ALERT TRIGGERED] Device: ${deviceId}, Title: ${title}, Message: ${message}`);

    const alertRef = db.ref(`devices/${deviceId}/alerts`);
    const newAlert = {
      title: title || 'Alert',
      message: message || '',
      timestamp: Date.now(),
      read: false,
    };

    await alertRef.push(newAlert);

    return NextResponse.json({ success: true, alert: newAlert });
  } catch (error: any) {
    console.error('Error creating alert:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const deviceId = searchParams.get('deviceId');

    if (!deviceId) {
      return NextResponse.json({ error: 'Device ID required' }, { status: 400 });
    }

    const snapshot = await db.ref(`devices/${deviceId}/alerts`).limitToLast(20).once('value');
    const alerts = snapshot.val() || {};
    
    const alertList = Object.entries(alerts).map(([id, data]: [string, any]) => ({
      id,
      ...data
    })).reverse();

    return NextResponse.json(alertList);
  } catch (error: any) {
    console.error('Error fetching alerts:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
