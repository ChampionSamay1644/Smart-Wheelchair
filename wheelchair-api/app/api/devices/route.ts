import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';

export async function GET(request: NextRequest) {
  try {
    const snapshot = await db.ref('devices').once('value');
    const devices = snapshot.val() || {};
    
    const deviceList = Object.keys(devices).map(deviceId => ({
      deviceId,
      lastUpdate: devices[deviceId].current?.serverTimestamp || null,
      online: devices[deviceId].current?.serverTimestamp 
        ? (Date.now() - devices[deviceId].current.serverTimestamp) < 30000 // online if updated in last 30s
        : false,
    }));

    return NextResponse.json({ devices: deviceList });
  } catch (error: any) {
    console.error('Error fetching devices:', error);
    return NextResponse.json(
      { error: 'Internal server error', message: error.message },
      { status: 500 }
    );
  }
}
