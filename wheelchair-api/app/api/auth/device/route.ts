import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const { deviceId, role, password } = data;

    if (!deviceId || !password || !role) {
      return NextResponse.json(
        { error: 'deviceId, role, and password are required' },
        { status: 400 }
      );
    }

    const deviceRef = db.ref(`devices/${deviceId}/metadata`);
    const snapshot = await deviceRef.once('value');
    const metadata = snapshot.val();

    if (!metadata) {
      return NextResponse.json(
        { error: `Device ${deviceId} not found or not registered` },
        { status: 404 }
      );
    }

    // Role-based password validation
    let valid = false;
    if (role === 'patient') {
      valid = metadata.patientPassword === password;
    } else if (role === 'guardian') {
      valid = metadata.guardianPassword === password;
    }

    if (!valid) {
      return NextResponse.json(
        { error: `Incorrect password for ${role} on device ${deviceId}` },
        { status: 401 }
      );
    }

    return NextResponse.json({
      success: true,
      deviceId,
      role,
      user: {
        id: deviceId, // Using device ID as user ID since we gutted email auth
        name: role === 'patient' ? 'Patient User' : 'Guardian User',
        email: `${role}@${deviceId}.local`, // Mock email for internal app structures
        role: role
      }
    });
  } catch (error: any) {
    console.error('Device Auth Error:', error);
    return NextResponse.json(
      { error: 'Internal server error', message: error.message },
      { status: 500 }
    );
  }
}
