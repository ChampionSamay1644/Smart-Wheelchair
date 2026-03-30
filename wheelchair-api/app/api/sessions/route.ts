import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';
import { KILLSWITCH_ENABLED } from '@/lib/config';

export async function POST(request: NextRequest) {
    try {
        if (KILLSWITCH_ENABLED) {
            return NextResponse.json(
                { error: 'Service temporarily disabled', killswitch: true },
                { status: 503 }
            );
        }

        const data = await request.json();
        const { deviceId, role, name } = data;

        if (!deviceId || !role) {
            return NextResponse.json(
                { error: 'deviceId and role are required' },
                { status: 400 }
            );
        }

        const timestamp = Date.now();
        const sessionData = {
            role,
            name: name || 'Unknown',
            lastPulse: timestamp,
            status: 'active'
        };

        // Store in Firebase under sessions/{deviceId}/{role}
        await db.ref(`sessions/${deviceId}/${role}`).set(sessionData);

        return NextResponse.json({ success: true, timestamp });
    } catch (error: any) {
        console.error('Error reporting session:', error);
        return NextResponse.json(
            { error: 'Internal server error', message: error.message },
            { status: 500 }
        );
    }
}

export async function GET(request: NextRequest) {
    try {
        if (KILLSWITCH_ENABLED) {
            return NextResponse.json(
                { error: 'Service temporarily disabled', killswitch: true },
                { status: 503 }
            );
        }

        const { searchParams } = new URL(request.url);
        const deviceId = searchParams.get('deviceId');

        if (deviceId) {
            const snapshot = await db.ref(`sessions/${deviceId}`).once('value');
            return NextResponse.json(snapshot.val() || { error: 'No active session' });
        }

        // List all active sessions
        const snapshot = await db.ref('sessions').once('value');
        const sessions = snapshot.val() || {};

        return NextResponse.json({
            sessions,
            count: Object.keys(sessions).length,
            timestamp: Date.now()
        });
    } catch (error: any) {
        console.error('Error fetching sessions:', error);
        return NextResponse.json(
            { error: 'Internal server error', message: error.message },
            { status: 500 }
        );
    }
}
