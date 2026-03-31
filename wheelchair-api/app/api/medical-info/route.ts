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
        const { deviceId, info } = data;

        if (!deviceId || !info) {
            return NextResponse.json(
                { error: 'deviceId and info are required' },
                { status: 400 }
            );
        }

        // Update in Firebase under medical_info/{deviceId} (merges top-level fields)
        await db.ref(`medical_info/${deviceId}`).update({
            ...info,
            lastUpdated: Date.now()
        });

        return NextResponse.json({ success: true });
    } catch (error: any) {
        console.error('Error updating medical info:', error);
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

        if (!deviceId) {
            return NextResponse.json(
                { error: 'deviceId parameter required' },
                { status: 400 }
            );
        }

        const snapshot = await db.ref(`medical_info/${deviceId}`).once('value');
        const data = snapshot.val();

        if (!data) {
            return NextResponse.json({}, { status: 404 });
        }

        return NextResponse.json(data);
    } catch (error: any) {
        console.error('Error fetching medical info:', error);
        return NextResponse.json(
            { error: 'Internal server error', message: error.message },
            { status: 500 }
        );
    }
}
