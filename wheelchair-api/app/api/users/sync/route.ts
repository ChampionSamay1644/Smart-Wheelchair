import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';

export async function POST(request: NextRequest) {
  try {
    const { uid, userData } = await request.json();

    if (!uid) {
      return NextResponse.json({ error: 'UID is required' }, { status: 400 });
    }

    // Sync authentication account to Realtime Database 'users/' tree
    const userRef = db.ref(`users/${uid}`);
    
    // Use .update() to preserve existing fields while adding new ones
    await userRef.update({
      ...userData,
      lastSync: Date.now(),
    });

    console.log(`✅ USER SYNC: UID=${uid} synced to RTDB`);

    return NextResponse.json({ success: true });
  } catch (error: any) {
    console.error('Error syncing user:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
