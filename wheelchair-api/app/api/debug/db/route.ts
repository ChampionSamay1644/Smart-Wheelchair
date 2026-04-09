import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';

export async function GET(request: NextRequest) {
  const result: any = {
    timestamp: new Date().toISOString(),
    envVars: {
      projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID ? `Set (${process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID})` : '❌ MISSING',
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL ? `Set (${process.env.FIREBASE_CLIENT_EMAIL?.substring(0, 20)}...)` : '❌ MISSING',
      privateKey: process.env.FIREBASE_PRIVATE_KEY ? `Set (${process.env.FIREBASE_PRIVATE_KEY.length} chars)` : '❌ MISSING',
      databaseUrl: process.env.NEXT_PUBLIC_FIREBASE_DATABASE_URL ? `Set (${process.env.NEXT_PUBLIC_FIREBASE_DATABASE_URL})` : '❌ MISSING',
    }
  };

  try {
    // Try writing a test value to Firebase
    const testRef = db.ref('debug_test');
    await testRef.set({
      lastCheck: Date.now(),
      status: 'API successfully connected to Firebase!',
      writtenAt: new Date().toISOString(),
    });

    // Try reading it back
    const snapshot = await testRef.once('value');
    const readBack = snapshot.val();

    result.firebaseWrite = '✅ SUCCESS';
    result.firebaseRead = readBack ? '✅ SUCCESS' : '❌ FAILED';
    result.message = 'Database connection is WORKING. Check your Firebase console for a debug_test node.';

    return NextResponse.json(result, { status: 200 });
  } catch (error: any) {
    result.firebaseWrite = '❌ FAILED';
    result.error = error.message;
    result.message = 'Database connection FAILED. See error above.';

    // Common error hints
    if (error.message?.includes('private key')) {
      result.hint = 'FIREBASE_PRIVATE_KEY may be malformed. Ensure newlines are \\n and the key is wrapped in double quotes in Vercel.';
    } else if (error.message?.includes('projectId')) {
      result.hint = 'NEXT_PUBLIC_FIREBASE_PROJECT_ID appears to be wrong or missing.';
    } else if (error.message?.includes('databaseURL')) {
      result.hint = 'NEXT_PUBLIC_FIREBASE_DATABASE_URL is wrong. Should be: https://smartnav-14012-default-rtdb.firebaseio.com';
    }

    return NextResponse.json(result, { status: 500 });
  }
}
