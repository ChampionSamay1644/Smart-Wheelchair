import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { getDatabase } from 'firebase-admin/database';

function getPrivateKey(): string {
  const raw = process.env.FIREBASE_PRIVATE_KEY || '';
  
  // Strip surrounding quotes if Vercel added them
  let key = raw.startsWith('"') && raw.endsWith('"') ? raw.slice(1, -1) : raw;
  
  // Handle both \\n (double-escaped) and \n (single-escaped)
  key = key.replace(/\\n/g, '\n');
  
  return key;
}

// Initialize Firebase Admin (singleton for serverless)
if (!getApps().length) {
  const projectId = process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID;
  const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
  const privateKey = getPrivateKey();
  const databaseURL = process.env.NEXT_PUBLIC_FIREBASE_DATABASE_URL;

  if (!projectId || !clientEmail || !privateKey || !databaseURL) {
    console.error('❌ Firebase Admin: Missing environment variables', {
      projectId: !!projectId,
      clientEmail: !!clientEmail,
      privateKey: !!privateKey,
      databaseURL: !!databaseURL,
    });
    throw new Error('Missing Firebase environment variables. Check Vercel settings.');
  }

  initializeApp({
    credential: cert({ projectId, clientEmail, privateKey }),
    databaseURL,
  });
}

export const db = getDatabase();
