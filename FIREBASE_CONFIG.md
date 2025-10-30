# Firebase Configuration

This document outlines the Firebase configuration for the Smart Wheelchair app, including security rules and Cloud Functions.

## Firestore Security Rules

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Helper functions
    function isAuthenticated() {
      return request.auth != null;
    }
    
    function isOwner(userId) {
      return request.auth.uid == userId;
    }
    
    function isGuardianOf(patientId) {
      let guardian = get(/databases/$(database)/documents/users/$(request.auth.uid));
      return guardian.data.userType == 'guardian' && 
             patientId in guardian.data.patientIds;
    }
    
    function isPatientOf(guardianId) {
      let patient = get(/databases/$(database)/documents/users/$(request.auth.uid));
      return patient.data.userType == 'patient' && 
             patient.data.guardianId == guardianId;
    }

    // Users collection
    match /users/{userId} {
      allow read: if isAuthenticated() && 
        (isOwner(userId) || 
         isGuardianOf(userId) || 
         isPatientOf(userId));
      allow create: if isAuthenticated() && isOwner(userId);
      allow update: if isAuthenticated() && 
        (isOwner(userId) || 
         resource.data.guardianId == request.auth.uid);
    }

    // Invites collection
    match /invites/{inviteId} {
      allow read: if isAuthenticated();
      allow create: if isAuthenticated() && 
        get(/databases/$(database)/documents/users/$(request.auth.uid)).data.userType == 'guardian';
      allow update: if isAuthenticated() && 
        (request.resource.data.guardianId == request.auth.uid ||
         resource.data.guardianId == request.auth.uid);
    }

    // Locations collection
    match /locations/{patientId} {
      allow read: if isAuthenticated() && 
        (isOwner(patientId) || isGuardianOf(patientId));
      allow write: if isAuthenticated() && isOwner(patientId);
    }

    // Health readings collection
    match /health_readings/{patientId} {
      match /readings/{readingId} {
        allow read: if isAuthenticated() && 
          (isOwner(patientId) || isGuardianOf(patientId));
        allow write: if isAuthenticated() && isOwner(patientId);
      }
    }

    // Health reports collection
    match /health_reports/{reportId} {
      allow read: if isAuthenticated() && 
        (isOwner(resource.data.patientId) || 
         isGuardianOf(resource.data.patientId));
      allow create: if isAuthenticated() && 
        (isOwner(request.resource.data.patientId) || 
         isGuardianOf(request.resource.data.patientId));
    }

    // Alerts collection
    match /alerts/{alertId} {
      allow read: if isAuthenticated() && 
        (isOwner(resource.data.patientId) || 
         resource.data.guardianId == request.auth.uid);
      allow create: if isAuthenticated() && 
        (isOwner(request.resource.data.patientId) || 
         request.resource.data.guardianId == request.auth.uid);
    }
  }
}
```

## Cloud Functions

Create the following TypeScript functions in your Firebase project:

```typescript
import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';

admin.initializeApp();

const db = admin.firestore();
const messaging = admin.messaging();

// Handle patient-guardian linking via invite codes
export const acceptInvite = functions.https.onCall(async (data, context) => {
  if (!context.auth) {
    throw new functions.https.HttpsError(
      'unauthenticated',
      'User must be authenticated'
    );
  }

  const { inviteCode } = data;
  const patientId = context.auth.uid;

  const inviteRef = db.collection('invites').doc(inviteCode);
  const inviteDoc = await inviteRef.get();

  if (!inviteDoc.exists) {
    throw new functions.https.HttpsError('not-found', 'Invalid invite code');
  }

  const invite = inviteDoc.data()!;
  if (invite.used || invite.expiresAt.toDate() < new Date()) {
    throw new functions.https.HttpsError(
      'failed-precondition',
      'Invite code is expired or already used'
    );
  }

  const batch = db.batch();

  // Update patient's guardian
  const patientRef = db.collection('users').doc(patientId);
  batch.update(patientRef, { guardianId: invite.guardianId });

  // Update guardian's patients list
  const guardianRef = db.collection('users').doc(invite.guardianId);
  batch.update(guardianRef, {
    patientIds: admin.firestore.FieldValue.arrayUnion(patientId),
  });

  // Mark invite as used
  batch.update(inviteRef, { used: true });

  await batch.commit();

  // Send FCM notification to guardian
  const guardianDoc = await guardianRef.get();
  const guardianData = guardianDoc.data()!;
  if (guardianData.fcmToken) {
    await messaging.send({
      token: guardianData.fcmToken,
      notification: {
        title: 'New Patient Connected',
        body: `A new patient has accepted your invitation`,
      },
      data: {
        type: 'PATIENT_LINKED',
        patientId,
      },
    });
  }

  return { success: true };
});

// Send FCM notifications when new alerts are created
export const onAlertCreated = functions.firestore
  .document('alerts/{alertId}')
  .onCreate(async (snap, context) => {
    const alert = snap.data();
    const guardianId = alert.guardianId;

    const guardianDoc = await db.collection('users').doc(guardianId).get();
    const guardianData = guardianDoc.data();

    if (guardianData?.fcmToken) {
      await messaging.send({
        token: guardianData.fcmToken,
        notification: {
          title: alert.title,
          body: alert.body,
        },
        data: {
          type: alert.type,
          alertId: context.params.alertId,
          patientId: alert.patientId,
        },
      });
    }
  });

// Clean up expired invite codes
export const cleanupExpiredInvites = functions.pubsub
  .schedule('every 24 hours')
  .onRun(async (context) => {
    const now = admin.firestore.Timestamp.now();
    const expiredInvites = await db
      .collection('invites')
      .where('expiresAt', '<', now)
      .where('used', '==', false)
      .get();

    const batch = db.batch();
    expiredInvites.docs.forEach((doc) => {
      batch.update(doc.ref, { used: true });
    });

    if (expiredInvites.size > 0) {
      await batch.commit();
    }

    return null;
  });

// Update location history when location changes
export const onLocationUpdate = functions.firestore
  .document('locations/{patientId}')
  .onWrite(async (change, context) => {
    const patientId = context.params.patientId;
    const newData = change.after.data();
    
    if (!newData) return null; // Location was deleted

    // Store in location history
    await db
      .collection('locations_history')
      .doc(patientId)
      .collection('updates')
      .add({
        ...newData,
        timestamp: admin.firestore.FieldValue.serverTimestamp(),
      });

    // Check for any geofence violations
    const userDoc = await db.collection('users').doc(patientId).get();
    const userData = userDoc.data();

    if (userData?.geofence) {
      const position = newData.position;
      const geofence = userData.geofence;

      // Simple circular geofence check
      const distance = calculateDistance(
        position.latitude,
        position.longitude,
        geofence.center.latitude,
        geofence.center.longitude
      );

      if (distance > geofence.radius) {
        // Create geofence violation alert
        await db.collection('alerts').add({
          patientId,
          guardianId: userData.guardianId,
          type: 'GEOFENCE_VIOLATION',
          title: 'Geofence Violation',
          body: 'Patient has left the designated safe area',
          data: {
            position,
            geofence,
            distance,
          },
          createdAt: admin.firestore.FieldValue.serverTimestamp(),
        });
      }
    }
  });

function calculateDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const R = 6371e3; // Earth's radius in meters
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const Δφ = ((lat2 - lat1) * Math.PI) / 180;
  const Δλ = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
    Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c; // Distance in meters
}
```

## Storage Rules

```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /health_reports/{patientId}/{reportId} {
      allow read: if request.auth != null && (
        request.auth.uid == patientId || 
        exists(/databases/$(database)/documents/users/$(request.auth.uid)/patientIds/$(patientId))
      );
      allow write: if request.auth != null && (
        request.auth.uid == patientId || 
        exists(/databases/$(database)/documents/users/$(request.auth.uid)/patientIds/$(patientId))
      );
    }
  }
}
```

## Schema Overview

### Collections:

1. `users/{userId}`
   ```typescript
   interface User {
     uid: string;
     email: string;
     name: string;
     userType: 'patient' | 'guardian';
     guardianId?: string; // For patients
     patientIds?: string[]; // For guardians
     fcmToken?: string;
     geofence?: {
       center: { latitude: number; longitude: number };
       radius: number; // meters
     };
     healthData?: {
       [key: string]: any;
     };
     preferences?: {
       [key: string]: any;
     };
     createdAt: Timestamp;
   }
   ```

2. `invites/{inviteCode}`
   ```typescript
   interface Invite {
     code: string;
     guardianId: string;
     guardianName: string;
     guardianEmail?: string;
     expiresAt: Timestamp;
     used: boolean;
     createdAt: Timestamp;
   }
   ```

3. `locations/{patientId}`
   ```typescript
   interface Location {
     position: { latitude: number; longitude: number };
     speed?: number;
     heading?: number;
     accuracy?: number;
     timestamp: Timestamp;
   }
   ```

4. `locations_history/{patientId}/updates/{updateId}`
   ```typescript
   interface LocationUpdate extends Location {
     // Inherits all Location fields
   }
   ```

5. `health_readings/{patientId}/readings/{readingId}`
   ```typescript
   interface HealthReading {
     timestamp: Timestamp;
     pulseRate?: number;
     bodyTemp?: number;
     bloodOxygen?: number;
     [key: string]: any; // Other health metrics
   }
   ```

6. `health_reports/{reportId}`
   ```typescript
   interface HealthReport {
     patientId: string;
     patientName: string;
     startDate: Timestamp;
     endDate: Timestamp;
     downloadUrl?: string;
     summary: {
       dataPoints: number;
       averages: { [metric: string]: number };
       ranges: { [metric: string]: { min: number; max: number } };
     };
     createdAt: Timestamp;
   }
   ```

7. `alerts/{alertId}`
   ```typescript
   interface Alert {
     patientId: string;
     guardianId: string;
     type: string;
     title: string;
     body: string;
     data?: any;
     read: boolean;
     createdAt: Timestamp;
   }
   ```

## Implementation Steps

1. Deploy Firestore Security Rules
2. Deploy Storage Rules
3. Initialize Firebase project with `firebase init`
4. Deploy Cloud Functions
5. Update your Flutter app's Firebase configuration
6. Test all critical paths:
   - User registration (both roles)
   - Invite code generation and acceptance
   - Location updates and tracking
   - Health data recording
   - Alert generation and delivery
   - Report generation and download