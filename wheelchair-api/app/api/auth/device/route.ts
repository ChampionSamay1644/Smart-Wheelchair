import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/firebase-admin';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const { deviceId, role, password, email } = data;
    console.log(`🔐 [AUTH ATTEMPT] Role: ${role}, Device: ${deviceId}, Email: ${email}`);

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
      if (role === 'guardian') {
        return NextResponse.json(
          { error: `Access Denied: The Wheelchair ${deviceId} has not been registered by a patient yet.` },
          { status: 404 }
        );
      } else {
        // Patient Auto-Registration
        console.log(`🚀 Patient auto-registering new device: ${deviceId}`);
        await deviceRef.set({
          patientPassword: password,
          patientEmail: email || null,
          registeredAt: Date.now()
        });

        if (email) {
          // Trigger Welcome Patient Email
          console.log(`📨 [Email Trigger] Attempting Welcome Patient -> ${email}`);
          fetch(`${request.nextUrl.origin}/api/notifications/email`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              type: 'welcome_patient',
              targetEmail: email,
              userName: email.split('@')[0]
            })
          })
          .then(r => r.json())
          .then(d => console.log('📨 [Email Response] Success:', d))
          .catch(e => console.error('❌ [Email Error] Welcome Patient:', e));
        }
        
        console.log(`✅ [REGISTRATION SUCCESS] Role: ${role}, Device: ${deviceId}`);
        return NextResponse.json({
          success: true,
          role,
          deviceId,
          user: { id: deviceId, name: 'Patient User', email: email || `patient@${deviceId}.local`, role }
        });
      }
    }

    // Role-based password validation for an EXISTING device
    let valid = false;
    
    if (role === 'patient') {
      valid = metadata.patientPassword === password || metadata.password === password;
      if (valid && email && metadata.patientEmail !== email) {
        await deviceRef.update({ patientEmail: email });
      }
    } else if (role === 'guardian') {
      // If guardian password hasn't been set yet, let this first guardian set it!
      if (!metadata.guardianPassword) {
         // Fallback: If they use the existing patient/generic password, count it as the new guardian password too
         if (metadata.patientPassword === password || metadata.password === password) {
            valid = true;
         } else {
            await deviceRef.update({ 
              guardianPassword: password,
              guardianEmail: email || null
            });
            valid = true;
         }
      } else {
         valid = metadata.guardianPassword === password || metadata.password === password;
      }

      if (valid && email) {
        // ... (existing email fetch calls)
        // Notify the Patient that this Guardian is now looking out for them
        if (metadata.patientEmail) {
           console.log(`📨 [Email Trigger] Attempting Guardian Alert for Patient -> ${metadata.patientEmail}`);
           fetch(`${request.nextUrl.origin}/api/notifications/email`, {
             method: 'POST',
             headers: { 'Content-Type': 'application/json' },
             body: JSON.stringify({
               type: 'guardian_alert',
               targetEmail: metadata.patientEmail,
               contextEmail: email,
               userName: metadata.patientEmail.split('@')[0]
             })
           })
           .then(r => r.json())
           .then(d => console.log('📨 [Email Response] Success:', d))
           .catch(e => console.error('❌ [Email Error] Guardian Alert:', e));
        }

        // Welcome the Guardian too!
        console.log(`📨 [Email Trigger] Attempting Welcome Guardian -> ${email}`);
        fetch(`${request.nextUrl.origin}/api/notifications/email`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: 'welcome_guardian',
            targetEmail: email,
            patientName: metadata.patientEmail ? metadata.patientEmail.split('@')[0] : deviceId,
            userName: email.split('@')[0]
          })
        })
        .then(r => r.json())
        .then(d => console.log('📨 [Email Response] Success:', d))
        .catch(e => console.error('❌ [Email Error] Welcome Guardian:', e));
      }
    }

    if (!valid) {
      console.warn(`❌ [AUTH FAILED] Role: ${role}, Device: ${deviceId}. Check password.`);
      return NextResponse.json(
        { error: `Incorrect password for ${role} on device ${deviceId}` },
        { status: 401 }
      );
    }

    console.log(`✅ [AUTH SUCCESS] Role: ${role}, Device: ${deviceId}`);

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
