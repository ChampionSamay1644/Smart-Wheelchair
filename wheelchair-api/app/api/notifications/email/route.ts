import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { type, targetEmail, contextEmail, userName, patientName } = await request.json();

    if (!targetEmail) {
      return NextResponse.json({ error: 'Target email is required' }, { status: 400 });
    }

    let subject = '';
    let htmlBody = '';

    switch (type) {
      case 'welcome_patient':
        subject = '🦽 Welcome to SmartNav Wheelchair';
        htmlBody = `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #f9f9f9; border-radius: 10px;">
            <h2 style="color: #4F46E5;">Welcome, ${userName || 'Patient'}! 👋</h2>
            <p>Your Smart Wheelchair system is now <strong>active and connected</strong>.</p>
            <ul>
              <li>📍 Location tracking is enabled</li>
              <li>❤️ Health vitals are being monitored</li>
              <li>🔔 Emergency alerts are armed and ready</li>
            </ul>
            <p style="color: #888; font-size: 12px;">SmartNav Wheelchair System</p>
          </div>`;
        break;

      case 'guardian_alert':
        subject = '🔔 Guardian Connected to Your Wheelchair';
        htmlBody = `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #fff8f0; border-radius: 10px;">
            <h2 style="color: #D97706;">Security Alert ⚠️</h2>
            <p>A Guardian has connected to your wheelchair system.</p>
            <table style="border-collapse: collapse; width: 100%;">
              <tr><td style="padding: 8px; font-weight: bold;">Guardian Name:</td><td style="padding: 8px;">${userName || 'Unknown'}</td></tr>
              <tr><td style="padding: 8px; font-weight: bold;">Guardian Email:</td><td style="padding: 8px;">${contextEmail || 'N/A'}</td></tr>
            </table>
            <p>If this was not authorized, please change your device password immediately.</p>
            <p style="color: #888; font-size: 12px;">SmartNav Wheelchair System</p>
          </div>`;
        break;

      case 'welcome_guardian':
        subject = '🦽 Guardian Access Confirmed - SmartNav';
        htmlBody = `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #f0fdf4; border-radius: 10px;">
            <h2 style="color: #16A34A;">You're now connected, ${userName || 'Guardian'}! ✅</h2>
            <p>You have successfully linked to your patient's wheelchair.</p>
            <table style="border-collapse: collapse; width: 100%;">
              <tr><td style="padding: 8px; font-weight: bold;">Patient Name:</td><td style="padding: 8px;">${patientName || 'Your Patient'}</td></tr>
              <tr><td style="padding: 8px; font-weight: bold;">Patient Email:</td><td style="padding: 8px;">${contextEmail || 'N/A'}</td></tr>
            </table>
            <p>You can now monitor their live vitals, location, and receive emergency alerts.</p>
            <p style="color: #888; font-size: 12px;">SmartNav Wheelchair System</p>
          </div>`;
        break;

      case 'health_alert':
        subject = '🚨 URGENT: Health Alert from Smart Wheelchair';
        htmlBody = `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #fff1f2; border-radius: 10px; border: 2px solid #e11d48;">
            <h2 style="color: #e11d48;">Health Emergency Detected! 🚨</h2>
            <p>The system has detected abnormal vitals for <strong>${patientName || 'the patient'}</strong>.</p>
            <div style="background: #ffffff; padding: 15px; border-radius: 8px; margin: 15px 0;">
              <p style="margin: 5px 0;"><strong>Vital Sign:</strong> ${contextEmail || 'N/A'}</p>
              <p style="margin: 5px 0;"><strong>Status:</strong> Critical Threshold Exceeded</p>
            </div>
            <p>Please check the SmartNav Guardian app immediately or contact the patient.</p>
            <p style="color: #888; font-size: 12px;">SmartNav Automatic Emergency Alert</p>
          </div>`;
        break;

      default:
        return NextResponse.json({ error: 'Invalid notification type' }, { status: 400 });
    }

    const apiKey = process.env.RESEND_API_KEY;

    if (!apiKey) {
      // Demo mode — just log
      console.log('================================================');
      console.log('📨 [DEMO MODE - No RESEND_API_KEY set]');
      console.log(`📍 To: ${targetEmail}`);
      console.log(`📝 Subject: ${subject}`);
      console.log('================================================');
      return NextResponse.json({ success: true, serverStatus: 'success', mode: 'demo_logging' });
    }

    // LIVE MODE — actually send via Resend
    const res = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        from: process.env.RESEND_FROM_EMAIL || 'SmartNav Wheelchair <alerts@smartnav.tech>',
        to: [targetEmail],
        subject,
        html: htmlBody,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      console.error('❌ [Resend API Error]:', JSON.stringify(data, null, 2));
      return NextResponse.json({ 
        error: 'Email sending failed', 
        details: data,
        apiKeyPrefix: `${apiKey.substring(0, 5)}...`
      }, { status: 500 });
    }

    console.log(`✅ [Email Success] ID: ${data.id} → To: ${targetEmail} (Type: ${type})`);
    return NextResponse.json({
      success: true,
      serverStatus: 'success',
      mode: 'live',
      emailId: data.id,
    });

  } catch (error: any) {
    console.error('Email Notification Error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
