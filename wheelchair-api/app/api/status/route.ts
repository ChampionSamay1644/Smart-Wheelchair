import { NextRequest, NextResponse } from 'next/server';
import { KILLSWITCH_ENABLED } from '@/lib/config';

export async function GET(request: NextRequest) {
  return NextResponse.json({
    killswitch: KILLSWITCH_ENABLED,
    status: KILLSWITCH_ENABLED ? 'disabled' : 'active',
    message: KILLSWITCH_ENABLED 
      ? 'All hardware communication is currently disabled' 
      : 'System is operational',
  });
}
