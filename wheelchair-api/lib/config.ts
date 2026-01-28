// Killswitch configuration
export const KILLSWITCH_ENABLED = process.env.KILLSWITCH_ENABLED === 'true';

// Sensor data validation
export interface SensorData {
  deviceId: string;
  timestamp: number;
  dht11?: {
    temperature: number;
    humidity: number;
  };
  ultrasonic?: {
    front: number;
    left: number;
    right: number;
  };
  max30100?: {
    ir: number;
    red: number;
  };
  motorStatus?: {
    lastCommand: string;
    mode: string;
  };
  obstacle?: string | null;
}

export function validateSensorData(data: any): data is SensorData {
  return (
    typeof data === 'object' &&
    typeof data.deviceId === 'string' &&
    typeof data.timestamp === 'number'
  );
}
