'use client';

import { useEffect, useState } from 'react';
import { database } from '@/lib/firebase-client';
import { ref, onValue, off, DataSnapshot } from 'firebase/database';

interface SensorData {
  deviceId: string;
  timestamp: number;
  serverTimestamp?: number;
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

interface Device {
  deviceId: string;
  lastUpdate: number | null;
  online: boolean;
}

export default function Dashboard() {
  const [devices, setDevices] = useState<Device[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [sensorData, setSensorData] = useState<SensorData | null>(null);
  const [sessions, setSessions] = useState<{ [key: string]: any }>({});
  const [systemStatus, setSystemStatus] = useState<{
    killswitch: boolean;
    status: string;
    message: string;
  } | null>(null);

  // Fetch system status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        setSystemStatus(data);
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Check every 10s
    return () => clearInterval(interval);
  }, []);

  // Fetch sessions
  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        setSessions(data.sessions || {});
      } catch (error) {
        console.error('Error fetching sessions:', error);
      }
    };

    fetchSessions();
    const interval = setInterval(fetchSessions, 5000); // Check every 5s
    return () => clearInterval(interval);
  }, []);

  // Fetch devices list
  useEffect(() => {
    const fetchDevices = async () => {
      try {
        const res = await fetch('/api/devices');
        const data = await res.json();
        setDevices(data.devices || []);

        // Auto-select first device
        if (data.devices.length > 0 && !selectedDevice) {
          setSelectedDevice(data.devices[0].deviceId);
        }
      } catch (error) {
        console.error('Error fetching devices:', error);
      }
    };

    fetchDevices();
    const interval = setInterval(fetchDevices, 5000);
    return () => clearInterval(interval);
  }, [selectedDevice]);

  // Listen to real-time sensor data
  useEffect(() => {
    if (!selectedDevice) return;

    const deviceRef = ref(database, `devices/${selectedDevice}/current`);

    const unsubscribe = onValue(deviceRef, (snapshot: DataSnapshot) => {
      const data = snapshot.val();
      if (data) {
        setSensorData(data);
      }
    });

    return () => off(deviceRef, 'value', unsubscribe);
  }, [selectedDevice]);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const getStatusColor = (online: boolean) => {
    return online ? 'bg-green-500' : 'bg-red-500';
  };

  const getRoleIcon = (role: string) => {
    switch (role.toLowerCase()) {
      case 'patient': return '♿';
      case 'guardian': return '👤';
      case 'doctor': return '👨‍⚕️';
      default: return '📱';
    }
  };

  const getObstacleColor = (obstacle: string | null | undefined) => {
    return obstacle ? 'text-red-400' : 'text-green-400';
  };

  return (
    <div className="min-h-screen p-6 bg-linear-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Smart Wheelchair Dashboard
          </h1>
          <p className="text-slate-400">Real-time sensor monitoring and control</p>
        </div>

        {/* System Status */}
        {systemStatus && (
          <div className={`mb-6 p-4 rounded-xl ${systemStatus.killswitch
            ? 'bg-red-900/20 border border-red-500/50'
            : 'bg-green-900/20 border border-green-500/50'
            }`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white">System Status</h3>
                <p className={systemStatus.killswitch ? 'text-red-400' : 'text-green-400'}>
                  {systemStatus.message}
                </p>
              </div>
              <div className={`px-4 py-2 rounded-full ${systemStatus.killswitch ? 'bg-red-500' : 'bg-green-500'
                } text-white font-bold`}>
                {systemStatus.status.toUpperCase()}
              </div>
            </div>
          </div>
        )}

        {/* Device Selection & Sessions */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-xs rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-semibold text-white mb-4">Devices</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {devices.map((device: Device) => (
                <button
                  key={device.deviceId}
                  onClick={() => setSelectedDevice(device.deviceId)}
                  className={`p-4 rounded-lg border-2 transition-all text-left ${selectedDevice === device.deviceId
                    ? 'border-blue-500 bg-blue-500/20'
                    : 'border-slate-600 bg-slate-700/50 hover:border-slate-500'
                    }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-white">{device.deviceId}</span>
                    <span className={`w-3 h-3 rounded-full ${getStatusColor(device.online)}`} />
                  </div>
                  <p className="text-sm text-slate-400">
                    {device.lastUpdate
                      ? `Updated: ${new Date(device.lastUpdate).toLocaleTimeString()}`
                      : 'No data'}
                  </p>
                </button>
              ))}
              {devices.length === 0 && (
                <p className="text-slate-400 text-center py-4">No devices connected yet</p>
              )}
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-semibold text-white mb-4">Active Sessions</h2>
            <div className="space-y-3">
              {Object.entries(sessions).map(([devId, deviceSessions]: [string, any]) => {
                // deviceSessions is now an object of roles if we changed the backend
                // or it might still be a single session if it's old data
                if (deviceSessions.role) {
                  // Old flat structure
                  return (
                    <div key={devId} className="flex items-center p-3 rounded-lg bg-slate-700/30 border border-slate-600">
                      <div className="mr-3 text-2xl">{getRoleIcon(deviceSessions.role)}</div>
                      <div className="flex-1">
                        <div className="text-white font-medium capitalize">{deviceSessions.role}</div>
                        <div className="text-xs text-slate-400">{devId}</div>
                      </div>
                      <div className="text-green-400 text-xs font-bold animate-pulse">LIVE</div>
                    </div>
                  );
                }

                // New nested structure: { patient: {...}, guardian: {...} }
                return Object.entries(deviceSessions).map(([role, session]: [string, any]) => (
                  <div key={`${devId}-${role}`} className="flex items-center p-3 rounded-lg bg-slate-700/30 border border-slate-600">
                    <div className="mr-3 text-2xl">{getRoleIcon(session.role)}</div>
                    <div className="flex-1">
                      <div className="text-white font-medium capitalize">{session.role}</div>
                      <div className="text-xs text-slate-400">{devId}</div>
                    </div>
                    <div className="text-green-400 text-xs font-bold animate-pulse">LIVE</div>
                  </div>
                ));
              })}
              {Object.keys(sessions).length === 0 && (
                <p className="text-slate-400 text-center py-4 text-sm">No active app logins</p>
              )}
            </div>
          </div>
        </div>

        {/* Sensor Data */}
        {sensorData && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* DHT11 Temperature & Humidity */}
            {sensorData.dht11 && (
              <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-6 border border-slate-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <span className="mr-2">🌡️</span>
                  Temperature & Humidity
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Temperature:</span>
                    <span className="text-2xl font-bold text-orange-400">
                      {sensorData.dht11.temperature.toFixed(1)}°C
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Humidity:</span>
                    <span className="text-2xl font-bold text-blue-400">
                      {sensorData.dht11.humidity.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Ultrasonic Sensors */}
            {sensorData.ultrasonic && (
              <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-6 border border-slate-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <span className="mr-2">📡</span>
                  Ultrasonic Sensors
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Front:</span>
                    <span className="text-xl font-bold text-cyan-400">
                      {sensorData.ultrasonic.front.toFixed(1)} cm
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Left:</span>
                    <span className="text-xl font-bold text-cyan-400">
                      {sensorData.ultrasonic.left.toFixed(1)} cm
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Right:</span>
                    <span className="text-xl font-bold text-cyan-400">
                      {sensorData.ultrasonic.right.toFixed(1)} cm
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* MAX30100 Heart Rate Sensor */}
            {sensorData.max30100 && (
              <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-6 border border-slate-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <span className="mr-2">❤️</span>
                  MAX30100 Sensor
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">IR Value:</span>
                    <span className="text-xl font-bold text-red-400">
                      {Math.round(sensorData.max30100.ir)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">RED Value:</span>
                    <span className="text-xl font-bold text-red-400">
                      {Math.round(sensorData.max30100.red)}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">
                    Raw sensor values
                  </p>
                </div>
              </div>
            )}

            {/* Motor Status */}
            {sensorData.motorStatus && (
              <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-6 border border-slate-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <span className="mr-2">⚙️</span>
                  Motor Status
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Mode:</span>
                    <span className="text-xl font-bold text-purple-400">
                      {sensorData.motorStatus.mode}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Last Command:</span>
                    <span className="text-xl font-bold text-purple-400">
                      {sensorData.motorStatus.lastCommand}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Obstacle Detection */}
            <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-6 border border-slate-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <span className="mr-2">⚠️</span>
                Obstacle Detection
              </h3>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Status:</span>
                <span className={`text-2xl font-bold ${getObstacleColor(sensorData.obstacle)}`}>
                  {sensorData.obstacle || 'Clear'}
                </span>
              </div>
            </div>

            {/* Timestamps */}
            <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-6 border border-slate-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <span className="mr-2">🕐</span>
                Timestamps
              </h3>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="text-slate-400">Device Time:</span>
                  <p className="text-white font-mono">
                    {formatTimestamp(sensorData.timestamp)}
                  </p>
                </div>
                {sensorData.serverTimestamp && (
                  <div>
                    <span className="text-slate-400">Server Time:</span>
                    <p className="text-white font-mono">
                      {formatTimestamp(sensorData.serverTimestamp)}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {!sensorData && selectedDevice && (
          <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-12 border border-slate-700 text-center">
            <p className="text-slate-400 text-lg">Waiting for data from {selectedDevice}...</p>
          </div>
        )}

        {!selectedDevice && devices.length === 0 && (
          <div className="bg-slate-800/50 backdrop-blur-xs rounded-xl p-12 border border-slate-700 text-center">
            <p className="text-slate-400 text-lg">No devices connected. Start your hardware to see data.</p>
          </div>
        )}
      </div>
    </div>
  );
}
