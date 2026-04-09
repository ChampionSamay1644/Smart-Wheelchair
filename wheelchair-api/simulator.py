#!/usr/bin/env python3
"""
Smart Wheelchair API Simulator

This script simulates a Raspberry Pi wheelchair sending sensor data to the Next.js API.
Use this to test that your API is working correctly before deploying to actual hardware.

Usage:
    python simulator.py --url https://your-app.vercel.app --api-key your_secret_key
"""

import time
import random
import argparse
import sys
from typing import Dict, Any
import requests

class WheelchairSimulator:
    """Simulates sensor data from a smart wheelchair"""
    
    def __init__(self, api_url: str, device_id: str = "simulator-001", password: str = "pat_123", no_motor: bool = False):
        """
        Initialize the simulator
        
        Args:
            api_url: Base URL of the Next.js API
            device_id: Unique device identifier
            password: Device password (used for both patient and guardian login)
        """
        self.api_url = api_url.rstrip('/')
        self.device_id = device_id
        self.password = password
        self.no_motor = no_motor
        self.sim_gps = False # Will be set by main
        self.upload_endpoint = f"{self.api_url}/api/sensors/upload"
        self.status_endpoint = f"{self.api_url}/api/status"
        
        # Initial GPS position (Mumbai - Gateway of India)
        self.lat = 18.92183
        self.lng = 72.83470
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-Device-Id': self.device_id,
            'X-Device-Password': self.password
        })
        
        # Simulation state
        self.temperature = 25.0
        self.humidity = 50.0
        self.distance_front = 100.0
        self.distance_left = 100.0
        self.distance_right = 100.0
        self.ir_value = 5000
        self.red_value = 4500
        self.motor_commands = ['F', 'B', 'L', 'R', 'S']
        self.current_command = 'S'
        self.mode = 'MANUAL'
        self.obstacle = None
        
    def check_status(self) -> Dict[str, Any]:
        """Check API status and killswitch"""
        try:
            response = self.session.get(self.status_endpoint, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def generate_sensor_data(self) -> Dict[str, Any]:
        """
        Generate realistic simulated sensor data
        
        Returns:
            dict: Simulated sensor readings
        """
        # Simulate temperature drift
        self.temperature += random.uniform(-0.5, 0.5)
        self.temperature = max(15.0, min(35.0, self.temperature))
        
        # Simulate humidity changes
        self.humidity += random.uniform(-2.0, 2.0)
        self.humidity = max(20.0, min(80.0, self.humidity))
        
        # Simulate obstacle detection
        self.distance_front += random.uniform(-5.0, 5.0)
        self.distance_front = max(10.0, min(150.0, self.distance_front))
        
        self.distance_left += random.uniform(-3.0, 3.0)
        self.distance_left = max(10.0, min(150.0, self.distance_left))
        
        self.distance_right += random.uniform(-3.0, 3.0)
        self.distance_right = max(10.0, min(150.0, self.distance_right))
        
        # Detect obstacles
        if self.distance_front < 30.0:
            self.obstacle = "front"
        elif self.distance_left < 30.0:
            self.obstacle = "left"
        elif self.distance_right < 30.0:
            self.obstacle = "right"
        else:
            self.obstacle = None
        
        # Simulate heart rate sensor values (Realistic human range)
        # Randomly spike values to test API thresholds (Pulse > 120, SpO2 < 90)
        chance = random.random()
        if chance < 0.05: # 5% chance of high pulse
            self.ir_value = random.randint(12100, 13500) # Results in 121-135 BPM
            self.red_value = random.randint(9200, 9500)
        elif chance < 0.10: # 5% chance of low SpO2
            self.ir_value = random.randint(7000, 8500)
            self.red_value = random.randint(8500, 8900) # Results in < 90%
        else:
            self.ir_value = max(6000, min(8500, self.ir_value + random.randint(-150, 150)))
            self.red_value = max(9400, min(9900, self.red_value + random.randint(-50, 50)))

        # Simulate small GPS drift if enabled
        if self.sim_gps:
            self.lat += random.uniform(-0.0001, 0.0001)
            self.lng += random.uniform(-0.0001, 0.0001)
        
        # Occasionally change motor command
        if random.random() < 0.2:  # 20% chance to change movement
            self.current_command = random.choice(self.motor_commands)
            # Assign a realistic mode based on the command
            self.mode = random.choice(['JOYSTICK', 'VOICE', 'MANUAL'])
        elif random.random() < 0.4: # 40% chance to keep moving
            pass # Keep current_command
        else:
            self.current_command = 'S' # 40% chance to stop
        
        # Build sensor data payload
        payload = {
            "deviceId": self.device_id,
            "timestamp": int(time.time() * 1000),
            "dht11": {
                "temperature": round(self.temperature, 1),
                "humidity": round(self.humidity, 1)
            },
            "ultrasonic": {
                "front": round(self.distance_front, 1),
                "left": round(self.distance_left, 1),
                "right": round(self.distance_right, 1)
            },
            "max30100": {
                "ir": int(self.ir_value),
                "red": int(self.red_value)
            },
            "obstacle": self.obstacle
        }

        # Include navigation if GPS simulation is enabled
        if self.sim_gps:
            payload["navigation"] = {
                "currentPosition": {
                    "lat": self.lat,
                    "lng": self.lng
                },
                "timestamp": int(time.time() * 1000)
            }
        
        if not self.no_motor:
            payload["motorStatus"] = {
                "lastCommand": self.current_command,
                "mode": self.mode
            }
        
        return payload
    
    def upload_data(self, data: Dict[str, Any]) -> bool:
        """
        Upload sensor data to the API
        
        Args:
            data: Sensor data to upload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self.session.post(
                self.upload_endpoint,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 503:
                result = response.json()
                if result.get('killswitch'):
                    print("⚠️  KILLSWITCH ENABLED - Uploads blocked by API")
                return False
            else:
                print(f"⚠️  Upload failed: HTTP {response.status_code}")
                print(f"    Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("⚠️  Request timeout")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def run(self, interval: float = 1.0, duration: int = None):
        """
        Run the simulator
        
        Args:
            interval: Time between uploads in seconds
            duration: Total duration to run (None = infinite)
        """
        print("=" * 60)
        print("🚀 Smart Wheelchair API Simulator")
        print("=" * 60)
        print(f"API URL:    {self.api_url}")
        print(f"Device ID:  {self.device_id}")
        print(f"Interval:   {interval}s")
        print("-" * 60)
        
        # Check initial status
        print("\n🔍 Checking API status...")
        status = self.check_status()
        
        if 'error' in status:
            print(f"❌ Error: {status['error']}")
            print("\n⚠️  Make sure:")
            print("   1. Your API is deployed and running")
            print("   2. The URL is correct")
            print("   3. You have internet connection")
            return
        
        print(f"✅ API Status: {status.get('status', 'unknown').upper()}")
        
        if status.get('killswitch'):
            print(f"⚠️  {status.get('message', 'Killswitch is enabled')}")
            print("\n   To enable uploads, set KILLSWITCH_ENABLED=false in your .env file")
        
        print("\n" + "=" * 60)
        print("Starting simulation... Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        upload_count = 0
        success_count = 0
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Generate and upload data
                data = self.generate_sensor_data()
                success = self.upload_data(data)
                
                upload_count += 1
                if success:
                    success_count += 1
                
                # Print status
                status_icon = "✅" if success else "❌"
                motor_val = data['motorStatus']['lastCommand'] if 'motorStatus' in data else "Disabled"
                
                pulse = data['max30100']['ir'] / 100 if 'max30100' in data else 0
                spo2 = data['max30100']['red'] / 100 if 'max30100' in data else 0
                gps_str = f"{self.lat:.5f},{self.lng:.5f}" if self.sim_gps else "Off"
                
                print(f"{status_icon} Upload #{upload_count:04d} | "
                      f"GPS: {gps_str} | "
                      f"Pulse: {pulse:.0f}bpm | SpO2: {spo2:.0f}% | "
                      f"Temp: {data['dht11']['temperature']:.1f}°C | "
                      f"Motor: {motor_val}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("🛑 Simulation stopped by user")
        
        # Print summary
        elapsed = time.time() - start_time
        success_rate = (success_count / upload_count * 100) if upload_count > 0 else 0
        
        print("=" * 60)
        print("📊 Simulation Summary")
        print("=" * 60)
        print(f"Duration:        {elapsed:.1f}s")
        print(f"Total uploads:   {upload_count}")
        print(f"Successful:      {success_count}")
        print(f"Failed:          {upload_count - success_count}")
        print(f"Success rate:    {success_rate:.1f}%")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Simulate wheelchair sensor data uploads to API'
    )
    parser.add_argument(
        '--url',
        default='https://wheelchair-api.vercel.app',
        help='API base URL (default: https://wheelchair-api.vercel.app)'
    )
    parser.add_argument(
        '--device-id',
        default='TEST_001',
        help='Device identifier (default: TEST_001)'
    )
    parser.add_argument(
        '--password',
        default='pat_123',
        help='Device password for both patient and guardian login (default: pat_123)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Upload interval in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Total duration in seconds (default: infinite)'
    )
    parser.add_argument(
        '--no-motor',
        action='store_true',
        help='Disable motor status simulation (to avoid conflicts with app)'
    )
    parser.add_argument(
        '--gps',
        action='store_true',
        help='Enable GPS simulation (useful for testing without hardware)'
    )
    
    args = parser.parse_args()
    
    # Create and run simulator
    simulator = WheelchairSimulator(
        api_url=args.url,
        device_id=args.device_id,
        password=args.password,
        no_motor=args.no_motor
    )
    simulator.sim_gps = args.gps
    
    simulator.run(interval=args.interval, duration=args.duration)


if __name__ == '__main__':
    main()
