#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Client for Smart Wheelchair - sends sensor data to Next.js API
"""

import time
import json
import requests
from typing import Dict, Any, Optional

class WheelchairAPIClient:
    """Client for sending sensor data to the Next.js API"""
    
    def __init__(self, api_url: str, device_id: str):
        """
        Initialize API client
        
        Args:
            api_url: Base URL of the API (e.g., https://your-app.vercel.app)
            device_id: Unique identifier for this wheelchair device
        """
        self.api_url = api_url.rstrip('/')
        self.device_id = device_id
        self.upload_endpoint = f"{self.api_url}/api/sensors/upload"
        self.status_endpoint = f"{self.api_url}/api/status"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        self.last_upload_time = 0
        self.upload_interval = 1.0  # Upload every 1 second
        
    def check_status(self) -> Dict[str, Any]:
        """
        Check if the API killswitch is enabled
        
        Returns:
            dict with killswitch status
        """
        try:
            response = self.session.get(self.status_endpoint, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"killswitch": True, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"Error checking API status: {e}")
            return {"killswitch": True, "error": str(e)}
    
    def upload_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """
        Upload sensor data to the API
        
        Args:
            sensor_data: Dictionary containing sensor readings
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_upload_time < self.upload_interval:
            return True  # Skip this upload
        
        # Build payload
        payload = {
            "deviceId": self.device_id,
            "timestamp": int(current_time * 1000),  # milliseconds
        }
        
        # Add sensor data if available
        if sensor_data.get('dht11'):
            payload['dht11'] = sensor_data['dht11']
            
        if sensor_data.get('ultrasonic'):
            payload['ultrasonic'] = sensor_data['ultrasonic']
            
        if sensor_data.get('max30100'):
            payload['max30100'] = sensor_data['max30100']
            
        if sensor_data.get('motorStatus'):
            payload['motorStatus'] = sensor_data['motorStatus']
            
        if 'obstacle' in sensor_data:
            payload['obstacle'] = sensor_data['obstacle']
        
        try:
            response = self.session.post(
                self.upload_endpoint,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                self.last_upload_time = current_time
                return True
            elif response.status_code == 503:
                # Killswitch enabled
                data = response.json()
                if data.get('killswitch'):
                    print("⚠️  API Killswitch is ENABLED - uploads are blocked")
                return False

            else:
                print(f"⚠️  Upload failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("⚠️  Upload timeout - API may be unreachable")
            return False
        except requests.exceptions.ConnectionError:
            print("⚠️  Connection error - check network and API URL")
            return False
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False


def get_sensor_snapshot(motors, ultra, dht, max30, state, state_lock):
    """
    Collect current sensor data from all sensors
    
    Args:
        motors: MotorDriver instance
        ultra: Ultra3 ultrasonic sensor instance
        dht: DHT11Reader instance
        max30: MAX30100 instance
        state: BotState instance
        state_lock: Threading lock for state
        
    Returns:
        dict: Sensor data snapshot
    """
    data = {}
    
    # DHT11 Temperature & Humidity
    if dht.temp_c is not None and dht.hum is not None:
        data['dht11'] = {
            'temperature': round(dht.temp_c, 2),
            'humidity': round(dht.hum, 2)
        }
    
    # Ultrasonic sensors
    front = ultra.distances.get("front")
    left = ultra.distances.get("left")
    right = ultra.distances.get("right")
    
    if front is not None and left is not None and right is not None:
        data['ultrasonic'] = {
            'front': round(front, 2),
            'left': round(left, 2),
            'right': round(right, 2)
        }
    
    # MAX30100
    if max30 and max30.present and max30.ir is not None and max30.red is not None:
        data['max30100'] = {
            'ir': int(max30.ir),
            'red': int(max30.red)
        }
    
    # Motor status
    with state_lock:
        data['motorStatus'] = {
            'lastCommand': motors.last(),
            'mode': state.mode
        }
        data['obstacle'] = state.obstacle_hit
    
    return data
