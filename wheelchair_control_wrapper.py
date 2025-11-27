#!/usr/bin/env python3
"""
Wrapper for wheelchair_control.py to support WebSocket server processing
Adds --process flag for subprocess audio processing with JSON output
"""

import sys
import json
import argparse
from pathlib import Path

# Add the script directory to the path for imports
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

def process_audio_file(audio_path, simulate_only=False):
    """
    Process audio file and return JSON result
    
    This is a simplified version that will be integrated into wheelchair_control.py
    For now, it returns a simulated response
    """
    try:
        from wheelchair_control import (
            process_voice_command,
            preprocess_and_noise_reduce
        )
        
        # Preprocess audio
        processed_path = preprocess_and_noise_reduce(audio_path)
        
        # Process voice command
        result = process_voice_command(processed_path)
        
        # Format result as JSON
        if result:
            return {
                "success": True,
                "command": result.get("command", "unknown"),
                "language": result.get("language", "en"),
                "confidence": result.get("confidence", 0.0),
                "mode": "command",  # command mode (later: add "ai" mode for Groq API)
                "action_taken": "simulated" if simulate_only else "executed",
                "translation": result.get("translation", None)  # Include translation if available
            }
        else:
            return {
                "success": False,
                "command": None,
                "language": None,
                "confidence": 0.0,
                "mode": "command",
                "action_taken": "no_command_detected",
                "error": "No valid command detected"
            }
            
    except ImportError as e:
        # Fallback: Return error when wheelchair_control can't be imported
        return {
            "success": False,
            "command": None,
            "language": None,
            "confidence": 0.0,
            "mode": "command",
            "action_taken": "import_error",
            "error": f"Failed to import wheelchair_control: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "command": None,
            "language": None,
            "confidence": 0.0,
            "mode": "command",
            "action_taken": "exception",
            "error": str(e)
        }

def enroll_voice(audio_path, user_name):
    """Enroll a user's voice for authentication"""
    try:
        from wheelchair_control import enroll_user_voice
        
        success = enroll_user_voice(audio_path, user_name)
        
        if success:
            return {
                "success": True,
                "user_name": user_name,
                "message": f"Voice enrolled successfully for {user_name}"
            }
        else:
            return {
                "success": False,
                "error": "Voice enrollment failed"
            }
            
    except ImportError as e:
        return {
            "success": False,
            "error": f"Failed to import wheelchair_control: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(
        description="Process audio file and return JSON result"
    )
    parser.add_argument(
        "--process",
        help="Path to audio file to process"
    )
    parser.add_argument(
        "--enroll-voice",
        help="Path to enrollment audio file"
    )
    parser.add_argument(
        "--user-name",
        help="User name for voice enrollment"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output result as JSON"
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Simulate commands without motor actuation"
    )
    
    args = parser.parse_args()
    
    if args.enroll_voice:
        # Voice enrollment mode
        if not args.user_name:
            print(json.dumps({"success": False, "error": "User name required for enrollment"}))
            sys.exit(1)
        result = enroll_voice(args.enroll_voice, args.user_name)
    elif args.process:
        # Process audio mode
        result = process_audio_file(args.process, args.simulate_only)
    else:
        print(json.dumps({"success": False, "error": "Either --process or --enroll-voice required"}))
        sys.exit(1)
    
    # Output JSON
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)

if __name__ == "__main__":
    main()
