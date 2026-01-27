#!/usr/bin/env python3
"""
LOCAL Configuration file for Smart Wheelchair API Integration

⚠️ IMPORTANT: This file should contain your API secret key
⚠️ Add this filename to .gitignore - NEVER commit secrets to git!

Copy this file to config_local.py and update the values.
"""

# ====================== API CONFIGURATION ======================

# API Secret Key - MUST match the API_SECRET_KEY in Vercel environment variables
# Generate with: openssl rand -base64 32
# ⚠️ NEVER commit this file to git!
API_SECRET_KEY = "your_secure_random_string_here"

# ====================== USAGE INSTRUCTIONS ======================
"""
SETUP:
------
1. Copy this file: cp config_template.py config_local.py
2. Add to .gitignore: echo "config_local.py" >> .gitignore
3. Edit config_local.py with your actual API secret key
4. NEVER commit config_local.py to git!

CONFIGURATION IN code.py:
-------------------------
Update the USER CONFIG section in code.py:
    API_URL = "https://your-app.vercel.app"  # Your Vercel URL (safe to commit)
    DEVICE_ID = "wheelchair-rpi-001"         # Device ID (safe to commit)
    API_UPLOAD_ENABLED = True                # Enable/disable (safe to commit)

The API_SECRET_KEY will be loaded automatically from:
    1. Environment variable: WHEELCHAIR_API_KEY (production - recommended)
    2. config_local.py file (development - gitignored)

PRODUCTION DEPLOYMENT (Recommended):
-------------------------------------
Use environment variables instead of config_local.py:

    export WHEELCHAIR_API_KEY="your_secure_random_string_here"
    python3 code.py

Or add to ~/.bashrc or ~/.bash_profile:
    export WHEELCHAIR_API_KEY="your_secure_random_string_here"

TESTING:
--------
    # Set API key for current session
    export WHEELCHAIR_API_KEY="your_key_from_vercel"
    
    # Run the code
    python3 code.py

SECURITY:
---------
✅ DO: Store secrets in environment variables or gitignored config files
✅ DO: Use different keys for development and production
❌ DON'T: Hardcode secrets in code.py
❌ DON'T: Commit config_local.py to git
❌ DON'T: Share your API secret key publicly
"""
