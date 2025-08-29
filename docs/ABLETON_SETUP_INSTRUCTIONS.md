# Ableton Live Integration Setup Instructions

## Overview
This guide will help you set up the PolyMAX parameter prediction integration with Ableton Live using a custom remote script.

## Current Status
✅ TCP Server is running on `localhost:9001`  
✅ Core integration components are tested and working  
⏳ Waiting for Ableton Live remote script configuration  

## Step 1: Install the Remote Script

### Option A: Copy to User Remote Scripts (Recommended)
1. Open Finder and navigate to:
   ```
   ~/Library/Preferences/Ableton/Live x.x.x/User Remote Scripts/
   ```
   (Replace `x.x.x` with your Live version, e.g., `Live 12.0.0`)

2. Create a new folder called `PolyMAX_Remote`

3. Copy the file `ableton_polymax_remote.py` into this folder

4. Rename it to `__init__.py` (this is important!)

### Option B: Copy to System Remote Scripts
1. Navigate to:
   ```
   /Applications/Ableton Live x Suite.app/Contents/App-Resources/MIDI Remote Scripts/
   ```

2. Create a new folder called `PolyMAX_Remote`

3. Copy `ableton_polymax_remote.py` and rename to `__init__.py`

## Step 2: Configure Ableton Live

1. **Open Ableton Live**

2. **Go to Preferences**:
   - macOS: `Live` → `Preferences` (or `Cmd+,`)
   - Windows: `Options` → `Preferences`

3. **Navigate to Link/Tempo/MIDI**:
   - Click on the `Link/Tempo/MIDI` tab

4. **Configure Control Surface**:
   - In the `Control Surface` dropdown, select `PolyMAX_Remote`
   - Set `Input` to `None` (we're using TCP, not MIDI)
   - Set `Output` to `None`

5. **Apply Settings**:
   - Click `OK` to apply the settings
   - Live will load the remote script

## Step 3: Test the Connection

### What Should Happen:
1. When you configure the remote script in Live, it should automatically try to connect to `localhost:9001`
2. You should see a connection message in the terminal where the TCP server is running
3. The remote script will send a "ping" message to test the connection

### Expected Terminal Output:
```
✅ New connection from ('127.0.0.1', XXXXX)
📨 Received: {'action': 'ping'}
📤 Sent: {'status': 'pong', 'timestamp': 1234567890.123}
```

## Step 4: Load PolyMAX VST

1. **Create a new MIDI track** in Ableton Live

2. **Load PolyMAX**:
   - Add the UAD PolyMAX VST3 plugin to the track
   - The plugin should be located at: `/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3`

3. **Verify Detection**:
   - The remote script should automatically detect the PolyMAX plugin
   - You should see detection messages in the terminal

## Step 5: Test Parameter Control

### Manual Test:
The TCP server can send test parameters to Live. In the terminal, you can:
- Type `broadcast` to send test parameters
- Type `status` to check server status
- Type `quit` to stop the server

### Expected Behavior:
- Parameters sent from the server should automatically update the PolyMAX plugin
- You should hear the sound change as parameters are applied

## Troubleshooting

### Remote Script Not Appearing in Live:
- Check that the file is named `__init__.py` (not `ableton_polymax_remote.py`)
- Verify the folder structure is correct
- Restart Ableton Live completely
- Check Live's log file for Python errors

### Connection Issues:
- Ensure the TCP server is running (`python simple_tcp_server.py`)
- Check that port 9001 is not blocked by firewall
- Verify Live is trying to connect (check terminal output)

### PolyMAX Not Detected:
- Make sure PolyMAX is loaded on a track (not just in the browser)
- Check that the VST3 version is being used
- Verify the plugin name matches what the script expects

### No Parameter Changes:
- Check that the PolyMAX track is selected in Live
- Verify parameters are being sent (check terminal output)
- Ensure parameter names match between script and plugin

## Next Steps

Once the basic connection is working:

1. **Test with Real Audio**: Use the full prediction pipeline with actual audio files
2. **Train Better Models**: Continue training for improved parameter prediction accuracy
3. **Customize Parameters**: Adjust which PolyMAX parameters are controlled
4. **Add Features**: Implement real-time audio analysis and automatic parameter updates

## Files Overview

- `simple_tcp_server.py` - Test server (currently running)
- `ableton_polymax_remote.py` - Ableton Live remote script
- `predict_polymax_params.py` - Main inference script
- `ableton_integration.py` - Integration helper
- `polymax_web_api.py` - Web API for remote access

---

**Current Status**: TCP server is running and ready for connections. Please follow the setup instructions above to configure Ableton Live.