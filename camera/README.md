# Camera Module - Virtual Saree Mirror

## Overview

The camera module handles real-time webcam capture and frame processing for the Virtual Saree Mirror system. It provides:

- **Multi-camera support** - Handle multiple cameras with automatic fallback
- **Real-time capture** - Background thread captures frames at target FPS
- **Frame buffering** - Thread-safe queue prevents frame drops
- **FPS control** - Maintains consistent frame rate
- **Preprocessing** - Flip, color conversion, and image optimization
- **Performance monitoring** - Track actual FPS and buffer stats

## Architecture

```
┌─────────────────┐
│  CameraDevice   │ (Low-level camera access)
│  - OpenCV wrap  │
│  - Properties   │
└────────┬────────┘
         │
┌────────▼────────┐
│ CameraManager   │ (High-level interface)
│ - Thread mgmt   │
│ - FPS control   │
└────────┬────────┘
         │
┌────────▼────────┐
│  FrameBuffer    │ (Thread-safe queue)
│ - Circular buf   │
│ - Drop tracking  │
└────────┬────────┘
         │
      Frame
       Flow
         │
     Consumer
```

## Components

### CameraDevice

Low-level camera abstraction using OpenCV's VideoCapture.

**Features:**
- Support for different backends (DirectShow, V4L2, MSMF)
- Property control (brightness, contrast, exposure)
- Handles hardware limitations
- Error recovery

**Example:**
```python
from camera.device import CameraDevice

device = CameraDevice(device_id=0, width=1920, height=1080, fps=30)
if device.is_opened:
    ret, frame = device.read()
    device.set_brightness(100)
device.release()
```

### FrameBuffer

Thread-safe circular buffer for frames.

**Features:**
- Non-blocking operations
- Producer-consumer pattern
- Automatic frame dropping on overflow
- Statistics tracking

**Example:**
```python
from camera.frame_buffer import FrameBuffer

buffer = FrameBuffer(max_size=10)

# Producer (camera thread)
buffer.put(frame)

# Consumer (AI thread)
ret, frame = buffer.get(timeout=0.1)

# Get stats
stats = buffer.get_stats()
print(stats)
```

### CameraManager

High-level camera interface with background capture thread.

**Features:**
- Automatic device initialization
- Background capture thread
- FPS control and monitoring
- Frame preprocessing
- Statistics and diagnostics

**Example:**
```python
from camera.manager import CameraManager
import time

# Initialize
manager = CameraManager(
    device_id=0,
    width=1920,
    height=1080,
    fps=30,
    buffer_size=5,
    flip_h=True  # Mirror effect
)

if manager.is_ready:
    # Start capture
    manager.start()
    
    # Let buffer fill
    time.sleep(0.5)
    
    # Get frames
    for i in range(100):
        ret, frame = manager.get_frame(timeout=0.1)
        if ret:
            print(f"Frame shape: {frame.shape}")
            # Process frame...
    
    # Check stats
    print(manager.get_camera_info())
    print(manager.get_buffer_stats())
    
    # Stop
    manager.stop()
```

## Usage in Virtual Saree Mirror

### Integration with Main Application

```python
from camera import CameraManager
from utils.config import ConfigManager

config = ConfigManager()

# Get settings from config/default.yaml
camera_config = config.get("camera")

camera_manager = CameraManager(
    device_id=camera_config["device_id"],
    width=camera_config["width"],
    height=camera_config["height"],
    fps=camera_config["fps"],
    flip_h=camera_config["flip_horizontal"]
)

camera_manager.start()
```

### With Pose Detection (Phase 3 Integration)

```python
from camera import CameraManager
from ai.pose import PoseDetector  # Phase 3

camera = CameraManager()
camera.start()

pose_detector = PoseDetector()

while True:
    ret, frame = camera.get_frame(timeout=0.1)
    if ret:
        landmarks = pose_detector.detect(frame)
        # Use landmarks for draping...
```

## Performance Considerations

### FPS Control

The camera manager maintains target FPS through:

1. **Frame interval calculation**: `interval = 1 / target_fps`
2. **Sleep-based sync**: If frame comes faster than interval, sleep the difference
3. **FPS monitoring**: Actual FPS tracked over 30-frame window

### Buffer Management

- **Size**: Default 5 frames (buffer_size parameter)
- **Overflow handling**: Old frames dropped automatically
- **Statistics**: Track dropped frames and utilization

### Thread Safety

- All buffer operations protected by locks
- Frame copies used to prevent corruption
- Condition variables for efficient waiting

## Configuration

Camera settings in `config/default.yaml`:

```yaml
camera:
  device_id: 0           # Camera index
  width: 1920            # Frame width
  height: 1080           # Frame height
  fps: 30                # Target FPS
  flip_horizontal: true  # Mirror effect
  flip_vertical: false
  brightness: 1.0        # Brightness multiplier
  contrast: 1.0          # Contrast multiplier
```

## Troubleshooting

### Camera Not Found

```python
# List available cameras
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```

### Low FPS

**Causes:**
- Resolution too high
- CPU bottleneck
- Insufficient buffer size

**Solutions:**
```python
# Reduce resolution
manager = CameraManager(width=1280, height=720)

# Reduce buffer size
manager = CameraManager(buffer_size=3)

# Check actual FPS
print(f"Actual FPS: {manager.actual_fps}")
```

### Frame Drops

**Check statistics:**
```python
stats = manager.get_buffer_stats()
print(f"Dropped: {stats['dropped_frames']}")
print(f"Utilization: {stats['utilization']}")
```

**Solutions:**
- Increase buffer size
- Reduce camera resolution
- Optimize downstream processing

## Testing

### Unit Tests

```bash
pytest tests/test_camera.py -v
```

### Manual Testing

```python
from camera import CameraManager
import time

manager = CameraManager()
manager.start()

for i in range(100):
    ret, frame = manager.get_frame(timeout=0.1)
    if ret:
        print(f"Frame {i}: {frame.shape}")
    time.sleep(0.01)

print(manager.get_camera_info())
manager.stop()
```

## Phase 2 Completion Status

✅ **CameraDevice** - Low-level camera access with property control
✅ **FrameBuffer** - Thread-safe circular buffer with statistics
✅ **CameraManager** - High-level interface with background capture
✅ **FPS Control** - Maintains target frame rate
✅ **Multi-camera support** - Device selection and fallback
✅ **Preprocessing** - Frame flipping and color conversion
✅ **Performance monitoring** - FPS and buffer statistics
✅ **Documentation** - Complete module documentation

## Next Phase

**Phase 3: Pose Detection**
- MediaPipe Pose integration
- 33-point body landmark detection
- Jitter smoothing with Kalman filter
- Real-time inference

---

**Last Updated**: 2026-06-29
**Status**: Phase 2 Complete ✅
