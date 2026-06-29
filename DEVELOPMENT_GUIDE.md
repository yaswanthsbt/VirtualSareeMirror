# Development Guide - Virtual Saree Mirror

## Getting Started for Developers

### Prerequisites

1. **Python 3.12+**
2. **Git** for version control
3. **NVIDIA CUDA 11.8+** (for GPU support)
4. **cuDNN 8.x** (paired with CUDA)
5. **Visual Studio Build Tools** (Windows) or GCC (Linux)

### Development Environment Setup

#### 1. Clone and Setup
```bash
git clone https://github.com/yaswanthsbt/VirtualSareeMirror.git
cd VirtualSareeMirror

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .  # Editable install
```

#### 2. NVIDIA GPU Setup (Windows)
```bash
# Install CUDA 11.8
# Install cuDNN
# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. NVIDIA GPU Setup (Linux)
```bash
# Ubuntu 22.04 example
sudo apt-get update
sudo apt-get install cuda-11-8 libcudnn8

# Add to PATH in ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

#### 4. Verify Setup
```bash
python -c "
import torch
import cv2
import mediapipe
print('PyTorch:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('OpenCV:', cv2.__version__)
print('MediaPipe:', mediapipe.__version__)
"
```

## Project Structure Reference

```
VirtualSareeMirror/
├── camera/                 # Phase 2: Camera capture
├── ai/                     # Phases 3-8: AI modules
│   ├── pose/              # Phase 3: Pose detection
│   ├── segmentation/      # Phase 4: Segmentation
│   ├── body_mesh/         # Phase 5: Body estimation
│   ├── tracking/          # Phase 7: Motion tracking
│   └── draping/           # Phase 8: Saree draping ⭐
├── saree/                 # Phases 5-6: Saree management
│   ├── loader/            # Phase 5: Saree loading
│   └── database/          # Phase 5: Database
├── renderer/              # Phase 8: Rendering
├── ui/                    # Phase 9: GUI
├── models/                # Pre-trained weights
├── assets/                # UI resources
├── config/                # YAML configs
├── utils/                 # Helper utilities
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Coding Standards

### 1. Type Hints (Mandatory)
```python
def process_frame(frame: np.ndarray, confidence: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
    """Process camera frame.
    
    Args:
        frame: Input image array (H, W, 3)
        confidence: Detection confidence threshold (0.0-1.0)
    
    Returns:
        Processed frame and detection results
    """
    pass
```

### 2. Logging (Mandatory)
```python
import logging

logger = logging.getLogger(__name__)

logger.info("Processing frame")
logger.debug(f"Detected {len(landmarks)} landmarks")
logger.warning("Low confidence detection")
logger.error(f"Failed to load model: {error}")
```

### 3. Error Handling
```python
try:
    result = model.inference(frame)
except RuntimeError as e:
    logger.error(f"GPU error: {e}")
    result = cpu_fallback(frame)
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

### 4. Configuration Usage
```python
from utils.config import ConfigManager

config = ConfigManager()
pose_threshold = config.get("pose.confidence_threshold", default=0.5)
fps = config.get("camera.fps", default=30)
```

### 5. Documentation
```python
class PoseDetector:
    """MediaPipe-based pose detection for body landmarks.
    
    This class wraps MediaPipe Pose to detect 33 body landmarks in real-time.
    It includes optional smoothing to reduce jitter.
    
    Attributes:
        confidence_threshold: Minimum confidence for landmark detection
        smoothing_enabled: Whether to apply temporal smoothing
        landmarks: Last detected landmarks
    
    Example:
        >>> detector = PoseDetector(confidence_threshold=0.5)
        >>> landmarks = detector.detect(frame)
        >>> print(landmarks[0].x, landmarks[0].y)
    """
    pass
```

## Phase Development Workflow

### Before Starting a Phase

1. Create a feature branch:
   ```bash
   git checkout -b phase-N-module-name
   ```

2. Plan the phase (document in issue)

3. Review architecture documentation

### During Development

1. **Small Commits**
   ```bash
   git add file.py
   git commit -m "feat(module): Add functionality X"
   ```

2. **Regular Testing**
   ```bash
   pytest tests/
   ```

3. **Code Quality**
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```

4. **Logging During Development**
   ```bash
   # Check logs for errors
   tail -f logs/app.log
   ```

### Before Submitting PR

1. Test on actual hardware (if possible)
2. Verify FPS performance
3. Check memory usage
4. Run full test suite
5. Update documentation

## Testing Strategy

### Unit Tests
```bash
# Test a specific module
pytest tests/test_pose.py -v

# Test with coverage
pytest --cov=ai.pose tests/test_pose.py
```

### Integration Tests
```bash
# Test module interactions
pytest tests/test_integration.py -v
```

### Performance Tests
```bash
# Run benchmarks
python tests/benchmark.py
```

## Common Development Tasks

### Add a New Configuration Option

1. Edit `config/default.yaml`:
   ```yaml
   new_module:
     new_option: value
   ```

2. Use in code:
   ```python
   config = ConfigManager()
   value = config.get("new_module.new_option")
   ```

### Add a New Dependency

1. Install: `pip install package_name`
2. Update `requirements.txt`: `pip freeze > requirements.txt`
3. Commit both files

### Debug GPU Issues

```python
import torch

print("GPU Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("GPU Memory:", torch.cuda.memory_allocated())
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## Debugging Tips

### Enable Debug Logging
```bash
export APP_LOG_LEVEL=DEBUG
python main.py
```

### Visualize Pose Landmarks
```python
import cv2
from ai.pose import PoseDetector

detector = PoseDetector()
landmarks = detector.detect(frame)

for landmark in landmarks:
    x, y = int(landmark.x * width), int(landmark.y * height)
    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Landmarks', frame)
cv2.waitKey(1)
```

### Check Memory Usage
```bash
# Monitor during execution
python -m memory_profiler main.py
```

## Performance Benchmarking

### FPS Measurement
```python
import time

fps_list = []
for i in range(100):
    start = time.time()
    result = process_frame(frame)
    fps = 1 / (time.time() - start)
    fps_list.append(fps)

avg_fps = sum(fps_list) / len(fps_list)
print(f"Average FPS: {avg_fps:.2f}")
```

### Latency Measurement
```python
import time

latencies = []
for i in range(100):
    start = time.time()
    result = process_frame(frame)
    latency = (time.time() - start) * 1000  # ms
    latencies.append(latency)

avg_latency = sum(latencies) / len(latencies)
print(f"Average Latency: {avg_latency:.2f}ms")
```

## Commit Message Convention

Follow this format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Examples:
```
feat(pose): Add Kalman filter smoothing
fix(camera): Handle USB disconnection
docs(draping): Update algorithm documentation
test(segmentation): Add unit tests
refactor(renderer): Optimize shader compilation
```

## Pull Request Checklist

- [ ] Code follows PEP 8
- [ ] Type hints on all functions
- [ ] Logging implemented
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Performance validated
- [ ] Error handling comprehensive
- [ ] Commit messages descriptive
- [ ] Branch up-to-date with main

## Resources

- [MediaPipe Documentation](https://mediapipe.dev/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [OpenCV Tutorials](https://docs.opencv.org/)
- [PySide6 Guide](https://wiki.qt.io/Qt_for_Python)
- [CUDA Programming](https://docs.nvidia.com/cuda/)

---

**Questions?** Open an issue or check the Architecture documentation.
