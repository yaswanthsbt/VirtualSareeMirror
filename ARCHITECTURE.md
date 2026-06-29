# Virtual Saree Mirror - Architecture Documentation

## System Architecture Overview

The Virtual Saree Mirror is built as a modular, multi-layered system designed for real-time performance and scalability.

```
┌─────────────────────────────────────────────────────┐
│          User Interface (PySide6 Qt)               │
├──────────────────────────────��──────────────────────┤
│         Application Controller / Event Loop         │
├─────────────────────────────────────────────────────┤
│  Camera  │  Pose      │  Segmentation │  Renderer  │
│  Manager │  Detection │  Engine       │  (OpenGL)  │
├──────────┴────────────┴───────────────┴────────────┤
│          Core AI Modules                           │
│  - Body Mesh Estimation                            │
│  - Saree Draping Engine ⭐                         │
│  - Motion Tracking                                 │
│  - Cloth Simulation                                │
├─────────────────────────────────────────────────────┤
│  Database Layer (SQLite)                           │
├─────────────────────────────────────────────────────┤
│  GPU/CUDA Acceleration                             │
└���────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. Camera Manager (Phase 2)
- **Purpose**: Capture and manage video frames from webcam
- **Responsibilities**:
  - Multi-camera support and selection
  - Frame rate control and buffering
  - Image preprocessing (resize, rotate, color conversion)
  - Frame synchronization with AI modules
- **Key Classes**:
  - `CameraManager`: Main camera interface
  - `FrameBuffer`: Thread-safe frame queue
  - `CameraDevice`: Device abstraction

### 2. Pose Detection (Phase 3)
- **Purpose**: Detect body landmarks and posture
- **Model**: MediaPipe Pose (33 landmarks)
- **Responsibilities**:
  - Real-time landmark detection
  - Jitter smoothing with Kalman filter
  - Confidence scoring
  - Movement velocity calculation
- **Key Classes**:
  - `PoseDetector`: MediaPipe wrapper
  - `Landmark`: Individual point tracking
  - `PoseSmoother`: Jitter reduction

### 3. Human Segmentation (Phase 4)
- **Purpose**: Separate body from background
- **Models**: MediaPipe Selfie / SAM2
- **Responsibilities**:
  - Binary mask generation
  - Precise edge detection
  - GPU-accelerated inference
  - Mask refinement and morphological operations
- **Key Classes**:
  - `SegmentationEngine`: Main segmentation interface
  - `MaskProcessor`: Mask refinement
  - `BodyMask`: Mask data container

### 4. Body Mesh (Phase 5)
- **Purpose**: Estimate full 3D body shape
- **Responsibilities**:
  - Shoulder position and width estimation
  - Waist and hip measurement
  - Chest circumference calculation
  - Leg length and positioning
  - Body type classification
- **Key Classes**:
  - `BodyMeshEstimator`: Main estimation engine
  - `BodyMetrics`: Body measurements container
  - `ClothingSize`: Automatic sizing (XS to XXL)

### 5. Saree Loader (Phase 5)
- **Purpose**: Load and process saree images
- **Responsibilities**:
  - Support PNG (transparent) and JPG formats
  - High-resolution image handling (up to 4K)
  - Automatic border detection
  - Pallu region identification
  - Pleats extraction and analysis
- **Key Classes**:
  - `SareeLoader`: Main loading interface
  - `SareeImage`: Processed saree container
  - `BorderDetector`: Automatic border finding
  - `PleatAnalyzer`: Pleats detection

### 6. AI Draping Engine (Phase 8) ⭐ CRITICAL
- **Purpose**: Realistic saree draping and cloth simulation
- **Responsibilities**:
  - Wrap saree around body based on measurements
  - Generate realistic pleats
  - Place pallu over shoulder
  - Adaptive sizing to body shape
  - Support multiple draping styles:
    - Nivi Style (Tamil Nadu)
    - Gujarati Style
    - Bengali Style
    - Maharashtrian Style
    - Punjabi Style
  - Physics-based cloth simulation
- **Key Classes**:
  - `DrapingEngine`: Main draping logic
  - `ClothSimulation`: Physics simulation
  - `PleatGenerator`: Procedural pleat creation
  - `DrapingStyle`: Style-specific implementations

### 7. Motion Tracking (Phase 7)
- **Purpose**: Track body movement and adapt saree in real-time
- **Responsibilities**:
  - Frame-to-frame consistency
  - Walk detection and gait analysis
  - Hand raise and pose changes
  - Body rotation tracking
  - Optical flow computation
  - Velocity-based saree deformation
- **Key Classes**:
  - `MotionTracker`: Main tracking system
  - `OpticalFlow`: Optical flow computation
  - `GaitAnalyzer`: Walking pattern detection
  - `VelocityEstimator`: Movement speed calculation

### 8. Cloth Simulation (Integrated with Draping)
- **Purpose**: Physics-based fabric behavior
- **Fabric Types**:
  - **Silk**: Low stiffness, high damping (elegant drape)
  - **Cotton**: Medium stiffness, medium damping (structured)
  - **Georgette**: Medium stiffness, high damping (flowing)
  - **Chiffon**: Very low stiffness, very high damping (light, airy)
  - **Linen**: High stiffness, medium damping (stiff, natural)
- **Physics Parameters**:
  - Mass distribution
  - Stiffness coefficient
  - Damping ratio
  - Gravity and wind simulation
  - Collision detection

### 9. Renderer (Phase 8)
- **Purpose**: Real-time OpenGL rendering
- **Responsibilities**:
  - Composite camera, saree, and segmentation
  - Anti-aliasing and filtering
  - Shadow and lighting effects
  - Frame rate management (target 60 FPS)
  - GPU memory optimization
- **Key Classes**:
  - `Renderer`: Main rendering engine
  - `ShaderProgram`: GPU shader management
  - `FrameBuffer`: Off-screen rendering

### 10. Saree Database (Phase 5)
- **Purpose**: Catalog management
- **Database Schema**:
  - Saree metadata (ID, name, brand, color, fabric)
  - Pricing and occasion information
  - Image paths and thumbnails
  - Barcode and QR code support
- **Features**:
  - Search by category, color, brand
  - Recently viewed tracking
  - Favorites management

### 11. UI (Phase 9)
- **Purpose**: Shopping mall interface
- **Components**:
  - Live camera feed panel
  - Virtual mirror display
  - Saree catalog sidebar
  - Control buttons
- **Features**:
  - Dark theme for shopping environment
  - FPS counter (optional)
  - Landmark visualization (debug)

## Data Flow

```
Camera Frame
    ↓
[Pose Detection] ────→ Body Landmarks (33 points)
    ↓
[Segmentation] ──────→ Body Mask
    ↓
[Body Mesh] ─────────→ Body Measurements
    ↓
[Motion Tracking] ───→ Velocity & Motion Vectors
    ↓
[Draping Engine] ────→ Saree Positioning
    ↓
[Cloth Simulation] ──→ Physics-based Deformation
    ↓
[Renderer] ──────────→ Final Composite Image
    ↓
 Display to Screen (30-60 FPS)
```

## Threading Model

```
Main Thread (GUI)
    ↓
[Event Loop] ─────→ Listens for user input
    ↓
Worker Threads (ThreadPool)
    ├─ Camera Capture Thread
    ├─ Pose Detection Thread
    ├─ Segmentation Thread
    ├─ Motion Tracking Thread
    ├─ Draping Engine Thread
    └─ Rendering Thread

Queues connect threads for thread-safe communication
```

## Performance Optimization Strategy

1. **GPU Acceleration**: All AI models on GPU via CUDA
2. **Multi-threading**: Parallel processing of independent modules
3. **Frame Skipping**: Process every Nth frame for lightweight operations
4. **Batch Processing**: Process multiple frames together
5. **Memory Pooling**: Reuse buffers instead of allocating new ones
6. **Lazy Loading**: Load models on-demand
7. **Resolution Scaling**: Adaptive resolution based on FPS

## Design Patterns Used

1. **Singleton**: Configuration, Logger
2. **Observer**: GUI event updates
3. **Strategy**: Different draping styles
4. **Factory**: Model creation
5. **Builder**: Complex object construction
6. **Pool**: Thread pool, memory pool

## Error Handling Strategy

- Try-catch blocks around GPU operations
- Graceful degradation (CPU fallback if GPU fails)
- User-friendly error messages
- Comprehensive logging for debugging
- Recovery mechanisms for common failures

## Testing Strategy

- Unit tests for each module
- Integration tests for data flow
- Performance benchmarks
- UI tests with mock data
- Real hardware testing

---

**Note**: This architecture is designed to be modular and scalable. Each phase builds upon previous phases without breaking existing functionality.
