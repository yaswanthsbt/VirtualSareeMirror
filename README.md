# Virtual Saree Mirror - AI-Powered Virtual Try-On System

![Project Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Objective

Build a **production-ready AI Virtual Saree Try-On system** for shopping malls. Customers stand in front of a camera and instantly see themselves wearing any selected saree with realistic draping and motion tracking.

## ✨ Key Features

- ✅ **Real-time Webcam Capture** - 30-60 FPS processing
- ✅ **Body Tracking** - MediaPipe Pose for 33 body landmarks
- ✅ **Human Segmentation** - Separate body from background
- ✅ **Pose Estimation** - Full body shape estimation
- ✅ **AI Draping Engine** - Realistic saree wrapping and pleating
- ✅ **Motion Tracking** - Natural saree deformation with body movement
- ✅ **Cloth Simulation** - Fabric-specific physics (Silk, Cotton, Georgette, etc.)
- ✅ **Occlusion Handling** - Hands and hair rendered above saree
- ✅ **Real-time Rendering** - OpenGL-based GPU acceleration
- ✅ **Shopping Mall UI** - Professional interface with saree catalog
- ✅ **Database** - SQLite saree catalog management

## 🏗️ Architecture

```
VirtualSareeMirror/
├── camera/                 # Camera capture & frame handling
├── ai/
│   ├── pose/              # Body landmark detection
│   ├── segmentation/      # Human segmentation
│   ├── body_mesh/         # Full body shape estimation
│   ├── tracking/          # Motion tracking & smoothing
│   └── draping/           # AI saree draping engine
├── renderer/              # Real-time rendering
├── saree/
│   ├── loader/            # Saree image loading
│   └── database/          # Saree catalog management
├── ui/                    # PySide6 GUI
├── models/                # Pre-trained model weights
├── assets/                # UI assets
├── config/                # Configuration files
├── utils/                 # Utilities & helpers
├── logs/                  # Application logs
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.12+ |
| **GUI** | PySide6 (Qt6) |
| **Computer Vision** | OpenCV |
| **Pose Detection** | MediaPipe Pose |
| **Segmentation** | MediaPipe Selfie / SAM2 |
| **Deep Learning** | PyTorch |
| **Rendering** | OpenGL |
| **GPU** | CUDA 11.8+ |
| **Database** | SQLite |
| **Config** | YAML |

## 📋 Requirements

- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **GPU**: NVIDIA CUDA-compatible (RTX 2060 or better recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models and assets
- **Camera**: 1080p or higher

## ⚡ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yaswanthsbt/VirtualSareeMirror.git
cd VirtualSareeMirror
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Run Application
```bash
python main.py
```

## 📚 Development Phases

- [x] **Phase 1** - Project Setup & Configuration
- [ ] **Phase 2** - Camera System Implementation
- [ ] **Phase 3** - Pose Estimation Module
- [ ] **Phase 4** - Human Segmentation
- [ ] **Phase 5** - Saree Loader & Database
- [ ] **Phase 6** - Basic Virtual Try-On
- [ ] **Phase 7** - Real-time Motion Tracking
- [ ] **Phase 8** - Advanced AI Draping Engine
- [ ] **Phase 9** - UI & Shopping Interface
- [ ] **Phase 10** - Performance Optimization & Deployment

## 🎨 User Interface

The application features a modern shopping mall interface:

```
┌─────────────────────────────────────────────────────────┐
│  Virtual Saree Mirror - Shopping Mall                  │
├──────────────┬──────────────────┬──────────────────────┤
│              │                  │                      │
│  Live        │  Virtual Mirror  │  Saree Catalog       │
│  Camera      │                  │  • Search            │
│              │                  │  • Category          │
│              │                  │  • Recently Viewed   │
│              │                  │  • Favorites         │
├──────────────┴──────────────────┴──────────────────────┤
│  [Previous] [Next] [Capture] [Compare] [Fullscreen]   │
└─────────────────────────────────────────────────────────┘
```

## 🗄️ Database Schema

### Saree Catalog Table
```sql
CREATE TABLE sarees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT,
    color TEXT,
    fabric TEXT,
    price REAL,
    occasion TEXT,
    image_path TEXT,
    thumbnail_path TEXT,
    barcode TEXT UNIQUE,
    qr_code TEXT,
    created_at TIMESTAMP
);
```

## ⚙️ Configuration

Edit `config/default.yaml` to customize:

- Camera settings (resolution, FPS, brightness)
- Pose detection thresholds
- Saree draping parameters
- Fabric physics properties
- UI theme and window settings
- Performance tuning

## 🔧 Key Modules

### Camera Manager
- Multi-camera support
- Frame rate management
- Image preprocessing

### Pose Detection
- 33-point body landmark detection
- Smooth tracking with jitter reduction
- Real-time inference

### Human Segmentation
- Background removal
- Precise mask generation
- GPU-accelerated processing

### AI Draping Engine ⭐
- Realistic saree wrapping
- Automatic pleats generation
- Pallu placement over shoulder
- Adaptive to body shape and size
- Multiple draping styles

### Motion Tracking
- Frame-to-frame consistency
- Walk detection and response
- Hand raise handling
- Body rotation tracking

### Cloth Simulation
- Fabric-specific physics
- Wind and gravity effects
- Collision detection
- Real-time deformation

## 🚀 Performance Goals

- **Latency**: < 100ms end-to-end
- **FPS**: 30-60 FPS (target 60)
- **GPU Memory**: < 4GB
- **CPU Usage**: < 60% on modern processors
- **No frame freezing** during real-time try-on

## 📝 Coding Standards

- ✅ Object-Oriented Programming (OOP)
- ✅ Type Hints for all functions
- ✅ Comprehensive logging
- ✅ PEP 8 compliant
- ✅ Detailed docstrings
- ✅ Error handling & validation
- ✅ Unit tests for all modules
- ✅ No code duplication

## 📖 Documentation

Detailed documentation for each module is provided in the respective folders:

- `camera/README.md` - Camera system documentation
- `ai/pose/README.md` - Pose detection guide
- `ai/segmentation/README.md` - Segmentation details
- `ai/draping/README.md` - Saree draping engine
- `ui/README.md` - UI component guide

## 🐛 Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Camera Not Working
- Check device ID in `.env`
- Verify camera permissions
- Test with OpenCV: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Low FPS
- Enable GPU acceleration
- Reduce camera resolution
- Disable pose landmarks visualization

## 📧 Support

For issues and feature requests, please open a GitHub Issue.

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Credits

Built with ❤️ using:
- [MediaPipe](https://mediapipe.dev/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [PySide6](https://wiki.qt.io/Qt_for_Python)

---

**Project Status**: Phase 1 Complete ✅  
**Last Updated**: 2026-06-29  
**Maintained By**: AI Computer Vision Team
