// AutoWheel Object Detection Application
class AutoWheelDetector {
    constructor() {
        this.model = null;
        this.session = null;
        this.isModelLoaded = false;
        this.isCameraActive = false;
        this.stream = null;
        this.isRealtime = false;
        this.animationFrame = null;
        this.lastFpsTime = 0;
        this.frameCount = 0;
        this.currentFps = 0;
        
        // YOLO configuration
        this.inputSize = 640;
        this.classNames = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'wheelchair'
        ];
        
        this.initializeApp();
    }

    async initializeApp() {
        this.showLoading(true);
        await this.loadModel();
        this.setupEventListeners();
        this.setupDemoImages();
        this.updateUI();
        this.showLoading(false);
        this.showMessage('AutoWheel AI Demo Ready! 🚀', 'success');
    }

    async loadModel() {
        try {
            this.updateLoadingProgress(30);
            this.showMessage('Loading YOLOv8 AI model...', 'info');
            
            // Try to load ONNX model first
            try {
                this.session = await ort.InferenceSession.create('./model/best.onnx', {
                    executionProviders: ['webgl'],
                    graphOptimizationLevel: 'all'
                });
                this.isModelLoaded = true;
                this.updateLoadingProgress(100);
                this.showMessage('YOLOv8 model loaded successfully! 🤖', 'success');
                return;
            } catch (onnxError) {
                console.warn('ONNX model failed, using TensorFlow.js fallback:', onnxError);
            }
            
            // Fallback to TensorFlow.js
            this.updateLoadingProgress(60);
            if (typeof cocoSsd !== 'undefined') {
                this.model = await cocoSsd.load();
                this.isModelLoaded = true;
                this.updateLoadingProgress(100);
                this.showMessage('TensorFlow.js model loaded! 🔍', 'success');
            } else {
                throw new Error('No AI model available');
            }
            
        } catch (error) {
            console.error('Model loading failed:', error);
            this.isModelLoaded = false;
            this.showMessage('AI model failed to load. Using demo mode.', 'error');
        }
    }

    setupEventListeners() {
        // Camera controls
        document.getElementById('startCamera').addEventListener('click', () => this.startCamera());
        document.getElementById('stopCamera').addEventListener('click', () => this.stopCamera());
        
        // Detection modes
        document.getElementById('realtimeToggle').addEventListener('click', () => this.toggleRealtime());
        document.getElementById('singleDetection').addEventListener('click', () => this.singleDetection());
        
        // File upload
        document.getElementById('fileInput').addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Handle visibility change for camera
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.isCameraActive) {
                this.stopCamera();
            }
        });
    }

    setupDemoImages() {
        // Preload demo images
        const demoImages = [
            'assets/demo-images/classroom.jpg',
            'assets/demo-images/corridor.jpg', 
            'assets/demo-images/crowded.jpg'
        ];
        
        demoImages.forEach(src => {
            const img = new Image();
            img.src = src;
        });
    }

    async startCamera() {
        try {
            this.showMessage('Starting camera...', 'info');
            
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            const video = document.getElementById('webcam');
            
            video.srcObject = this.stream;
            
            video.onloadedmetadata = () => {
                video.play();
                this.isCameraActive = true;
                this.hideCameraPlaceholder();
                this.updateUI();
                this.showMessage('Camera started successfully! 📷', 'success');
                
                // Start real-time detection if enabled
                if (this.isRealtime) {
                    this.startRealtimeDetection();
                }
            };
            
        } catch (error) {
            console.error('Camera error:', error);
            this.handleCameraError(error);
        }
    }

    stopCamera() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.isCameraActive = false;
        this.isRealtime = false;
        this.showCameraPlaceholder();
        this.updateUI();
        this.showMessage('Camera stopped', 'info');
    }

    async toggleRealtime() {
        this.isRealtime = !this.isRealtime;
        
        if (this.isRealtime && this.isCameraActive) {
            this.startRealtimeDetection();
            this.showMessage('Real-time detection started! 🔄', 'success');
        } else if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
            this.showMessage('Real-time detection stopped', 'info');
        }
        
        this.updateUI();
    }

    async singleDetection() {
        if (!this.isCameraActive && !document.getElementById('uploadedImage').style.display !== 'none') {
            this.showMessage('Please start camera or upload an image first', 'warning');
            return;
        }
        
        await this.detectObjects();
    }

    async startRealtimeDetection() {
        if (!this.isCameraActive || !this.isRealtime) return;
        
        const processFrame = async () => {
            if (!this.isCameraActive || !this.isRealtime) return;
            
            await this.detectObjects();
            this.updateFPS();
            
            this.animationFrame = requestAnimationFrame(processFrame);
        };
        
        this.animationFrame = requestAnimationFrame(processFrame);
    }

    async detectObjects() {
        if (!this.isModelLoaded) {
            this.showMessage('AI model not loaded', 'error');
            return;
        }

        const startTime = performance.now();
        
        try {
            let canvas = document.getElementById('outputCanvas');
            let sourceElement;
            
            if (this.isCameraActive) {
                sourceElement = document.getElementById('webcam');
            } else {
                sourceElement = document.getElementById('uploadedImage');
            }
            
            // Set canvas dimensions
            canvas.width = sourceElement.videoWidth || sourceElement.naturalWidth;
            canvas.height = sourceElement.videoHeight || sourceElement.naturalHeight;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(sourceElement, 0, 0, canvas.width, canvas.height);
            
            let detections = [];
            
            if (this.session) {
                // Use ONNX model
                detections = await this.detectWithONNX(canvas);
            } else if (this.model) {
                // Use TensorFlow.js model
                detections = await this.model.detect(canvas);
                detections = detections.map(det => ({
                    bbox: det.bbox,
                    confidence: det.score,
                    class: det.class,
                    label: det.class
                }));
            } else {
                // Demo mode
                detections = this.generateDemoDetections();
            }
            
            // Draw detections
            this.drawDetections(ctx, detections);
            
            // Display results
            this.displayResults(detections);
            
            const endTime = performance.now();
            const processingTime = endTime - startTime;
            
            this.updatePerformanceMetrics(detections, processingTime);
            
        } catch (error) {
            console.error('Detection error:', error);
            this.showMessage('Detection failed', 'error');
        }
    }

    async detectWithONNX(canvas) {
        // Preprocess image
        const input = this.preprocessImage(canvas);
        
        // Run inference
        const feeds = { 
            images: new ort.Tensor('float32', input, [1, 3, this.inputSize, this.inputSize]) 
        };
        const results = await this.session.run(feeds);
        
        // Post-process results
        return this.postprocessResults(results, canvas.width, canvas.height);
    }

    preprocessImage(canvas) {
        const ctx = document.createElement('canvas').getContext('2d');
        ctx.canvas.width = this.inputSize;
        ctx.canvas.height = this.inputSize;
        
        // Draw and resize image
        ctx.drawImage(canvas, 0, 0, this.inputSize, this.inputSize);
        
        const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
        const input = new Float32Array(3 * this.inputSize * this.inputSize);
        
        // Normalize and convert to CHW format
        for (let i = 0; i < imageData.data.length; i += 4) {
            input[i / 4] = imageData.data[i] / 255.0;
            input[imageData.data.length / 4 + i / 4] = imageData.data[i + 1] / 255.0;
            input[imageData.data.length / 2 + i / 4] = imageData.data[i + 2] / 255.0;
        }
        
        return input;
    }

    postprocessResults(results, originalWidth, originalHeight) {
        const detections = [];
        const output = results.output0.data;
        
        for (let i = 0; i < output.length; i += 6) {
            const [x1, y1, x2, y2, confidence, classId] = output.slice(i, i + 6);
            
            if (confidence > 0.5) {
                detections.push({
                    bbox: [
                        x1 * originalWidth / this.inputSize,
                        y1 * originalHeight / this.inputSize,
                        (x2 - x1) * originalWidth / this.inputSize,
                        (y2 - y1) * originalHeight / this.inputSize
                    ],
                    confidence: confidence,
                    class: parseInt(classId),
                    label: this.classNames[parseInt(classId)] || 'unknown'
                });
            }
        }
        
        return detections;
    }

    drawDetections(ctx, detections) {
        // Clear previous drawings by redrawing the original image
        if (this.isCameraActive) {
            ctx.drawImage(document.getElementById('webcam'), 0, 0, ctx.canvas.width, ctx.canvas.height);
        } else {
            ctx.drawImage(document.getElementById('uploadedImage'), 0, 0, ctx.canvas.width, ctx.canvas.height);
        }
        
        detections.forEach(det => {
            const [x, y, width, height] = det.bbox;
            const label = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
            
            // Choose color based on class
            let color = '#00ff00'; // Default green
            if (det.label === 'person') color = '#ef4444';
            if (det.label === 'chair') color = '#10b981';
            if (det.label === 'door') color = '#3b82f6';
            
            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);
            
            // Draw label background
            ctx.fillStyle = color;
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x, y - 25, textWidth + 10, 25);
            
            // Draw label text
            ctx.fillStyle = '#000000';
            ctx.font = '14px Arial';
            ctx.fillText(label, x + 5, y - 8);
        });
    }

    displayResults(detections) {
        const resultsContainer = document.getElementById('results');
        
        if (!detections || detections.length === 0) {
            resultsContainer.innerHTML = `
                <div class="initial-state">
                    <div class="results-icon">🔍</div>
                    <p>No objects detected</p>
                    <p class="results-sub">Try adjusting camera or image</p>
                </div>
            `;
            return;
        }
        
        // Sort by confidence
        detections.sort((a, b) => b.confidence - a.confidence);
        
        let html = '';
        detections.forEach(det => {
            const confidencePercent = (det.confidence * 100).toFixed(1);
            let confidenceClass = 'confidence-low';
            if (det.confidence > 0.7) confidenceClass = 'confidence-high';
            else if (det.confidence > 0.5) confidenceClass = 'confidence-medium';
            
            html += `
                <div class="detection-item ${confidenceClass}">
                    <div class="detection-label">${det.label}</div>
                    <div class="detection-confidence">${confidencePercent}% confidence</div>
                </div>
            `;
        });
        
        resultsContainer.innerHTML = html;
    }

    updatePerformanceMetrics(detections, processingTime) {
        const objectsCount = detections ? detections.length : 0;
        const avgConfidence = detections.length > 0 
            ? detections.reduce((sum, det) => sum + det.confidence, 0) / detections.length 
            : 0;
        
        document.getElementById('objectsCount').textContent = `${objectsCount} objects`;
        document.getElementById('confidenceAvg').textContent = `${(avgConfidence * 100).toFixed(1)}% avg`;
        document.getElementById('processingTime').textContent = `${processingTime.toFixed(1)} ms`;
        
        // Update confidence bar
        const confidenceBar = document.getElementById('confidenceBar');
        confidenceBar.style.width = `${avgConfidence * 100}%`;
        document.getElementById('confidenceValue').textContent = `${(avgConfidence * 100).toFixed(1)}%`;
        
        // Update model status
        document.getElementById('modelStatus').textContent = this.session ? 'YOLOv8 (ONNX)' : 'COCO-SSD (TF.js)';
    }

    updateFPS() {
        this.frameCount++;
        const now = performance.now();
        
        if (now >= this.lastFpsTime + 1000) {
            this.currentFps = Math.round((this.frameCount * 1000) / (now - this.lastFpsTime));
            this.frameCount = 0;
            this.lastFpsTime = now;
            
            document.getElementById('fpsValue').textContent = this.currentFps;
            document.getElementById('detectionStatus').textContent = 
                this.isRealtime ? `Real-time (${this.currentFps}FPS)` : 'Single Detection';
        }
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.showMessage('Loading image...', 'info');
        
        const img = document.getElementById('uploadedImage');
        const reader = new FileReader();
        
        reader.onload = (e) => {
            img.src = e.target.result;
            img.style.display = 'block';
            this.hideCameraPlaceholder();
            
            // Stop camera if active
            if (this.isCameraActive) {
                this.stopCamera();
            }
            
            document.getElementById('fileInfo').textContent = `File: ${file.name}`;
            
            // Auto-detect on image load
            img.onload = async () => {
                await this.detectObjects();
            };
        };
        
        reader.readAsDataURL(file);
    }

    loadDemoImage(src) {
        this.showMessage('Loading demo image...', 'info');
        
        const img = document.getElementById('uploadedImage');
        img.src = src;
        img.style.display = 'block';
        this.hideCameraPlaceholder();
        
        // Stop camera if active
        if (this.isCameraActive) {
            this.stopCamera();
        }
        
        document.getElementById('fileInfo').textContent = 'Demo image loaded';
        
        img.onload = async () => {
            await this.detectObjects();
        };
    }

    generateDemoDetections() {
        // Fallback demo detections
        return [
            {
                bbox: [100, 100, 150, 300],
                confidence: 0.92,
                class: 0,
                label: 'person'
            },
            {
                bbox: [300, 200, 100, 150],
                confidence: 0.88,
                class: 56,
                label: 'chair'
            },
            {
                bbox: [500, 150, 120, 200],
                confidence: 0.85,
                label: 'door'
            }
        ];
    }

    hideCameraPlaceholder() {
        document.getElementById('cameraPlaceholder').style.display = 'none';
    }

    showCameraPlaceholder() {
        document.getElementById('cameraPlaceholder').style.display = 'flex';
        document.getElementById('uploadedImage').style.display = 'none';
    }

    updateUI() {
        const startBtn = document.getElementById('startCamera');
        const stopBtn = document.getElementById('stopCamera');
        const realtimeBtn = document.getElementById('realtimeToggle');
        const singleBtn = document.getElementById('singleDetection');
        
        startBtn.disabled = this.isCameraActive;
        stopBtn.disabled = !this.isCameraActive;
        realtimeBtn.disabled = !this.isCameraActive;
        singleBtn.disabled = !this.isCameraActive && !document.getElementById('uploadedImage').style.display !== 'none';
        
        // Update realtime button text
        realtimeBtn.innerHTML = this.isRealtime ? 
            '<span class="btn-icon">⏸️</span> Stop Real-time' : 
            '<span class="btn-icon">🔍</span> Real-time Detection';
    }

    handleCameraError(error) {
        let message = 'Camera access failed. ';
        
        if (error.name === 'NotAllowedError') {
            message += 'Please allow camera permissions in your browser settings.';
        } else if (error.name === 'NotFoundError') {
            message += 'No camera found. Please use image upload instead.';
        } else {
            message += 'Please try using image upload feature.';
        }
        
        this.showMessage(message, 'error');
    }

    showMessage(message, type = 'info') {
        // Create or get status element
        let statusElement = document.getElementById('statusMessage');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.id = 'statusMessage';
            statusElement.style.cssText = `
                position: fixed;
                top: 100px;
                right: 20px;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                color: white;
                z-index: 1000;
                max-width: 300px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                font-weight: 500;
                transition: all 0.3s;
            `;
            document.body.appendChild(statusElement);
        }
        
        statusElement.textContent = message;
        statusElement.style.background = 
            type === 'success' ? '#10b981' :
            type === 'error' ? '#ef4444' :
            type === 'warning' ? '#f59e0b' : '#3b82f6';
        
        statusElement.style.display = 'block';
        
        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 4000);
    }

    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = show ? 'flex' : 'none';
    }

    updateLoadingProgress(percent) {
        const progressBar = document.getElementById('loadingProgress');
        progressBar.style.width = `${percent}%`;
    }
}

// Utility functions
function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

// Initialize application
let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new AutoWheelDetector();
});

// Service Worker for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
