// AutoWheel Object Detection Application - FIXED VERSION
class AutoWheelDetector {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.isCameraActive = false;
        this.stream = null;
        this.isRealtime = false;
        this.animationFrame = null;
        this.lastFpsTime = 0;
        this.frameCount = 0;
        this.currentFps = 0;
        
        // COCO-SSD class names (80 objects)
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
            'toothbrush'
        ];
        
        this.initializeApp();
    }

    async initializeApp() {
        this.showLoading(true);
        this.showMessage('🚀 Starting AutoWheel AI Demo...', 'info');
        
        try {
            await this.loadModel();
            this.setupEventListeners();
            this.updateUI();
            this.showLoading(false);
            this.showMessage('✅ AutoWheel AI Demo Ready! Click "Start Camera" to begin.', 'success');
        } catch (error) {
            this.showLoading(false);
            this.showMessage('❌ Failed to initialize application. Please refresh the page.', 'error');
            console.error('Initialization error:', error);
        }
    }

    async loadModel() {
        try {
            this.updateLoadingProgress(20);
            this.showMessage('🔧 Loading TensorFlow.js engine...', 'info');
            
            // Check if TensorFlow.js is available
            if (typeof tf === 'undefined') {
                throw new Error('TensorFlow.js not loaded');
            }
            
            this.updateLoadingProgress(50);
            this.showMessage('🤖 Loading COCO-SSD model (80 objects)...', 'info');
            
            // Load COCO-SSD model - this is reliable and works online
            this.model = await cocoSsd.load();
            
            this.updateLoadingProgress(100);
            this.isModelLoaded = true;
            
            // Update model status
            document.getElementById('modelStatusText').textContent = 'Loaded ✅';
            document.getElementById('modelStatusText').style.color = '#10b981';
            
            console.log('✅ COCO-SSD model loaded successfully');
            this.showMessage('✅ AI Model loaded! 80+ object classes ready.', 'success');
            
        } catch (error) {
            console.error('❌ Model loading failed:', error);
            this.isModelLoaded = false;
            document.getElementById('modelStatusText').textContent = 'Failed ❌';
            document.getElementById('modelStatusText').style.color = '#ef4444';
            
            this.showMessage('❌ AI model failed to load. Using demo mode.', 'error');
            
            // Even if model fails, we can continue with demo mode
            this.isModelLoaded = true;
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
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.isCameraActive) {
                this.stopCamera();
            }
        });
    }

    async startCamera() {
        try {
            this.showMessage('📷 Accessing camera...', 'info');
            
            // Try different camera constraints for better compatibility
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 },
                    facingMode: 'environment'
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
                this.showMessage('✅ Camera started! Point at objects to detect.', 'success');
                
                // Auto-start real-time detection
                this.isRealtime = true;
                this.startRealtimeDetection();
                
            };
            
        } catch (error) {
            console.error('Camera error:', error);
            this.handleCameraError(error);
        }
    }

    stopCamera() {
        // Stop animation frame
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        // Stop camera stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.isCameraActive = false;
        this.isRealtime = false;
        this.showCameraPlaceholder();
        this.updateUI();
        this.showMessage('⏹️ Camera stopped', 'info');
    }

    async toggleRealtime() {
        this.isRealtime = !this.isRealtime;
        
        if (this.isRealtime && this.isCameraActive) {
            this.startRealtimeDetection();
            this.showMessage('🔄 Real-time detection started!', 'success');
        } else if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
            this.showMessage('⏸️ Real-time detection paused', 'info');
        }
        
        this.updateUI();
    }

    async singleDetection() {
        if (!this.isCameraActive && document.getElementById('uploadedImage').style.display !== 'block') {
            this.showMessage('⚠️ Please start camera or upload an image first', 'warning');
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
            this.showMessage('❌ AI model not ready', 'error');
            return;
        }

        const startTime = performance.now();
        
        try {
            const canvas = document.getElementById('outputCanvas');
            let sourceElement;
            
            if (this.isCameraActive) {
                sourceElement = document.getElementById('webcam');
            } else {
                sourceElement = document.getElementById('uploadedImage');
            }
            
            // Set canvas dimensions to match source
            canvas.width = sourceElement.videoWidth || sourceElement.naturalWidth;
            canvas.height = sourceElement.videoHeight || sourceElement.naturalHeight;
            
            const ctx = canvas.getContext('2d');
            
            // Draw the source image onto canvas
            ctx.drawImage(sourceElement, 0, 0, canvas.width, canvas.height);
            
            // Perform object detection
            let predictions = [];
            
            if (this.model) {
                // Use COCO-SSD model
                predictions = await this.model.detect(canvas);
                
                // Convert to our format
                predictions = predictions.map(pred => ({
                    bbox: [pred.bbox[0], pred.bbox[1], pred.bbox[2], pred.bbox[3]],
                    confidence: pred.score,
                    class: pred.class,
                    label: pred.class
                }));
            } else {
                // Fallback demo mode
                predictions = this.generateDemoDetections();
            }
            
            // Draw bounding boxes and labels
            this.drawDetections(ctx, predictions);
            
            // Display results
            this.displayResults(predictions);
            
            const endTime = performance.now();
            const processingTime = endTime - startTime;
            
            this.updatePerformanceMetrics(predictions, processingTime);
            
        } catch (error) {
            console.error('Detection error:', error);
            this.showMessage('❌ Detection failed', 'error');
        }
    }

    drawDetections(ctx, predictions) {
        // Clear previous drawings by redrawing the original image
        if (this.isCameraActive) {
            const video = document.getElementById('webcam');
            ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);
        } else {
            const img = document.getElementById('uploadedImage');
            ctx.drawImage(img, 0, 0, ctx.canvas.width, ctx.canvas.height);
        }
        
        predictions.forEach(pred => {
            const [x, y, width, height] = pred.bbox;
            const label = `${pred.label} ${(pred.confidence * 100).toFixed(1)}%`;
            
            // Choose color based on class
            let color = '#00ff00'; // Default green
            if (pred.label === 'person') color = '#ef4444';
            else if (pred.label === 'chair') color = '#10b981';
            else if (pred.label === 'car' || pred.label === 'truck' || pred.label === 'bus') color = '#3b82f6';
            else if (pred.label === 'door') color = '#f59e0b';
            
            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);
            
            // Draw label background
            ctx.fillStyle = color;
            const textWidth = ctx.measureText(label).width;
            const textHeight = 20;
            ctx.fillRect(x, y - textHeight, textWidth + 10, textHeight);
            
            // Draw label text
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 14px Arial';
            ctx.fillText(label, x + 5, y - 5);
        });
    }

    displayResults(predictions) {
        const resultsContainer = document.getElementById('results');
        
        if (!predictions || predictions.length === 0) {
            resultsContainer.innerHTML = `
                <div class="initial-state">
                    <div class="results-icon">🔍</div>
                    <p>No objects detected</p>
                    <p class="results-sub">Try pointing at different objects</p>
                </div>
            `;
            return;
        }
        
        // Sort by confidence (highest first)
        predictions.sort((a, b) => b.confidence - a.confidence);
        
        let html = '';
        predictions.forEach(pred => {
            const confidencePercent = (pred.confidence * 100).toFixed(1);
            let confidenceClass = 'confidence-low';
            if (pred.confidence > 0.7) confidenceClass = 'confidence-high';
            else if (pred.confidence > 0.5) confidenceClass = 'confidence-medium';
            
            html += `
                <div class="detection-item ${confidenceClass}">
                    <div class="detection-label">${pred.label}</div>
                    <div class="detection-confidence">${confidencePercent}% confidence</div>
                </div>
            `;
        });
        
        resultsContainer.innerHTML = html;
    }

    updatePerformanceMetrics(predictions, processingTime) {
        const objectsCount = predictions ? predictions.length : 0;
        const avgConfidence = predictions && predictions.length > 0 
            ? predictions.reduce((sum, pred) => sum + pred.confidence, 0) / predictions.length 
            : 0;
        
        // Update counters
        document.getElementById('objectsCount').textContent = `${objectsCount} objects`;
        document.getElementById('confidenceAvg').textContent = `${(avgConfidence * 100).toFixed(1)}% avg`;
        document.getElementById('processingTime').textContent = `${processingTime.toFixed(1)} ms`;
        
        // Update confidence bar
        const confidenceBar = document.getElementById('confidenceBar');
        confidenceBar.style.width = `${avgConfidence * 100}%`;
        document.getElementById('confidenceValue').textContent = `${(avgConfidence * 100).toFixed(1)}%`;
        
        // Update confidence bar color based on average confidence
        if (avgConfidence > 0.7) {
            confidenceBar.style.background = '#10b981';
        } else if (avgConfidence > 0.5) {
            confidenceBar.style.background = '#f59e0b';
        } else {
            confidenceBar.style.background = '#ef4444';
        }
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

        this.showMessage('📁 Loading image...', 'info');
        
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
                this.showMessage('✅ Image analyzed!', 'success');
            };
        };
        
        reader.readAsDataURL(file);
    }

    generateDemoDetections() {
        // Generate realistic demo detections for fallback
        return [
            {
                bbox: [100, 100, 80, 200],
                confidence: 0.92,
                class: 'person',
                label: 'person'
            },
            {
                bbox: [300, 250, 120, 150],
                confidence: 0.88,
                class: 'chair',
                label: 'chair'
            },
            {
                bbox: [450, 150, 100, 180],
                confidence: 0.78,
                class: 'laptop',
                label: 'laptop'
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
        singleBtn.disabled = !this.isCameraActive && document.getElementById('uploadedImage').style.display !== 'block';
        
        // Update realtime button text and style
        if (this.isRealtime) {
            realtimeBtn.innerHTML = '<span class="btn-icon">⏸️</span> Pause Detection';
            realtimeBtn.style.background = '#ef4444';
        } else {
            realtimeBtn.innerHTML = '<span class="btn-icon">🔄</span> Real-time Detection';
            realtimeBtn.style.background = '#10b981';
        }
    }

    handleCameraError(error) {
        let message = 'Camera access failed. ';
        
        if (error.name === 'NotAllowedError') {
            message += 'Please allow camera permissions in your browser settings.';
        } else if (error.name === 'NotFoundError') {
            message += 'No camera found. Please use image upload or demo images.';
        } else if (error.name === 'NotSupportedError') {
            message += 'Your browser does not support camera access.';
        } else {
            message += 'Please try using image upload feature.';
        }
        
        this.showMessage(message, 'error');
        
        // Suggest alternative
        setTimeout(() => {
            this.showMessage('💡 Try using the demo images or upload your own photo!', 'info');
        }, 3000);
    }

    showMessage(message, type = 'info') {
        const statusElement = document.getElementById('statusMessage');
        
        // Set styles based on type
        const backgroundColor = 
            type === 'success' ? '#10b981' :
            type === 'error' ? '#ef4444' :
            type === 'warning' ? '#f59e0b' : '#3b82f6';
        
        statusElement.textContent = message;
        statusElement.style.background = backgroundColor;
        statusElement.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 5000);
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

// Global functions for demo images
function loadDemoImage(type) {
    const app = window.autoWheelApp;
    if (!app) return;
    
    const images = {
        classroom: 'https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=400&h=300&fit=crop',
        corridor: 'https://images.unsplash.com/photo-1560448204-603b3fc33ddc?w=400&h=300&fit=crop',
        office: 'https://images.unsplash.com/photo-1497366754035-f200968a6e72?w=400&h=300&fit=crop'
    };
    
    app.showMessage('🖼️ Loading demo image...', 'info');
    
    const img = document.getElementById('uploadedImage');
    img.src = images[type];
    img.style.display = 'block';
    app.hideCameraPlaceholder();
    
    // Stop camera if active
    if (app.isCameraActive) {
        app.stopCamera();
    }
    
    document.getElementById('fileInfo').textContent = `Demo: ${type}`;
    
    img.onload = async () => {
        await app.detectObjects();
        app.showMessage('✅ Demo image analyzed!', 'success');
    };
}

function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

// Initialize application when page loads
let autoWheelApp;

document.addEventListener('DOMContentLoaded', () => {
    autoWheelApp = new AutoWheelDetector();
    window.autoWheelApp = autoWheelApp; // Make it globally available
});

// Add CSS for status message
const statusStyle = document.createElement('style');
statusStyle.textContent = `
    .status-message {
        position: fixed;
        top: 100px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        z-index: 1000;
        max-width: 400px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        font-weight: 500;
        display: none;
        animation: slideInRight 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .camera-tips {
        background: #fef3cd;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.875rem;
    }
    
    .camera-tips p {
        margin: 0;
        color: #92400e;
    }
    
    .demo-image-card {
        background: var(--surface);
        border: 2px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .demo-image-card:hover {
        border-color: var(--primary);
        transform: translateY(-2px);
    }
    
    .demo-image-placeholder {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .model-info {
        background: var(--background);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .model-stats {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .model-stat {
        display: flex;
        justify-content: space-between;
        font-size: 0.875rem;
    }
    
    .model-badge {
        background: #ff6b35;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .loading-note {
        font-size: 0.875rem;
        color: var(--text-light);
        margin-top: 0.5rem;
    }
`;
document.head.appendChild(statusStyle);
