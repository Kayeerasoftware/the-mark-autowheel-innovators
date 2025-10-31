class ObjectDetector {
    constructor() {
        this.model = null;
        this.session = null;
        this.isModelLoaded = false;
        this.isCameraActive = false;
        this.stream = null;
        
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
            'toothbrush', 'wheelchair' // Added wheelchair as it's important for our project
        ];
        
        this.initializeApp();
    }

    async initializeApp() {
        await this.loadModel();
        this.setupEventListeners();
        this.updateUI();
    }

    async loadModel() {
        try {
            this.showMessage('Loading AI model...', 'info');
            
            // For GitHub Pages, the model should be in the same directory
            this.session = await ort.InferenceSession.create('./best.onnx', {
                executionProviders: ['webgl'],
                graphOptimizationLevel: 'all'
            });
            
            this.isModelLoaded = true;
            this.showMessage('AI model loaded successfully!', 'success');
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
            this.showMessage('Error loading AI model. Please check console for details.', 'error');
        }
    }

    setupEventListeners() {
        document.getElementById('startCamera').addEventListener('click', () => this.startCamera());
        document.getElementById('stopCamera').addEventListener('click', () => this.stopCamera());
        document.getElementById('captureImage').addEventListener('click', () => this.captureAndDetect());
    }

    async startCamera() {
        try {
            this.showMessage('Accessing camera...', 'info');
            
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            
            const video = document.getElementById('webcam');
            video.srcObject = this.stream;
            
            video.onloadedmetadata = () => {
                video.play();
                this.isCameraActive = true;
                this.hideCameraPlaceholder();
                this.updateUI();
                this.showMessage('Camera activated successfully!', 'success');
            };
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showMessage('Error accessing camera. Please ensure you have granted camera permissions.', 'error');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.isCameraActive = false;
        this.showCameraPlaceholder();
        this.updateUI();
        this.showMessage('Camera stopped', 'info');
    }

    hideCameraPlaceholder() {
        document.getElementById('cameraPlaceholder').style.display = 'none';
    }

    showCameraPlaceholder() {
        document.getElementById('cameraPlaceholder').style.display = 'flex';
    }

    async captureAndDetect() {
        if (!this.isCameraActive || !this.isModelLoaded) {
            this.showMessage('Please start camera and ensure model is loaded', 'error');
            return;
        }

        try {
            const startTime = performance.now();
            this.showMessage('Processing image...', 'info');

            const video = document.getElementById('webcam');
            const canvas = document.getElementById('outputCanvas');
            const resultsContainer = document.getElementById('results');
            
            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Perform detection
            const detections = await this.detectObjects(canvas);
            
            // Draw bounding boxes
            this.drawDetections(ctx, detections);
            
            // Display results
            this.displayResults(detections);
            
            const endTime = performance.now();
            const processingTime = endTime - startTime;
            
            this.updatePerformanceMetrics(detections, processingTime);
            this.showMessage(`Detection completed in ${processingTime.toFixed(0)}ms`, 'success');
            
        } catch (error) {
            console.error('Error during detection:', error);
            this.showMessage('Error during object detection', 'error');
        }
    }

    async detectObjects(canvas) {
        // Preprocess image for YOLO
        const input = this.preprocessImage(canvas);
        
        // Run inference
        const feeds = { images: new ort.Tensor('float32', input, [1, 3, this.inputSize, this.inputSize]) };
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
            input[i / 4] = imageData.data[i] / 255.0;         // R
            input[imageData.data.length / 4 + i / 4] = imageData.data[i + 1] / 255.0; // G
            input[imageData.data.length / 2 + i / 4] = imageData.data[i + 2] / 255.0; // B
        }
        
        return input;
    }

    postprocessResults(results, originalWidth, originalHeight) {
        const detections = [];
        const output = results.output0.data;
        
        for (let i = 0; i < output.length; i += 6) {
            const [x1, y1, x2, y2, confidence, classId] = output.slice(i, i + 6);
            
            if (confidence > 0.5) { // Confidence threshold
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
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.drawImage(document.getElementById('webcam'), 0, 0, ctx.canvas.width, ctx.canvas.height);
        
        detections.forEach(det => {
            const [x, y, width, height] = det.bbox;
            
            // Draw bounding box
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);
            
            // Draw label background
            ctx.fillStyle = '#00ff00';
            const text = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(text).width;
            ctx.fillRect(x, y - 20, textWidth + 10, 20);
            
            // Draw label text
            ctx.fillStyle = '#000000';
            ctx.font = '16px Arial';
            ctx.fillText(text, x + 5, y - 5);
        });
    }

    displayResults(detections) {
        const resultsContainer = document.getElementById('results');
        
        if (detections.length === 0) {
            resultsContainer.innerHTML = `
                <div class="initial-state">
                    <span class="results-icon">🔍</span>
                    <p>No objects detected with high confidence</p>
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
        const avgConfidence = detections.length > 0 
            ? detections.reduce((sum, det) => sum + det.confidence, 0) / detections.length 
            : 0;
        
        document.getElementById('confidenceValue').textContent = `${(avgConfidence * 100).toFixed(1)}%`;
        document.getElementById('confidenceBar').style.width = `${avgConfidence * 100}%`;
        document.getElementById('processingTime').textContent = `${processingTime.toFixed(0)} ms`;
    }

    updateUI() {
        const startBtn = document.getElementById('startCamera');
        const stopBtn = document.getElementById('stopCamera');
        const captureBtn = document.getElementById('captureImage');
        
        startBtn.disabled = this.isCameraActive;
        stopBtn.disabled = !this.isCameraActive;
        captureBtn.disabled = !(this.isCameraActive && this.isModelLoaded);
    }

    showMessage(message, type = 'info') {
        // Simple message display - you can enhance this with toast notifications
        console.log(`${type.toUpperCase()}: ${message}`);
        
        // Update a status element if you add one to your HTML
        const statusElement = document.getElementById('status') || this.createStatusElement();
        statusElement.textContent = message;
        statusElement.className = `status status-${type}`;
    }

    createStatusElement() {
        const status = document.createElement('div');
        status.id = 'status';
        status.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 5px;
            color: white;
            z-index: 1000;
            max-width: 300px;
        `;
        document.body.appendChild(status);
        return status;
    }
}

// Color schemes for status messages
const statusColors = {
    info: '#3b82f6',
    success: '#10b981',
    error: '#ef4444',
    warning: '#f59e0b'
};

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ObjectDetector();
});

// Add CSS for status messages
const style = document.createElement('style');
style.textContent = `
    .status { 
        position: fixed; 
        top: 80px; 
        right: 20px; 
        padding: 12px 20px; 
        border-radius: 8px; 
        color: white; 
        z-index: 1000; 
        max-width: 300px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-weight: 500;
    }
    .status-info { background: #3b82f6; }
    .status-success { background: #10b981; }
    .status-error { background: #ef4444; }
    .status-warning { background: #f59e0b; }
`;
document.head.appendChild(style);
