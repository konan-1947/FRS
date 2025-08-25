// Main JavaScript file for Face Detection System

class FaceDetectionApp {
    constructor() {
        this.isDetectionActive = false;
        this.statusUpdateInterval = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.setupErrorHandling();
    }
    
    bindEvents() {
        // Add event listeners that are common across pages
        document.addEventListener('DOMContentLoaded', () => {
            this.initializeTooltips();
            this.setupGlobalErrorHandling();
        });
    }
    
    initializeTooltips() {
        // Initialize Bootstrap tooltips if present
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    setupGlobalErrorHandling() {
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            this.showNotification('An unexpected error occurred', 'danger');
        });
        
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            this.showNotification('An unexpected error occurred', 'danger');
        });
    }
    
    setupErrorHandling() {
        // Setup global fetch error handling
        const originalFetch = window.fetch;
        window.fetch = (...args) => {
            return originalFetch(...args)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response;
                })
                .catch(error => {
                    console.error('Fetch error:', error);
                    this.showNotification('Network error occurred', 'danger');
                    throw error;
                });
        };
    }
    
    showNotification(message, type = 'info', duration = 5000) {
        const alertContainer = document.getElementById('alert-container');
        if (!alertContainer) {
            console.warn('Alert container not found');
            return;
        }
        
        const alertId = 'alert-' + Date.now();
        const alertDiv = document.createElement('div');
        alertDiv.id = alertId;
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            <i class="fas fa-${this.getIconForType(type)}"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        alertContainer.appendChild(alertDiv);
        
        // Auto-remove after duration
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert && alert.parentNode) {
                alert.remove();
            }
        }, duration);
    }
    
    getIconForType(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle',
            'primary': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
    
    async makeRequest(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`Request to ${url} failed:`, error);
            throw error;
        }
    }
    
    async startDetection() {
        try {
            const result = await this.makeRequest('/start_detection');
            
            if (result.status === 'success') {
                this.isDetectionActive = true;
                this.showNotification(result.message, 'success');
                return true;
            } else {
                this.showNotification(result.message, 'danger');
                return false;
            }
        } catch (error) {
            this.showNotification('Failed to start detection', 'danger');
            return false;
        }
    }
    
    async stopDetection() {
        try {
            const result = await this.makeRequest('/stop_detection');
            
            this.isDetectionActive = false;
            this.showNotification(result.message, 'info');
            return true;
        } catch (error) {
            this.showNotification('Failed to stop detection', 'danger');
            return false;
        }
    }
    
    async getDetectionStatus() {
        try {
            return await this.makeRequest('/detection_status');
        } catch (error) {
            console.error('Failed to get detection status:', error);
            return null;
        }
    }
    
    async addUser(formData) {
        try {
            const response = await fetch('/add_user', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showNotification(result.message, 'success');
                return true;
            } else {
                this.showNotification(result.message, 'danger');
                return false;
            }
        } catch (error) {
            this.showNotification('Failed to add user', 'danger');
            return false;
        }
    }
    
    formatTimestamp(date = new Date()) {
        return date.toLocaleTimeString();
    }
    
    formatConfidence(confidence) {
        return (confidence * 100).toFixed(1) + '%';
    }
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    throttle(func, limit) {
        let lastFunc;
        let lastRan;
        return function() {
            const context = this;
            const args = arguments;
            if (!lastRan) {
                func.apply(context, args);
                lastRan = Date.now();
            } else {
                clearTimeout(lastFunc);
                lastFunc = setTimeout(function() {
                    if ((Date.now() - lastRan) >= limit) {
                        func.apply(context, args);
                        lastRan = Date.now();
                    }
                }, limit - (Date.now() - lastRan));
            }
        };
    }
    
    validateImageFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
        const maxSize = 5 * 1024 * 1024; // 5MB
        
        if (!validTypes.includes(file.type)) {
            this.showNotification('Please select a valid image file (JPEG, PNG, GIF)', 'warning');
            return false;
        }
        
        if (file.size > maxSize) {
            this.showNotification('Image file size must be less than 5MB', 'warning');
            return false;
        }
        
        return true;
    }
    
    previewImage(file, previewElement) {
        if (this.validateImageFile(file)) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.src = e.target.result;
                previewElement.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }
    
    // Utility method to update UI elements safely
    updateElement(selector, content, isHTML = false) {
        const element = document.querySelector(selector);
        if (element) {
            if (isHTML) {
                element.innerHTML = content;
            } else {
                element.textContent = content;
            }
        }
    }
    
    // Utility method to toggle element visibility
    toggleElement(selector, show) {
        const element = document.querySelector(selector);
        if (element) {
            element.style.display = show ? 'block' : 'none';
        }
    }
    
    // Utility method to enable/disable buttons
    toggleButton(selector, enabled) {
        const button = document.querySelector(selector);
        if (button) {
            button.disabled = !enabled;
        }
    }
}

// Initialize the app
const app = new FaceDetectionApp();

// Make app globally available for debugging
window.FaceDetectionApp = app;

// Export for use in other modules if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceDetectionApp;
}