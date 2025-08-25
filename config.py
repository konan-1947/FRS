import os

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    
    # Camera settings
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Face detection settings
    FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.9
    FACE_RECOGNITION_SIMILARITY_THRESHOLD = 0.8
    PROCESS_EVERY_N_FRAMES = 5  # Process every 5th frame for better performance
    
    # Frame processing settings
    MIN_FRAME_WIDTH = 48
    MIN_FRAME_HEIGHT = 48
    FRAME_RESIZE_MAX_WIDTH = 480  # Reduced for better performance
    FRAME_INTERPOLATION = 'INTER_AREA'  # Better for downsampling
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'data'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    THREADED = True
    
    # Performance settings
    GPU_MEMORY_GROWTH = True
    TENSORFLOW_LOG_LEVEL = 'ERROR'
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # Set TensorFlow logging
        import tensorflow as tf
        tf.get_logger().setLevel(Config.TENSORFLOW_LOG_LEVEL)
        
        # Configure GPU memory growth if available
        if Config.GPU_MEMORY_GROWTH:
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                try:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                except RuntimeError as e:
                    print(f"GPU memory growth setting failed: {e}")

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key'
    HOST = '127.0.0.1'  # More secure for production
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}