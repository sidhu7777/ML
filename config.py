import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ---------------------------------------------------
    # FLASK BASIC CONFIG
    # ---------------------------------------------------
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())
    DEBUG = False
    TESTING = False

    # Server
    PORT = int(os.getenv('PORT', 8080))

    # Base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------------------------------
    # UPLOAD / OUTPUT DIRECTORY HANDLING
    # ---------------------------------------------------
    if os.getenv('RENDER'):  # Render.com production
        UPLOAD_FOLDER = '/tmp/uploads'
        OUTPUT_FOLDER = '/tmp/outputs'
    else:  # Local development
        UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
        OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'geojson', 'json'}

    # ---------------------------------------------------
    # CORS
    # ---------------------------------------------------
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

    # ---------------------------------------------------
    # STORAGE SETTINGS (Optional)
    # ---------------------------------------------------
    USE_S3 = os.getenv('USE_S3', 'false').lower() == 'true'
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    S3_REGION = os.getenv('S3_REGION', 'us-east-1')
    CLOUDINARY_URL = os.getenv('CLOUDINARY_URL')

    # ---------------------------------------------------
    # DATABASE CONFIG (MOST IMPORTANT PART)
    # ---------------------------------------------------
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 🔥 FIXED: This prevents MySQL timeouts (your main issue)
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,     # auto-reconnect if connection is dead
        "pool_recycle": 280,       # recycle before MySQL 300s timeout
        "pool_size": 10,           # recommended size
        "max_overflow": 20,        # extra temp connections
    }

    # ---------------------------------------------------
    # TOOL-SPECIFIC SETTINGS
    # ---------------------------------------------------
    CELL_SITE_MIN_SAMPLES = int(os.getenv('CELL_SITE_MIN_SAMPLES', 30))
    CELL_SITE_BIN_SIZE = int(os.getenv('CELL_SITE_BIN_SIZE', 5))

    @staticmethod
    def init_app():
        """Ensure required folders exist."""
        for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER]:
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Created directory: {folder}")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class RenderConfig(ProductionConfig):
    """Production config for Render.com"""
    pass


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'render':       RenderConfig,
    'default':      DevelopmentConfig
}