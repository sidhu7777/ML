# In models.py

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import (
    BIGINT, DATETIME, FLOAT, INTEGER, VARCHAR, TEXT, TINYINT
)
from sqlalchemy.schema import UniqueConstraint
from extensions import db
import datetime

# --- Main Prediction Log Table ---

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # What was processed?
    input_file = db.Column(db.String(512)) 
    output_dir = db.Column(db.String(512), index=True) 
    filename = db.Column(db.String(512), index=True, unique=True)
    
    # Link to project
    project_id = db.Column(db.Integer, index=True)
    
    # How was it processed?
    method = db.Column(db.String(50))
    min_samples = db.Column(db.Integer)
    
    def __repr__(self):
        return f'<Prediction {self.id} [{self.method}] -> {self.filename}>'


# --- Cell Site Locator Output Table (NO-ML) ---

class SiteNoMl(db.Model):
    __tablename__ = 'site_noMl'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, index=True)
    
    # --- Composite Key ---
    network = db.Column(db.String(100), nullable=False, index=True)
    earfcn_or_narfcn = db.Column(db.Integer, nullable=False, index=True)
    
    # --- 泙 START OF FIX 泙 ---
    # Changed from db.String(1) to db.String(100) to hold values like "1******7"
    site_key_inferred = db.Column(db.String(100), nullable=False, index=True)
    # --- 泙 END OF FIX 泙 ---
    
    pci_or_psi = db.Column(db.Integer, nullable=False, index=True)
    # --- End Key ---

    samples = db.Column(db.Integer)
    lat_pred = db.Column(db.Float)
    lon_pred = db.Column(db.Float)
    
    azimuth_deg_5 = db.Column(db.Integer)
    azimuth_deg_5_soft = db.Column(db.Integer)
    azimuth_deg_label_soft = db.Column(db.String(50))
    azimuth_adjustment_deg = db.Column(db.Float)
    template_spacing_deg = db.Column(db.Float)
    beamwidth_deg_est = db.Column(db.Integer)
    median_sample_distance_m = db.Column(db.Float)
    cell_id_representative = db.Column(db.String(255)) 
    sector_count = db.Column(db.Integer)
    azimuth_reliability = db.Column(db.Float)
    spacing_used = db.Column(db.String(50))

    __table_args__ = (
        UniqueConstraint('network', 'earfcn_or_narfcn', 'site_key_inferred', 'pci_or_psi', 'project_id', name='uq_site_noml_key_v2'),
    )

    def __repr__(self):
        return f'<SiteNoMl {self.id} (PCI {self.pci_or_psi})>'


# --- Cell Site Locator Output Table (ML) ---

class SiteMl(db.Model):
    __tablename__ = 'site_ml'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, index=True)
    
    # --- Composite Key ---
    network = db.Column(db.String(100), nullable=False, index=True)
    earfcn_or_narfcn = db.Column(db.Integer, nullable=False, index=True)
    site_key_inferred = db.Column(db.String(100), nullable=False, index=True) 
    pci_or_psi = db.Column(db.Integer, nullable=False, index=True)
    # --- End Key ---
    
    samples = db.Column(db.Integer)
    lat_pred = db.Column(db.Float)
    lon_pred = db.Column(db.Float)
    
    azimuth_deg_5 = db.Column(db.Integer)
    azimuth_deg_5_soft = db.Column(db.Integer)
    azimuth_deg_label_soft = db.Column(db.String(50))
    azimuth_adjustment_deg = db.Column(db.Float)
    template_spacing_deg = db.Column(db.Float)
    beamwidth_deg_est = db.Column(db.Integer)
    cell_id_representative = db.Column(db.String(255)) 
    sector_count = db.Column(db.Integer)
    azimuth_reliability = db.Column(db.Float)
    spacing_used = db.Column(db.String(50))
    
    # ML-specific
    range_mae_m = db.Column(db.Float)
    site_spread_m = db.Column(db.Float)
    
    __table_args__ = (
        UniqueConstraint('network', 'earfcn_or_narfcn', 'site_key_inferred', 'pci_or_psi', 'project_id', name='uq_site_ml_key_v2'),
    )

    def __repr__(self):
        return f'<SiteMl {self.id} (PCI {self.pci_or_psi})>'

class NetworkLog(db.Model):
    __tablename__ = 'tbl_network_log'
    
    id = db.Column(INTEGER(unsigned=True), primary_key=True)
    session_id = db.Column(BIGINT, nullable=False, index=True)
    timestamp = db.Column(DATETIME, index=True)
    lat = db.Column(FLOAT(precision=10, scale=6))
    lon = db.Column(FLOAT(precision=10, scale=6))
    band = db.Column(VARCHAR(64), index=True)
    m_alpha_long = db.Column(VARCHAR(45), index=True)
    network = db.Column(VARCHAR(45), index=True)
    earfcn = db.Column(VARCHAR(45))
    pci = db.Column(VARCHAR(45))
    rsrp = db.Column(FLOAT(precision=5, scale=2))
    rsrq = db.Column(FLOAT(precision=5, scale=2))
    sinr = db.Column(FLOAT(precision=5, scale=2))
    primary_cell_info_1 = db.Column(TEXT)
    ta = db.Column(VARCHAR(128))
    nodeb_id = db.Column(VARCHAR(255))
    cell_id = db.Column(VARCHAR(255))

    def __repr__(self):
        return f"<NetworkLog session={self.session_id} pci={self.pci} rsrp={self.rsrp}>"
