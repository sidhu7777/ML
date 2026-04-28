# In tools/cell_site/services.py

from flask import current_app
from werkzeug.utils import secure_filename
import os
import time
import pandas as pd
import traceback
import numpy as np
import uuid  
from sqlalchemy import create_engine
from dotenv import load_dotenv

from . import cell_site_processing as site
from extensions import db

from models import SiteNoMl, SiteMl
from sqlalchemy.dialects.mysql import insert as mysql_insert


# ============================================================
# 🔵 MULTI-REGION ENGINE SETUP
# ============================================================
load_dotenv()

taiwan_engine = create_engine(
    os.getenv("DATABASE_URL_Taiwan"),
    pool_pre_ping=True,
    pool_recycle=180,
    pool_size=20,
    max_overflow=40
) if os.getenv("DATABASE_URL_Taiwan") else None

def get_regional_engine(region_name):
    """Returns Taiwan engine if requested, otherwise defaults to India (db.engine)"""
    if str(region_name).lower() == "taiwan":
        if not taiwan_engine:
            raise ValueError("❌ DATABASE_URL_Taiwan not found in .env")
        return taiwan_engine
    return db.engine


class CellSiteService:
    
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    def allowed_file(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def process_file(self, file, params, project_id=None, region="india"):
        """Process uploaded cell site file"""

        filename = secure_filename(getattr(file, 'filename', 'uploaded.csv'))

        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if hasattr(file, 'save'):
            file.save(filepath)
        else:
            with open(filepath, 'wb') as f:
                f.write(file.read())

        timestamp = str(int(time.time()))
        outdir = os.path.join(current_app.config['OUTPUT_FOLDER'], f'cellsite_{timestamp}')
        os.makedirs(outdir, exist_ok=True)

        site.setup_logger(outdir, tag=params['method'])

        try:
            # ================= PROCESS ENGINE =================
            if params['method'] == 'noml':
                results = site.run_noml(
                    input_path=filepath,
                    outdir=outdir,
                    min_samples=params.get('min_samples', 30),
                    bin_size=params.get('bin_size', 5),
                    soft_spacing=params.get('soft_spacing', False),
                    use_ta=params.get('use_ta', False),
                    make_map=params.get('make_map', False),
                    merge_sites=params.get('soft_spacing', False)
                )
            else:
                results = site.run_ml()

            df = results.pop('dataframe', None)

            # ================= SAVE TO DATABASE =================
            if df is not None and not df.empty:
                
                # ✅ Get the correct engine for UPSERT
                current_engine = get_regional_engine(region)

                if project_id is not None:
                    df['project_id'] = project_id

                Model = SiteNoMl if params['method'] == 'noml' else SiteMl
                db_columns = Model.__table__.columns.keys()

                df = df[df.columns.intersection(db_columns)].copy()

                df['network'] = df.get('network', '').astype(str)

                for col in ['earfcn_or_narfcn', 'pci_or_psi', 'project_id']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                if 'site_key_inferred' in df.columns:
                    df['site_key_inferred'] = (
                        df['site_key_inferred']
                        .astype(str)
                        .replace(['nan', 'None', '', None], None)
                    )

                df['site_key_inferred'] = df.apply(
                    lambda row: row['site_key_inferred']
                    if isinstance(row['site_key_inferred'], str) and row['site_key_inferred'].strip() != ""
                    else f"AUTO_{uuid.uuid4().hex[:8]}_{int(row['earfcn_or_narfcn']) if pd.notna(row['earfcn_or_narfcn']) else 0}_{int(row['pci_or_psi']) if pd.notna(row['pci_or_psi']) else 0}_{int(row['project_id']) if pd.notna(row['project_id']) else 0}",
                    axis=1
                )

                df = df.dropna(subset=['network', 'earfcn_or_narfcn', 'pci_or_psi', 'project_id'])

                df = df.drop_duplicates(
                    subset=['network', 'earfcn_or_narfcn', 'site_key_inferred', 'pci_or_psi', 'project_id'],
                    keep='last'
                )

                df_clean = df.replace({np.nan: None})
                rows = df_clean.to_dict(orient='records')

                table = Model.__table__
                update_cols = [c for c in df.columns if c not in ['network', 'earfcn_or_narfcn', 'site_key_inferred', 'pci_or_psi', 'project_id', 'id']]

                # ✅ Use the regional engine here!
                with current_engine.begin() as conn:
                    stmt = mysql_insert(table).values(rows)
                    if update_cols:
                        stmt = stmt.on_duplicate_key_update(**{c: stmt.inserted[c] for c in update_cols})
                    else:
                        stmt = stmt.on_duplicate_key_update(id=stmt.inserted.id)
                    conn.execute(stmt)

            # ================= FORMAT RESPONSE =================
            relative_results = {
                key: os.path.basename(path)
                for key, path in results.items()
                if path and isinstance(path, str) and os.path.exists(path)
            }

            return {
                'success': True,
                'results': relative_results,
                'output_dir': os.path.basename(outdir),
                'message': 'File processed successfully'
            }
        
        except Exception as e:
            current_app.logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}", exc_info=True)
            raise
        
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass