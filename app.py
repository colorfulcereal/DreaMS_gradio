"""
DreaMS Gradio Web Application

This module provides a web interface for the DreaMS (Deep Representations Empowering
the Annotation of Mass Spectra) tool using Gradio. It allows users to upload MS/MS
files and perform library matching with DreaMS embeddings.

Author: DreaMS Team
License: MIT
"""

import gradio as gr
import spaces
import shutil
import torch
import urllib.request
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from io import BytesIO
from PIL import Image
import io
import dreams.utils.spectra as su
import dreams.utils.io as dio
from dreams.utils.data import MSData
from dreams.api import dreams_embeddings, dreams_predictions
from dreams.definitions import *
from massspecgym.models.pfas import HalogenDetectorDreamsTest
from pathlib import Path
from tqdm import tqdm
from dreams.utils.io import append_to_stem
from dreams.utils.dformats import DataFormatA

# ============================================================================
# DETERMINISTIC BEHAVIOR FOR REPRODUCIBILITY
# ============================================================================
# Ensure consistent predictions across CPU and GPU
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch, 'use_deterministic_algorithms'):
    torch.use_deterministic_algorithms(True, warn_only=True)

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Optimized image sizes for better performance
SMILES_IMG_SIZE = 120  # Reduced from 200 for faster rendering
SPECTRUM_IMG_SIZE = 800  # Reduced from 1500 for faster generation

# Library and data paths
LIBRARY_PATH = Path('DreaMS/data/MassSpecGym_DreaMS.hdf5')
DATA_PATH = Path('./DreaMS/data')
EXAMPLE_PATH = Path('./data')

# PFAS model configuration
PFAS_MODEL_PATH = Path('/Users/ramsindhu/Downloads/HalogenDetection-FocalLoss-MergedMassSpecNIST20_NISTNew_NormalPFAS_ujmvyfxm_checkpoints_epoch=0-step=9285.ckpt')
PFAS_THRESHOLD = 0.90
N_HIGHEST_PEAKS = 60

# Load model with weights_only=False for PyTorch 2.6+ compatibility
# Safe because this is a trusted checkpoint
# Use map_location='cpu' to handle models saved on CUDA devices
try:
    # Try with safe_globals context manager (PyTorch 2.6+)
    with torch.serialization.safe_globals([getattr]):
        MODEL = HalogenDetectorDreamsTest.load_from_checkpoint(PFAS_MODEL_PATH, map_location='cpu')
except Exception as e:
    # Fallback: patch torch.load temporarily
    import functools
    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    MODEL = HalogenDetectorDreamsTest.load_from_checkpoint(PFAS_MODEL_PATH, map_location='cpu')
    torch.load = original_load

# Cache for SMILES images to avoid regeneration
_smiles_cache = {}

def clear_smiles_cache():
    """Clear the SMILES image cache to free memory"""
    global _smiles_cache
    _smiles_cache.clear()
    print("SMILES image cache cleared")

# =============================================================================
# UTILITY FUNCTIONS FOR IMAGE CONVERSION
# =============================================================================

def _validate_input_file(file_path):
    """
    Validate that the input file exists and has a supported format
    
    Args:
        file_path: Path to the input file
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file_path or not Path(file_path).exists():
        return False
    
    supported_extensions = ['.mgf', '.mzML', '.mzml']
    file_ext = Path(file_path).suffix.lower()
    
    return file_ext in supported_extensions


def _convert_pil_to_base64(img, format='PNG'):
    """
    Convert a PIL Image to base64 encoded string
    
    Args:
        img: PIL Image object
        format: Image format (default: 'PNG')
    
    Returns:
        str: Base64 encoded image string
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format, optimize=True)  # Added optimize=True
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/{format.lower()};base64,{repr(img_str)[2:-1]}"


def _crop_transparent_edges(img):
    """
    Crop transparent edges from a PIL Image
    
    Args:
        img: PIL Image object (should be RGBA)
    
    Returns:
        PIL Image: Cropped image
    """
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get the bounding box of non-transparent pixels
    bbox = img.getbbox()
    if bbox:
        # Crop the image to remove transparent space
        img = img.crop(bbox)
    
    return img


def smiles_to_html_img(smiles, img_size=SMILES_IMG_SIZE):
    """
    Convert SMILES string to HTML image for display in Gradio dataframe
    Uses caching to avoid regenerating the same molecule images
    
    Args:
        smiles: SMILES string representation of molecule
        img_size: Size of the output image (default: SMILES_IMG_SIZE)
    
    Returns:
        str: HTML img tag with base64 encoded image
    """
    # Check cache first
    cache_key = f"{smiles}_{img_size}"
    if cache_key in _smiles_cache:
        return _smiles_cache[cache_key]
    
    try:
        # Parse SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result = f"<div style='text-align: center; color: red;'>Invalid SMILES</div>"
            _smiles_cache[cache_key] = result
            return result
        
        # Create PNG drawing with Cairo backend for better control
        d2d = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
        opts = d2d.drawOptions()
        opts.clearBackground = False
        opts.padding = 0.05  # Minimal padding
        opts.bondLineWidth = 1.5  # Reduced from 2.0 for smaller images
        
        # Draw the molecule
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        
        # Get PNG data and convert to PIL Image
        png_data = d2d.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        
        # Crop transparent edges and convert to base64
        img = _crop_transparent_edges(img)
        img_str = _convert_pil_to_base64(img)
        
        result = f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='{smiles}' />"
        
        # Cache the result
        _smiles_cache[cache_key] = result
        return result
        
    except Exception as e:
        result = f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"
        _smiles_cache[cache_key] = result
        return result


def spectrum_to_html_img(spec1, spec2, img_size=SPECTRUM_IMG_SIZE):
    """
    Convert spectrum plot to HTML image for display in Gradio dataframe
    Optimized version based on working code

    Args:
        spec1: First spectrum data
        spec2: Second spectrum data (for mirror plot)
        img_size: Size of the output image (default: SPECTRUM_IMG_SIZE)

    Returns:
        str: HTML img tag with base64 encoded spectrum plot
    """
    try:
        # Use non-interactive matplotlib backend
        matplotlib.use('Agg')

        # Create the spectrum plot using DreaMS utility function
        su.plot_spectrum(spec=spec1, mirror_spec=spec2, figsize=(1.6, 0.8))  # Reduced size for performance

        # Save figure to buffer with transparent background
        buffered = BytesIO()
        plt.savefig(buffered, format='png', bbox_inches='tight', dpi=80, transparent=True)
        buffered.seek(0)

        # Convert to PIL Image, crop edges, and convert to base64
        img = Image.open(buffered)
        img = _crop_transparent_edges(img)
        img_str = _convert_pil_to_base64(img)

        # Clean up matplotlib figure to free memory
        plt.close()

        return f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='Spectrum comparison' />"

    except Exception as e:
        return f"<div style='text-align: center; color: red;'>Error: {str(e)}</div>"


def pfas_to_html(pfas_prob, threshold=PFAS_THRESHOLD):
    """
    Convert PFAS probability to HTML with visual styling

    Args:
        pfas_prob: PFAS probability value (0-1)
        threshold: Threshold for highlighting (default: PFAS_THRESHOLD)

    Returns:
        str: HTML formatted PFAS prediction with styling
    """
    try:
        prob_pct = pfas_prob * 100

        if pfas_prob >= threshold:
            # Bold text with PFAS badge for values above threshold
            return f"<span style='font-weight: bold; font-size: 14px;'>{prob_pct:.1f}%</span><br/><span style='background-color: #ff6b6b; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold;'>PFAS</span>"
        else:
            # Regular text for values below threshold
            return f"{prob_pct:.1f}%"

    except Exception as e:
        return f">Error: {str(e)}"


def check_mass_defect(precursor_mz):
    """
    Check if first decimal of precursor m/z matches PFAS pattern (0.6, 0.7, 0.8, or 0.9)

    Args:
        precursor_mz: Precursor m/z value

    Returns:
        str: HTML formatted mass defect check result
    """
    try:
        # Get the first decimal digit
        # e.g., 413.9787 -> 9, 312.6543 -> 6
        first_decimal = int(str(precursor_mz).split('.')[1][0]) if '.' in str(precursor_mz) else 0

        if first_decimal in [6, 7, 8, 9]:
            # PFAS-like mass defect pattern
            return f"<span style='background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold;'>✓ .{first_decimal}</span>"
        else:
            # Not a typical PFAS pattern
            return f"<span style='color: #999;'>.{first_decimal}</span>"

    except Exception as e:
        return f"<span style='color: red;'>Error</span>"


# =============================================================================
# DATA DOWNLOAD AND SETUP FUNCTIONS
# =============================================================================

def _download_file(url, target_path, description):
    """
    Download a file from URL if it doesn't exist
    
    Args:
        url: Source URL
        target_path: Target file path
        description: Description for logging
    """
    if not target_path.exists():
        print(f"Downloading {description}...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target_path)
        print(f"Downloaded {description} to {target_path}")


def setup():
    """
    Initialize the application by downloading required data files

    Downloads:
    - Example MS/MS files for testing

    Raises:
        Exception: If critical setup steps fail
    """
    print("=" * 60)
    print("Setting up DreaMS-PFAS screening application...")
    print("=" * 60)
    
    try:
        # Download example files
        example_urls = [
            ('https://raw.githubusercontent.com/pluskal-lab/DreaMS/refs/heads/main/data/examples/example_5_spectra.mgf',
             EXAMPLE_PATH / 'example_5_spectra.mgf',
             "DreaMS example spectra")
        ]

        for url, path, desc in example_urls:
            _download_file(url, path, desc)

        print(f"\n✓ Setup complete - PFAS screening tool ready")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Setup failed: {e}")
        print("The application may not work properly. Please check your internet connection and try again.")
        raise


# =============================================================================
# CORE PREDICTION FUNCTIONS
# =============================================================================

@spaces.GPU
def _predict_gpu(in_pth, progress):
    """
    GPU-accelerated prediction of DreaMS embeddings

    Args:
        in_pth: Input file path
        progress: Gradio progress tracker

    Returns:
        numpy.ndarray: DreaMS embeddings
    """
    progress(0.2, desc="Loading spectra data...")
    msdata = MSData.load(in_pth)

    progress(0.3, desc="Computing DreaMS embeddings...")
    embs = dreams_embeddings(msdata)
    print(f'Shape of the query embeddings: {embs.shape}')

    return embs


@spaces.GPU
def _predict_pfas_gpu(in_pth, progress):
    """
    GPU-accelerated prediction of PFAS probabilities

    Args:
        in_pth: Input file path
        progress: Gradio progress tracker

    Returns:
        numpy.ndarray: PFAS probabilities
    """
    progress(0.4, desc="Computing PFAS predictions...")
    # Use different loading method for mzML files
    if str(in_pth).lower().endswith('.mzml'):
        try:
            pfas_preds = _find_PFAS_mzML(in_pth)
        except ValueError as e:
            print(f'Error loading mzML file: {e}')
            raise
    else:
        msdata = MSData.load(in_pth)
        pfas_preds = dreams_predictions(
            spectra=msdata,
            model_ckpt=MODEL,
            n_highest_peaks=N_HIGHEST_PEAKS
        )

    pfas_preds = torch.sigmoid(torch.from_numpy(pfas_preds)).cpu().numpy()
    print(f'Shape of PFAS predictions: {pfas_preds.shape}')

    return pfas_preds

def _find_PFAS_mzML(in_pth):
    # in_pth = 'data/teo/abc.mzML
    # in_pth = Path('/teamspace/studios/this_studio/SLI23_040.mzML')

    n_highest_peaks = 60

    print(f'Processing {in_pth}...')

    # Load data
    try:
        msdata = MSData.from_mzml(in_pth, verbose_parser=True)
    except ValueError as e:
        print(f'Skipping {in_pth} because of {e}.')
        return

    # Get spectra (m/z and inetsnity arrays) and precursor m/z values from the input dataset
    spectra = msdata['spectrum']
    prec_mzs = msdata['precursor_mz']

    # Ref: https://dreams-docs.readthedocs.io/en/latest/tutorials/spectral_quality.html
    # Subject each spectrum to spectral quality checks
    dformat = DataFormatA()
    quality_lvls = [dformat.val_spec(s, p, return_problems=True) for s, p in zip(spectra, prec_mzs)]

    # Check how many spectra passed all filters (`All checks passed`) and how many spectra did not pass some of the filters
    print(pd.Series(quality_lvls).value_counts())

    # Define path for output high-quality file
    hq_pth = append_to_stem(in_pth, 'high_quality').with_suffix('.hdf5')

    # Pick only high-quality spectra and save them to `hq_pth`
    msdata.form_subset(
        idx=np.where(np.array(quality_lvls) == 'All checks passed')[0],
        out_pth=hq_pth
    )

    # Try reading the new file
    msdata_hq = MSData.load(hq_pth)

    # Compute PFAS logits predictions
    f_preds = dreams_predictions(
        spectra=msdata_hq,
        model_ckpt=MODEL,
        n_highest_peaks=n_highest_peaks
    )

    return f_preds

def _create_result_row(i, msdata, pfas_preds):
    """
    Create a single result row for the DataFrame (PFAS screening only)

    Args:
        i: Query spectrum index
        msdata: Query MS data
        pfas_preds: PFAS predictions

    Returns:
        dict: Result row data
    """
    pfas_prob = pfas_preds[i]
    precursor_mz = msdata.get_prec_mzs(i)

    # Create row data
    row_data = {
        'scan_number': msdata.get_values(SCAN_NUMBER, i) if SCAN_NUMBER in msdata.columns() else None,
        'rt': msdata.get_values(RT, i) if RT in msdata.columns() else None,
        'charge': msdata.get_values(CHARGE, i) if CHARGE in msdata.columns() else None,
        'precursor_mz': precursor_mz,
        'precursor_mz_raw': precursor_mz,
        'PFAS_prediction': pfas_to_html(pfas_prob),
        'PFAS_probability_raw': pfas_prob,
        'mass_defect_check': check_mass_defect(precursor_mz),
    }

    return row_data


def _process_results_dataframe(df, in_pth):
    """
    Process and clean the results DataFrame (PFAS screening only)

    Args:
        df: Raw results DataFrame
        in_pth: Input file path for CSV export

    Returns:
        tuple: (processed_df, csv_path)
    """
    # Round numerical values
    df['PFAS_probability_raw'] = df['PFAS_probability_raw'].astype(float).round(4)
    df['precursor_mz_raw'] = df['precursor_mz_raw'].astype(float).round(4)

    # Handle optional columns
    if 'rt' in df.columns:
        df['rt'] = df['rt'].astype(float).round(2)
    if 'charge' in df.columns:
        df['charge'] = df['charge'].astype(str)

    # Add mass defect first decimal column for filtering
    df['mass_defect_first_decimal'] = df['precursor_mz_raw'].apply(
        lambda x: int(str(x).split('.')[1][0]) if '.' in str(x) else 0
    )

    # Rename columns for display
    column_mapping = {
        "scan_number": "Scan number",
        "rt": "Retention time",
        "charge": "Charge",
        "precursor_mz": "Precursor m/z",
        "precursor_mz_raw": "Precursor m/z (raw)",
        "PFAS_prediction": "PFAS Prediction",
        "PFAS_probability_raw": "PFAS Probability",
        "mass_defect_check": "Mass Defect",
    }

    df = df.rename(columns=column_mapping)

    # Save full results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_path = dio.append_to_stem(in_pth, f"PFAS_screening_{timestamp}").with_suffix('.csv')
    df_to_save = df.drop(columns=['PFAS Prediction', 'Mass Defect', 'mass_defect_first_decimal'])
    df_to_save.to_csv(df_path, index=False)

    # Filter: Show entries with PFAS probability >= 0.95
    df = df[(df['PFAS Probability'] >= PFAS_THRESHOLD)]

    # Sort by PFAS probability (descending) before dropping the raw column
    df = df.sort_values('PFAS Probability', ascending=False)

    # Prepare final display DataFrame
    df = df.drop(columns=['PFAS Probability', 'Precursor m/z (raw)', 'mass_defect_first_decimal'])

    # Add row numbers
    df.insert(0, 'Row', range(1, len(df) + 1))

    return df, str(df_path)


def _predict_core(lib_pth, in_pth, similarity_threshold, calculate_modified_cosine, progress):
    """
    Core prediction function for PFAS detection (library matching disabled)

    Args:
        lib_pth: Library file path (not used, kept for compatibility)
        in_pth: Input file path
        similarity_threshold: Not used (kept for compatibility)
        calculate_modified_cosine: Not used (kept for compatibility)
        progress: Gradio progress tracker

    Returns:
        tuple: (results_dataframe, csv_file_path)
    """
    in_pth = Path(in_pth)

    # Create temporary copy of input file
    progress(0, desc="Creating temporary file copy...")
    temp_in_path = in_pth.parent / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{in_pth.name}"
    shutil.copy2(in_pth, temp_in_path)

    try:
        # Get PFAS predictions
        pfas_preds = _predict_pfas_gpu(temp_in_path, progress)

        # Load query data for processing
        progress(0.5, desc="Loading spectra data...")
        msdata = MSData.load(temp_in_path, in_mem=True)
        print(f'Available columns: {msdata.columns()}')

        # Construct results DataFrame
        progress(0.6, desc="Constructing results table...")
        df = []
        total_spectra = len(pfas_preds)

        for i in range(total_spectra):
            progress(0.6 + 0.3 * (i / total_spectra),
                    desc=f"Processing spectrum {i+1}/{total_spectra}...")

            row_data = _create_result_row(i, msdata, pfas_preds)
            df.append(row_data)

        df = pd.DataFrame(df)

        # Process and clean results
        progress(0.9, desc="Post-processing results...")
        df, csv_path = _process_results_dataframe(df, in_pth)

        progress(1.0, desc=f"PFAS screening complete! Analyzed {len(df)} spectra.")

        return df, csv_path

    finally:
        # Clean up temporary files
        if temp_in_path.exists():
            temp_in_path.unlink()


def predict(lib_pth, in_pth, progress=gr.Progress(track_tqdm=True)):
    """
    Main PFAS screening function with error handling

    Args:
        lib_pth: Library file path (not used, kept for compatibility)
        in_pth: Input file path
        progress: Gradio progress tracker

    Returns:
        tuple: (results_dataframe, csv_file_path)

    Raises:
        gr.Error: If prediction fails or input is invalid
    """
    try:
        # Validate input file
        if not _validate_input_file(in_pth):
            raise gr.Error("Invalid input file. Please provide a valid .mgf or .mzML file.")

        df, csv_path = _predict_core(lib_pth, in_pth, None, None, progress)

        return df, csv_path

    except gr.Error:
        # Re-raise Gradio errors as-is
        raise
    except Exception as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg:
            error_msg = f"GPU/CUDA error: {error_msg}. The app is falling back to CPU mode."
        elif "RuntimeError" in error_msg:
            error_msg = f"Runtime error: {error_msg}. This may be due to memory or device issues."
        else:
            error_msg = f"Error: {error_msg}"

        print(f"PFAS screening failed: {error_msg}")
        raise gr.Error(error_msg)


# =============================================================================
# GRADIO INTERFACE SETUP
# =============================================================================

def _create_gradio_interface():
    """
    Create and configure the Gradio interface
    
    Returns:
        gr.Blocks: Configured Gradio app
    """
    # JavaScript for theme management
    js_func = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
    """
    
    # Create app with custom theme
    app = gr.Blocks(
        theme=gr.themes.Default(primary_hue="yellow", secondary_hue="pink"), 
        js=js_func
    )
    
    with app:
        # Header and description
        #gr.Image("https://raw.githubusercontent.com/pluskal-lab/DreaMS/cc806fa6fea281c1e57dd81fc512f71de9290017/assets/dreams_background.png", 
        #        label="DreaMS")
        gr.Image("./DreaMS-PFAS.png", label="DreaMS-PFAS Screening Tool")
        
        gr.Markdown(value="""
            **DreaMS-PFAS Screening Tool** - This tool uses a DreaMS-based model to predict the probability of MS/MS spectra being PFAS (Per- and Polyfluoroalkyl Substances).
            Upload your MS/MS file (.mgf or .mzML) to screen for potential PFAS compounds. The tool provides PFAS probability predictions and checks for characteristic PFAS mass defect patterns (first decimal of m/z = 0.6, 0.7, 0.8, or 0.9).
        """)
        
        # Input section
        with gr.Row(equal_height=True):
            in_pth = gr.File(
                file_count="single",
                label="Input MS/MS file (.mgf or .mzML)",
            )
        
        # Example files
        examples = gr.Examples(
            examples=["./data/example_5_spectra.mgf"],
            inputs=[in_pth],
            label="Examples (click on a file to load as input)",
        )

        # Prediction button
        predict_button = gr.Button(value="Run PFAS Screening", variant="primary")
        
        # Results table
        gr.Markdown("## Predictions")
        df_file = gr.File(label="Download predictions as .csv", interactive=False, visible=True)
        
        # Results table
        headers = ["Row", "Scan number", "Retention time", "Charge", "Precursor m/z", "PFAS Prediction", "Mass Defect"]
        datatype = ["number", "number", "number", "str", "number", "html", "html"]
        column_widths = ["50px", "80px", "100px", "60px", "100px", "120px", "100px"]

        df = gr.Dataframe(
            headers=headers,
            datatype=datatype,
            col_count=(len(headers), "fixed"),
            column_widths=column_widths,
            max_height=1000,
            interactive=False,
        )

        # Connect prediction logic
        inputs = [in_pth]
        outputs = [df, df_file]

        predict_func = partial(predict, LIBRARY_PATH)
        predict_button.click(predict_func, inputs=inputs, outputs=outputs, show_progress="full")
    
    return app


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize the application
    setup()
    
    # Create and launch the Gradio interface
    app = _create_gradio_interface()
    app.launch(allowed_paths=['./assets'], share=True)
else:
    # When imported as a module, just run setup
    setup()
