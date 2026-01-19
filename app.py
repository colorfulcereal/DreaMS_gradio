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
import time
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


def spectrum_preview_to_html(spectrum, precursor_mz=None, width=200, height=80):
    """
    Create a small inline preview of a mass spectrum for table display

    Args:
        spectrum: Spectrum data (m/z and intensity arrays)
        precursor_mz: Precursor m/z value (optional, will be marked on plot)
        width: Width of preview image in pixels
        height: Height of preview image in pixels

    Returns:
        str: HTML img tag with base64 encoded spectrum preview
    """
    try:
        matplotlib.use('Agg')

        # Create a small figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        # Get spectrum data
        mz_array = spectrum[0]
        intensity_array = spectrum[1]

        # Normalize intensity
        max_intensity = np.max(intensity_array)
        if max_intensity > 0:
            intensity_array = (intensity_array / max_intensity) * 100

        # Create stem plot with minimal styling
        markerline, stemlines, baseline = ax.stem(mz_array, intensity_array,
                                                    linefmt='steelblue',
                                                    markerfmt=' ',
                                                    basefmt=' ')
        stemlines.set_linewidth(0.8)

        # Mark precursor if provided
        if precursor_mz is not None:
            ax.axvline(x=precursor_mz, color='red', linestyle='--',
                      linewidth=1, alpha=0.6)

        # Remove labels and ticks for compact display
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([np.min(mz_array) - 10, np.max(mz_array) + 10])
        ax.set_ylim([0, 110])

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Tight layout
        plt.tight_layout(pad=0)

        # Save to buffer
        buffered = BytesIO()
        plt.savefig(buffered, format='png', bbox_inches='tight',
                   dpi=100, transparent=True, pad_inches=0)
        buffered.seek(0)

        # Convert to base64
        img = Image.open(buffered)
        img = _crop_transparent_edges(img)
        img_str = _convert_pil_to_base64(img)

        # Clean up
        plt.close()

        return f"<img src='{img_str}' style='max-width: 100%; height: auto;' title='Spectrum preview' />"

    except Exception as e:
        return f"<div style='text-align: center; color: red; font-size: 10px;'>Error</div>"


def create_spectrum_plot(spectrum, precursor_mz=None, title="Mass Spectrum"):
    """
    Create an interactive mass spectrum plot

    Args:
        spectrum: Spectrum data (m/z and intensity arrays)
        precursor_mz: Precursor m/z value (optional, will be marked on plot)
        title: Plot title

    Returns:
        matplotlib.figure.Figure: The spectrum plot figure
    """
    try:
        matplotlib.use('Agg')

        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot spectrum as stem plot
        mz_array = spectrum[0]
        intensity_array = spectrum[1]

        # Normalize intensity to 100
        max_intensity = np.max(intensity_array)
        if max_intensity > 0:
            intensity_array = (intensity_array / max_intensity) * 100

        # Create stem plot
        markerline, stemlines, baseline = ax.stem(mz_array, intensity_array,
                                                    linefmt='blue',
                                                    markerfmt=' ',
                                                    basefmt=' ')
        stemlines.set_linewidth(1.5)

        # Mark precursor if provided
        if precursor_mz is not None:
            ax.axvline(x=precursor_mz, color='red', linestyle='--',
                      linewidth=2, label=f'Precursor m/z: {precursor_mz:.4f}', alpha=0.7)
            ax.legend()

        # Labels and formatting
        ax.set_xlabel('m/z', fontsize=12)
        ax.set_ylabel('Relative Intensity (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 110])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    except Exception as e:
        # Create error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error creating plot: {str(e)}',
                ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        return fig


def extract_metadata(msdata, scan_idx=0):
    """
    Extract metadata from MS data

    Args:
        msdata: MSData object
        scan_idx: Index of the scan to extract metadata from

    Returns:
        dict: Dictionary containing metadata
    """
    metadata = {}

    try:
        # Available columns
        available_cols = msdata.columns()
        metadata['Available Fields'] = ', '.join(available_cols)

        # Extract common fields if available
        if SCAN_NUMBER in available_cols:
            metadata['Scan Number'] = msdata.get_values(SCAN_NUMBER, scan_idx)

        if RT in available_cols:
            rt = msdata.get_values(RT, scan_idx)
            metadata['Retention Time'] = f"{rt:.2f} seconds" if rt is not None else "N/A"

        if CHARGE in available_cols:
            metadata['Charge'] = msdata.get_values(CHARGE, scan_idx)

        if PRECURSOR_MZ in available_cols:
            prec_mz = msdata.get_prec_mzs(scan_idx)
            metadata['Precursor m/z'] = f"{prec_mz:.4f}" if prec_mz is not None else "N/A"

        # Spectrum info
        if 'spectrum' in available_cols:
            spectrum = msdata['spectrum'][scan_idx]
            if spectrum is not None and len(spectrum) == 2:
                metadata['Number of Peaks'] = len(spectrum[0])
                metadata['m/z Range'] = f"{np.min(spectrum[0]):.2f} - {np.max(spectrum[0]):.2f}"
                metadata['Max Intensity'] = f"{np.max(spectrum[1]):.2e}"

        # Dataset info
        metadata['Total Spectra in File'] = len(msdata)

    except Exception as e:
        metadata['Error'] = str(e)

    return metadata


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

    # Get spectrum for preview
    spectrum = msdata['spectrum'][i]
    spectrum_preview = spectrum_preview_to_html(spectrum, precursor_mz, width=200, height=80)

    # Create row data
    row_data = {
        'scan_number': msdata.get_values(SCAN_NUMBER, i) if SCAN_NUMBER in msdata.columns() else None,
        'rt': msdata.get_values(RT, i) if RT in msdata.columns() else None,
        'charge': msdata.get_values(CHARGE, i) if CHARGE in msdata.columns() else None,
        'precursor_mz': precursor_mz,
        'precursor_mz_raw': precursor_mz,
        'spectrum_preview': spectrum_preview,
        'PFAS_prediction': pfas_to_html(pfas_prob),
        'PFAS_probability_raw': pfas_prob,
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
        "spectrum_preview": "Spectrum",
        "PFAS_prediction": "PFAS Prediction",
        "PFAS_probability_raw": "PFAS Probability",
    }

    df = df.rename(columns=column_mapping)

    # Save full results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_path = dio.append_to_stem(in_pth, f"PFAS_screening_{timestamp}").with_suffix('.csv')
    df_to_save = df.drop(columns=['PFAS Prediction', 'Spectrum', 'mass_defect_first_decimal'])
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
        # Get PFAS predictions with timing
        start_time = time.time()
        pfas_preds = _predict_pfas_gpu(temp_in_path, progress)
        end_time = time.time()

        total_model_time = end_time - start_time
        num_spectra = len(pfas_preds)

        # Load query data for processing
        progress(0.5, desc="Loading spectra data...")
        msdata = MSData.load(temp_in_path, in_mem=True)
        print(f'Available columns: {msdata.columns()}')

        # Construct results DataFrame
        progress(0.6, desc="Constructing results table...")
        df = []

        for i in range(num_spectra):
            progress(0.6 + 0.3 * (i / num_spectra),
                    desc=f"Processing spectrum {i+1}/{num_spectra}...")

            row_data = _create_result_row(i, msdata, pfas_preds)
            df.append(row_data)

        df = pd.DataFrame(df)

        # Calculate timing statistics
        avg_time_per_spectrum = total_model_time / num_spectra

        timing_stats = {
            'total_model_time_seconds': total_model_time,
            'total_model_time_ms': total_model_time * 1000,
            'avg_time_per_spectrum_seconds': avg_time_per_spectrum,
            'avg_time_per_spectrum_ms': avg_time_per_spectrum * 1000,
            'total_spectra': num_spectra
        }

        # Process and clean results
        progress(0.9, desc="Post-processing results...")
        df, csv_path = _process_results_dataframe(df, in_pth)

        progress(1.0, desc=f"PFAS screening complete! Analyzed {len(df)} spectra.")

        return df, csv_path, timing_stats

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
        tuple: (results_dataframe, csv_file_path, input_file_path, timing_info_html)

    Raises:
        gr.Error: If prediction fails or input is invalid
    """
    try:
        # Validate input file
        if not _validate_input_file(in_pth):
            raise gr.Error("Invalid input file. Please provide a valid .mgf or .mzML file.")

        df, csv_path, timing_stats = _predict_core(lib_pth, in_pth, None, None, progress)

        # Format timing statistics as HTML with better number formatting
        total_spectra = timing_stats['total_spectra']
        total_time_s = timing_stats['total_model_time_seconds']
        avg_time_ms = timing_stats['avg_time_per_spectrum_ms']

        # Format total time intelligently (use seconds for > 1s, else milliseconds)
        if total_time_s >= 1.0:
            if total_time_s >= 60:
                minutes = int(total_time_s // 60)
                seconds = total_time_s % 60
                total_time_str = f"{minutes}m {seconds:.2f}s"
            else:
                total_time_str = f"{total_time_s:.3f} seconds"
        else:
            total_time_str = f"{timing_stats['total_model_time_ms']:.1f} ms"

        # Format average time (use ms for < 1000ms, else seconds)
        if avg_time_ms >= 1000:
            avg_time_str = f"{timing_stats['avg_time_per_spectrum_seconds']:.3f} seconds"
        else:
            avg_time_str = f"{avg_time_ms:.2f} ms"

        timing_html = f"""
        <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin: 10px 0;'>
            <h3 style='margin-top: 0;'>Model Performance Statistics</h3>
            <p><strong>Total Spectra Processed:</strong> {total_spectra:,}</p>
            <p><strong>Total Model Inference Time:</strong> {total_time_str}</p>
            <p><strong>Average Time per Spectrum:</strong> {avg_time_str}</p>
        </div>
        """

        return df, csv_path, in_pth, timing_html

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


# Global variable to store the loaded MS data for visualization
_loaded_msdata = None
_loaded_file_path = None


def visualize_spectrum(in_pth, scan_number):
    """
    Visualize a spectrum and extract its metadata

    Args:
        in_pth: Input file path
        scan_number: Scan number to visualize

    Returns:
        tuple: (plot_figure, metadata_json)
    """
    global _loaded_msdata, _loaded_file_path

    try:
        # Load data if not already loaded or if file changed
        if _loaded_msdata is None or _loaded_file_path != in_pth:
            if in_pth is None:
                return None, {"Error": "No file loaded. Please run PFAS screening first."}

            _loaded_file_path = in_pth
            _loaded_msdata = MSData.load(in_pth, in_mem=True)

        # Find the index corresponding to the scan number
        available_cols = _loaded_msdata.columns()
        if SCAN_NUMBER in available_cols:
            scan_numbers = [_loaded_msdata.get_values(SCAN_NUMBER, i) for i in range(len(_loaded_msdata))]
            try:
                scan_idx = scan_numbers.index(scan_number)
            except ValueError:
                return None, {"Error": f"Scan number {scan_number} not found in file"}
        else:
            # If no scan numbers, use direct indexing
            scan_idx = int(scan_number) - 1 if scan_number > 0 else 0
            if scan_idx >= len(_loaded_msdata):
                return None, {"Error": f"Index {scan_idx} out of range"}

        # Get spectrum and metadata
        spectrum = _loaded_msdata['spectrum'][scan_idx]
        precursor_mz = _loaded_msdata.get_prec_mzs(scan_idx) if PRECURSOR_MZ in available_cols else None

        # Create plot
        title = f"Mass Spectrum - Scan {scan_number}"
        if precursor_mz:
            title += f" (Precursor m/z: {precursor_mz:.4f})"

        fig = create_spectrum_plot(spectrum, precursor_mz, title)

        # Extract metadata
        metadata = extract_metadata(_loaded_msdata, scan_idx)

        return fig, metadata

    except Exception as e:
        return None, {"Error": str(e)}


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
            Upload your MS/MS file (.mgf or .mzML) to screen for potential PFAS compounds.
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

        # Timing statistics display
        timing_display = gr.HTML(label="Processing Time Statistics", visible=True)

        # Results table
        headers = ["Row", "Scan number", "Retention time", "Charge", "Precursor m/z", "Spectrum", "PFAS Prediction"]
        datatype = ["number", "number", "number", "str", "number", "html", "html"]
        column_widths = ["50px", "80px", "100px", "60px", "100px", "200px", "120px"]

        df = gr.Dataframe(
            headers=headers,
            datatype=datatype,
            col_count=(len(headers), "fixed"),
            column_widths=column_widths,
            max_height=1000,
            interactive=False,
        )

        # Spectrum Visualization Section
        gr.Markdown("## Spectrum Visualization & Metadata")
        gr.Markdown("Enter a scan number from the results table above to visualize the spectrum and view detailed metadata.")

        with gr.Row():
            with gr.Column(scale=1):
                scan_input = gr.Number(
                    label="Scan Number",
                    value=1,
                    precision=0,
                    info="Enter the scan number to visualize"
                )
                visualize_button = gr.Button(value="Visualize Spectrum", variant="secondary")

            with gr.Column(scale=2):
                metadata_output = gr.JSON(
                    label="Spectrum Metadata",
                    container=True
                )

        spectrum_plot = gr.Plot(
            label="Mass Spectrum",
            container=True
        )

        # Hidden state to store input file path
        file_state = gr.State()

        # Connect prediction logic
        inputs = [in_pth]
        outputs = [df, df_file, file_state, timing_display]

        predict_func = partial(predict, LIBRARY_PATH)
        predict_button.click(predict_func, inputs=inputs, outputs=outputs, show_progress="full")

        # Connect visualization logic
        visualize_button.click(
            visualize_spectrum,
            inputs=[file_state, scan_input],
            outputs=[spectrum_plot, metadata_output],
            show_progress="minimal"
        )
    
    return app


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize the application
    setup()
    
    # Create and launch the Gradio interface
    app = _create_gradio_interface()
    app.launch(allowed_paths=['./assets'], share=False)
else:
    # When imported as a module, just run setup
    setup()
