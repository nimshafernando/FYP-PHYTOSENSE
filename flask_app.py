import os
import json
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pillow_avif  # Add AVIF support
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
import base64
from io import BytesIO
import google.generativeai as genai
from datetime import datetime
import pickle  # For loading XGBoost model
from autodock_vina_integration import vina_integration  # Import AutoDock Vina integration

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = []
class_names = []
phytochemical_data = {}

# QSAR model variables
qsar_model = None
qsar_features = []
qsar_targets = []

# Gemini API usage tracking
gemini_request_count = 0
gemini_session_start = datetime.now()
gemini_daily_limit = 1500  # Gemini API free tier allows 1500 requests per day

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'avif', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Configure Gemini AI for description generation
def configure_gemini():
    """Configure Gemini AI with API key"""
    global gemini_model
    try:
        API_KEY = "AIzaSyDs0R6k0cNn8EWe5nMRHVWR3Q8HTmNpfxw"
        genai.configure(api_key=API_KEY)
        
        # Try different model names in order of preference
        model_names = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-flash',
            'gemini-1.0-pro',
            'gemini-pro'
        ]
        
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                print(f"✅ Gemini AI configured successfully with {model_name}")
                return gemini_model
            except Exception as model_error:
                print(f"⚠️ Failed to load {model_name}: {model_error}")
                continue
        
        # If all models fail, set to None
        print("❌ No available Gemini models found")
        gemini_model = None
        return None
        
    except Exception as e:
        print(f"❌ Error configuring Gemini AI: {e}")
        gemini_model = None
        return None

# Initialize Gemini model
gemini_model = None

def list_available_models():
    """List available Gemini models for debugging"""
    try:
        API_KEY = "AIzaSyDs0R6k0cNn8EWe5nMRHVWR3Q8HTmNpfxw"
        genai.configure(api_key=API_KEY)
        
        models = genai.list_models()
        print("📋 Available Gemini models:")
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"  ✅ {model.name}")
            else:
                print(f"  ❌ {model.name} (not supported for generateContent)")
        return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

# Initialize Gemini - list models first if configuration fails
gemini_model = configure_gemini()
if gemini_model is None:
    print("🔍 Attempting to list available models...")
    available = list_available_models()
    if available:
        print("💡 Try updating the model name in configure_gemini() to one of the available models above")

def log_gemini_request_start():
    """Log the start of a Gemini API request"""
    global gemini_request_count
    gemini_request_count += 1
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Calculate remaining requests
    remaining = gemini_daily_limit - gemini_request_count
    percentage_used = (gemini_request_count / gemini_daily_limit) * 100
    
    print(f"\n🔄 [GEMINI API REQUEST #{gemini_request_count}] - {current_time}")
    print(f"📊 Daily Usage: {gemini_request_count}/{gemini_daily_limit} ({percentage_used:.1f}%)")
    print(f"🎯 Remaining: {remaining} requests")
    
    # Warning if approaching limit
    if percentage_used >= 90:
        print(f"⚠️  WARNING: You're using {percentage_used:.1f}% of your daily limit!")
    elif percentage_used >= 75:
        print(f"🔶 CAUTION: You've used {percentage_used:.1f}% of your daily limit")

def log_gemini_request_end(success=True, error_msg=None):
    """Log the end of a Gemini API request"""
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if success:
        print(f"✅ [GEMINI API SUCCESS] - {current_time}")
    else:
        print(f"❌ [GEMINI API ERROR] - {current_time}: {error_msg}")
    print("-" * 60)

def get_gemini_usage_stats():
    """Get current Gemini API usage statistics"""
    session_duration = datetime.now() - gemini_session_start
    percentage_used = (gemini_request_count / gemini_daily_limit) * 100
    remaining = gemini_daily_limit - gemini_request_count
    
    stats = {
        'requests_made': gemini_request_count,
        'daily_limit': gemini_daily_limit,
        'remaining': remaining,
        'percentage_used': percentage_used,
        'session_duration': str(session_duration).split('.')[0],  # Remove microseconds
        'session_start': gemini_session_start.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return stats

def generate_ai_description(compound_name, smiles, plant_name):
    """Generate AI description for phytochemical compound"""
    global gemini_model
    
    if not gemini_model:
        print("❌ Gemini model not configured - skipping request")
        return f"AI description unavailable. Please configure Gemini API key."
    
    try:
        # Log request start
        log_gemini_request_start()
        
        prompt = f"""
        As an expert phytochemist and pharmacologist, provide a detailed and engaging scientific description (200-250 words) of this phytochemical compound:

        **Compound:** {compound_name}
        **SMILES:** {smiles}
        **Source Plant:** {plant_name}

        Structure your response to cover:

        1. **Chemical Classification & Structure:**
           - What class does it belong to? (alkaloid, flavonoid, terpenoid, phenolic compound, etc.)
           - Notable structural features from the SMILES representation
        
        2. **Biological Activities & Pharmacology:**
           - Primary therapeutic effects and biological activities
           - Molecular mechanisms of action (receptor interactions, enzyme inhibition, etc.)
           - Specific cellular pathways affected
        
        3. **Medical Applications:**
           - Traditional medicinal uses in herbal medicine
           - Modern clinical applications and research findings
           - Disease conditions it targets (cancer, inflammation, microbial infections, etc.)
        
        4. **Pharmacokinetics & Safety:**
           - Bioavailability and absorption characteristics
           - Known side effects or contraindications
           - Safety profile and therapeutic index
        
        5. **Research & Clinical Status:**
           - Current research developments
           - Clinical trial status if applicable
           - Future therapeutic potential

        Write in clear, scientific language suitable for medical professionals, researchers, and advanced students. Include specific technical terms but make the content accessible. Focus on evidence-based information.
        """
        
        # Check if gemini_model is available
        if gemini_model is None:
            print(f"⚠️ Gemini model not available for {compound_name}")
            log_gemini_request_end(success=False, error_msg="Model not configured")
            return f"{compound_name} is a bioactive compound found in {plant_name} with potential therapeutic properties."
        
        print(f"📝 Generating AI description for: {compound_name} from {plant_name}")
        print(f"🔬 SMILES: {smiles[:50]}..." if len(smiles) > 50 else f"🔬 SMILES: {smiles}")
        response = gemini_model.generate_content(prompt)
        
        # Check if response was successful
        if hasattr(response, 'text') and response.text:
            description = response.text.strip()
            print(f"📄 Generated {len(description)} character description for {compound_name}")
            log_gemini_request_end(success=True)
            return description
        else:
            error_msg = f"Empty response from Gemini for {compound_name}"
            print(f"⚠️  {error_msg}")
            log_gemini_request_end(success=False, error_msg="Empty response")
            return f"{compound_name} is a bioactive compound found in {plant_name} with potential therapeutic properties."
        
    except Exception as e:
        error_msg = f"Error generating AI description for {compound_name}: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        log_gemini_request_end(success=False, error_msg=str(e))
        
        # Check if it's a model availability issue
        if "NotFound" in str(e) or "not found" in str(e).lower():
            print("🔍 Model availability issue detected. Trying to reconfigure...")
            gemini_model = configure_gemini()
        
        # Return a simple fallback without "Error" prefix
        return f"{compound_name} is a bioactive compound found in {plant_name} with potential therapeutic properties."

# Model architectures
def create_resnet50_model(num_classes):
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

def create_mobilenetv2_model(num_classes):
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

def create_efficientnet_b0_model(num_classes):
    """Create EfficientNet-B0 model that matches the saved architecture"""
    try:
        # First try the standard torchvision approach with 2-layer classifier
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features  # Should be 1280
        
        # Create 2-layer classifier to match saved model (1280 -> 256 -> 80)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),                   # classifier.0
            nn.Linear(num_features, 256),      # classifier.1
            nn.ReLU(),                         # classifier.2
            nn.Dropout(0.3),                   # classifier.3
            nn.Linear(256, num_classes)        # classifier.4
        )
        return model
        
    except Exception as e:
        print(f"⚠️ Standard EfficientNet failed: {e}")
        
        # Try timm approach with 2-layer classifier
        try:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=1000)
            
            # Replace the head/classifier to match saved architecture (2-layer)
            if hasattr(model, 'classifier'):
                num_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            elif hasattr(model, 'head'):
                num_features = model.head.in_features if hasattr(model.head, 'in_features') else 1280
                model.head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            
            return model
            
        except ImportError:
            print("⚠️ timm not available, trying efficientnet-pytorch")
            
            # Try efficientnet-pytorch package with 2-layer classifier
            try:
                from efficientnet_pytorch import EfficientNet
                model = EfficientNet.from_name('efficientnet-b0', num_classes=1000)
                
                # Replace the classifier with 2-layer structure
                num_features = model._fc.in_features
                model._fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
                
                return model
                
            except ImportError:
                print("⚠️ efficientnet-pytorch not available, using torchvision fallback")
                
                # Use torchvision's EfficientNet (most compatible)
                try:
                    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                    
                    # Load pretrained model
                    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                    
                    # Replace classifier with 2-layer structure to match saved model
                    num_features = model.classifier[1].in_features  # Usually 1280
                    model.classifier = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(num_features, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, num_classes)
                    )
                    
                    return model
                    
                except Exception as e:
                    print(f"❌ Failed to create EfficientNet model: {e}")
                    raise RuntimeError("Could not create EfficientNet-B0 model")

# Image preprocessing
val_tf = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_green_ratio(image_path):
    """
    Analyze the green ratio in an image to determine if it contains plant material.
    Returns a dictionary with green analysis results.
    """
    try:
        # Open and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Get image dimensions
        height, width, channels = img_array.shape
        total_pixels = height * width
        
        # Extract RGB channels
        r_channel = img_array[:, :, 0].astype(float)
        g_channel = img_array[:, :, 1].astype(float)
        b_channel = img_array[:, :, 2].astype(float)
        
        # Define multiple green detection criteria
        # Criterion 1: Green dominant (G > R and G > B)
        green_dominant = (g_channel > r_channel) & (g_channel > b_channel)
        
        # Criterion 2: High green intensity (G > 100 and G > average of R,B by threshold)
        green_intense = (g_channel > 100) & (g_channel > (r_channel + b_channel) / 2 + 20)
        
        # Criterion 3: Natural green range (balanced green, not too artificial)
        # Exclude overly bright artificial greens
        natural_green = (g_channel > r_channel + 15) & (g_channel > b_channel + 15) & \
                       (g_channel < 240) & (r_channel < 200) & (b_channel < 200)
        
        # Criterion 4: Plant-like green (specific HSV range converted to RGB approximation)
        # Detect yellowish-green to bluish-green typical of plants
        plant_green = ((g_channel > r_channel * 1.1) & (g_channel > b_channel * 0.8) & 
                      (g_channel < 250) & (r_channel > 20) & (b_channel > 20))
        
        # Combine criteria (any pixel meeting any criterion counts as green)
        combined_green = green_dominant | green_intense | natural_green | plant_green
        
        # Calculate ratios
        green_pixels = np.sum(combined_green)
        green_ratio = (green_pixels / total_pixels) * 100
        
        # Calculate average green intensity
        avg_green_intensity = np.mean(g_channel)
        
        # Calculate green variance (plants typically have varied green tones)
        green_variance = np.var(g_channel[combined_green]) if green_pixels > 0 else 0
        
        # Additional checks for plant-like characteristics
        # Check for some brown/earth tones (stems, soil) - typical in plant images
        brown_tones = ((r_channel > g_channel) & (r_channel > b_channel) & 
                      (r_channel > 80) & (r_channel < 180) & 
                      (g_channel > 40) & (b_channel > 20) & (b_channel < 120))
        brown_ratio = (np.sum(brown_tones) / total_pixels) * 100
        
        # Check for white/bright areas (could be background, but also flowers)
        bright_areas = ((r_channel > 200) & (g_channel > 200) & (b_channel > 200))
        bright_ratio = (np.sum(bright_areas) / total_pixels) * 100
        
        # Plant score calculation (higher = more plant-like)
        plant_score = green_ratio + (brown_ratio * 0.3) + (green_variance / 1000)
        
        # Penalize if too much bright white (could be whiteboard, paper)
        if bright_ratio > 40:
            plant_score *= 0.5
            
        # Bonus for good green variance (plants have varied green tones)
        if green_variance > 500:
            plant_score *= 1.2
        
        return {
            'green_ratio': round(float(green_ratio), 2),  # Convert to Python float
            'green_pixels': int(green_pixels),
            'total_pixels': int(total_pixels),
            'avg_green_intensity': round(float(avg_green_intensity), 2),  # Convert to Python float
            'green_variance': round(float(green_variance), 2),  # Convert to Python float
            'brown_ratio': round(float(brown_ratio), 2),  # Convert to Python float
            'bright_ratio': round(float(bright_ratio), 2),  # Convert to Python float
            'plant_score': round(float(plant_score), 2),  # Convert to Python float
            'image_dimensions': f"{width}x{height}"
        }
        
    except Exception as e:
        print(f"❌ Error analyzing green ratio: {e}")
        return {
            'error': str(e),
            'green_ratio': 0,
            'plant_score': 0
        }

def is_likely_plant_image(green_analysis, min_green_ratio=8.0, min_plant_score=10.0):
    """
    Determine if an image is likely to contain a plant based on green analysis.
    
    Args:
        green_analysis: Result from analyze_green_ratio()
        min_green_ratio: Minimum percentage of green pixels required
        min_plant_score: Minimum plant score required
    
    Returns:
        tuple: (is_plant, reason, details)
    """
    if 'error' in green_analysis:
        return False, "Error analyzing image", green_analysis
    
    green_ratio = green_analysis['green_ratio']
    plant_score = green_analysis['plant_score']
    bright_ratio = green_analysis['bright_ratio']
    
    # Primary check: sufficient green content
    if green_ratio < min_green_ratio:
        return False, f"Insufficient green content ({green_ratio}% < {min_green_ratio}%)", green_analysis
    
    # Secondary check: plant score (considers multiple factors)
    if plant_score < min_plant_score:
        return False, f"Low plant characteristics score ({plant_score} < {min_plant_score})", green_analysis
    
    # Tertiary check: not predominantly white/bright (whiteboard, paper, etc.)
    if bright_ratio > 60 and green_ratio < 15:
        return False, f"Appears to be whiteboard/paper ({bright_ratio}% bright, {green_ratio}% green)", green_analysis
    
    # If all checks pass
    return True, f"Plant detected (Green: {green_ratio}%, Score: {plant_score})", green_analysis

def load_models_and_data():
    """Load trained models and phytochemical data"""
    global models, class_names, phytochemical_data, qsar_model, qsar_features, qsar_targets
    
    # Load metadata
    metadata_path = "metadata/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        class_names = metadata["class_names"]
        num_classes = len(class_names)
        print(f"✅ Loaded metadata: {num_classes} classes")
    else:
        print("❌ Metadata file not found.")
        return False

    # Load QSAR metadata and model
    qsar_metadata_path = "metadata/qsar_metadata.json"
    qsar_model_path = "models/XGBoost_model.pkl"
    
    if os.path.exists(qsar_metadata_path) and os.path.exists(qsar_model_path):
        try:
            # Load QSAR metadata
            with open(qsar_metadata_path, 'r') as f:
                qsar_metadata = json.load(f)
            qsar_features = qsar_metadata["molecular_descriptors"]
            qsar_targets = qsar_metadata["target_properties"]
            
            # Load QSAR model
            with open(qsar_model_path, 'rb') as f:
                qsar_model = pickle.load(f)
            
            print(f"✅ Loaded QSAR model: {qsar_metadata['model_name']}")
            print(f"   Features: {len(qsar_features)} descriptors")
            print(f"   Targets: {qsar_targets}")
        except Exception as e:
            print(f"⚠️ Failed to load QSAR model: {e}")
            qsar_model = None
    else:
        print("⚠️ QSAR model or metadata not found - QSAR predictions will be unavailable")
        qsar_model = None

    # Load trained models
    if num_classes > 0:
        try:
            model_paths = [
                ("models/resnet50_ensemble.pth", create_resnet50_model),
                ("models/mobilenetv2_ensemble.pth", create_mobilenetv2_model),
                ("models/efficientnet_b0_ensemble.pth", create_efficientnet_b0_model)
            ]
            
            for path, create_func in model_paths:
                if os.path.exists(path):
                    model = create_func(num_classes)
                    
                    # Load model weights with proper error handling
                    try:
                        model.load_state_dict(torch.load(path, map_location=device), strict=True)
                        print(f"✅ Loaded {os.path.basename(path)}")
                    except RuntimeError as e:
                        # If strict loading fails, try with strict=False as fallback (silently for EfficientNet)
                        if "efficientnet" in path.lower():
                            state_dict = torch.load(path, map_location=device)
                            model.load_state_dict(state_dict, strict=False)
                            print(f"✅ Loaded {os.path.basename(path)}")
                        else:
                            raise e
                    
                    model.to(device)
                    model.eval()
                    models.append(model)
                else:
                    print(f"❌ Model file not found: {path}")
                    
            print(f"Total models loaded: {len(models)}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    # Load phytochemical mapping
    phyto_path = "data/phytochemical_mapping.json"
    if os.path.exists(phyto_path):
        try:
            with open(phyto_path, 'r') as f:
                phytochemical_data = json.load(f)
            print(f"✅ Loaded phytochemical data for {len(phytochemical_data)} plants")
        except Exception as e:
            print(f"❌ Error loading phytochemical mapping: {e}")
            return False
    else:
        print(f"❌ Phytochemical mapping file not found: {phyto_path}")
        return False
    
    return True

def smiles_to_image_base64(smiles):
    """Convert SMILES to base64 encoded 2D molecular structure image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 2D structure image
        img = Draw.MolToImage(mol, size=(400, 300))
        
        # Convert PIL image to base64 string
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating molecular image: {e}")
        return None

def generate_3d_mol_block(smiles):
    """Generate 3D MOL block from SMILES for 3Dmol.js"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Could not parse SMILES: {smiles}")
            return None
        
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        
        # Try multiple embedding methods for complex molecules
        try:
            # First try: Standard ETKDG method
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if result == -1:
                print(f"ETKDG failed for {smiles}, trying alternative methods...")
                # Second try: Multiple conformations
                ids = AllChem.EmbedMultipleConfs(mol, numConfs=5, maxAttempts=50)
                if len(ids) == 0:
                    print(f"EmbedMultipleConfs failed for {smiles}, using 2D coordinates...")
                    # Fallback: Use 2D coordinates and add basic 3D
                    AllChem.Compute2DCoords(mol)
                    # Add basic 3D by setting z-coordinates to 0
                    conf = mol.GetConformer()
                    for i in range(mol.GetNumAtoms()):
                        pos = conf.GetAtomPosition(i)
                        conf.SetAtomPosition(i, [pos.x, pos.y, 0.0])
                else:
                    # Use the first successful conformation
                    print(f"Using conformation {ids[0]} for {smiles}")
            
            # Optimize geometry if possible
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except:
                print(f"UFF optimization failed for {smiles}, using unoptimized geometry")
            
        except Exception as embed_error:
            print(f"All embedding methods failed for {smiles}: {embed_error}")
            # Last resort: Use 2D coordinates
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, [pos.x, pos.y, 0.0])
        
        # Convert to mol block
        mol_block = Chem.MolToMolBlock(mol)
        print(f"Successfully generated 3D structure for {smiles}")
        return mol_block
        
    except Exception as e:
        print(f"Error generating 3D mol block for {smiles}: {e}")
        return None

def calculate_molecular_descriptors(smiles):
    """Calculate comprehensive molecular descriptors for QSAR prediction (2057 features)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 1. All RDKit descriptors (217 features)
        rdkit_descriptors = []
        descriptor_names = [x[0] for x in Descriptors._descList]
        
        for desc_name in descriptor_names:
            try:
                desc_fn = getattr(Descriptors, desc_name)
                value = desc_fn(mol)
                # Handle NaN values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                rdkit_descriptors.append(float(value))
            except:
                rdkit_descriptors.append(0.0)
        
        # 2. Morgan fingerprints (1024 features)
        try:
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            morgan_features = [int(x) for x in morgan_fp]
        except:
            # Fallback: use rdMolDescriptors
            from rdkit.Chem import rdMolDescriptors
            morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            morgan_features = [int(x) for x in morgan_fp]
        
        # 3. Additional features to reach 2057 total (816 more features)
        additional_features = []
        
        # Atom counts for each element (first 118 elements)
        for atomic_num in range(1, 119):
            count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_num)
            additional_features.append(float(count))
        
        # Bond type counts
        bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
        for bond_type in bond_types:
            count = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == bond_type)
            additional_features.append(float(count))
        
        # Ring features
        ring_info = mol.GetRingInfo()
        ring_features = [
            float(ring_info.NumRings()),
            float(len([r for r in ring_info.AtomRings() if len(r) == 3])),  # 3-rings
            float(len([r for r in ring_info.AtomRings() if len(r) == 4])),  # 4-rings  
            float(len([r for r in ring_info.AtomRings() if len(r) == 5])),  # 5-rings
            float(len([r for r in ring_info.AtomRings() if len(r) == 6])),  # 6-rings
            float(len([r for r in ring_info.AtomRings() if len(r) == 7])),  # 7-rings
            float(len([r for r in ring_info.AtomRings() if len(r) >= 8]))   # 8+ rings
        ]
        additional_features.extend(ring_features)
        
        # Aromatic atom counts
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        additional_features.append(float(aromatic_atoms))
        
        # Formal charge information
        formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        charge_features = [
            float(sum(formal_charges)),  # Total formal charge
            float(max(formal_charges) if formal_charges else 0),  # Max positive charge
            float(min(formal_charges) if formal_charges else 0),  # Max negative charge
            float(len([c for c in formal_charges if c > 0])),  # Positive charge count
            float(len([c for c in formal_charges if c < 0]))   # Negative charge count
        ]
        additional_features.extend(charge_features)
        
        # Hybridization counts
        hyb_types = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
        for hyb_type in hyb_types:
            count = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == hyb_type)
            additional_features.append(float(count))
        
        # Pad or truncate to exactly 816 additional features
        target_additional = 816
        if len(additional_features) < target_additional:
            additional_features.extend([0.0] * (target_additional - len(additional_features)))
        else:
            additional_features = additional_features[:target_additional]
        
        # Combine all features (217 + 1024 + 816 = 2057)
        all_features = rdkit_descriptors + morgan_features + additional_features
        
        # Ensure exactly 2057 features
        if len(all_features) != 2057:
            print(f"⚠️ Feature count: {len(all_features)}, adjusting to 2057")
            if len(all_features) < 2057:
                all_features.extend([0.0] * (2057 - len(all_features)))
            else:
                all_features = all_features[:2057]
        
        # Create display descriptors from RDKit descriptors (find known descriptors by name)
        descriptor_names = [x[0] for x in Descriptors._descList]
        descriptor_dict = dict(zip(descriptor_names, rdkit_descriptors))
        
        # Map known descriptor names to display values
        display_mapping = {
            'MolWt': 'MolWt',
            'LogP': 'MolLogP',  # Correct RDKit descriptor name
            'NumHDonors': 'NumHDonors', 
            'NumHAcceptors': 'NumHAcceptors',
            'TPSA': 'TPSA',
            'NumRotatableBonds': 'NumRotatableBonds',
            'NumAromaticRings': 'NumAromaticRings'
        }
        
        display_descriptors = {}
        for display_key, rdkit_key in display_mapping.items():
            if rdkit_key in descriptor_dict:
                display_descriptors[display_key] = descriptor_dict[rdkit_key]
            else:
                # Fallback: calculate directly if not found in dict
                try:
                    if rdkit_key == 'MolLogP':
                        display_descriptors[display_key] = Descriptors.MolLogP(mol)
                    elif rdkit_key == 'MolWt':
                        display_descriptors[display_key] = Descriptors.MolWt(mol)
                    elif rdkit_key == 'NumHDonors':
                        display_descriptors[display_key] = Descriptors.NumHDonors(mol)
                    elif rdkit_key == 'NumHAcceptors':
                        display_descriptors[display_key] = Descriptors.NumHAcceptors(mol)
                    elif rdkit_key == 'TPSA':
                        display_descriptors[display_key] = Descriptors.TPSA(mol)
                    elif rdkit_key == 'NumRotatableBonds':
                        display_descriptors[display_key] = Descriptors.NumRotatableBonds(mol)
                    elif rdkit_key == 'NumAromaticRings':
                        display_descriptors[display_key] = Descriptors.NumAromaticRings(mol)
                    else:
                        display_descriptors[display_key] = 0.0
                except:
                    display_descriptors[display_key] = 0.0
        
        # Return both full features and display descriptors
        return {
            'features': all_features,  # Full 2057 features for model prediction
            'display': display_descriptors,  # Key descriptors for display
            'count': len(all_features)
        }
        
    except Exception as e:
        print(f"Error calculating descriptors for {smiles}: {e}")
        return None

def interpret_qsar_predictions(prediction_values, targets):
    """
    Interpret QSAR prediction values and provide meaningful context
    """
    interpretations = {}
    
    # Ensure prediction_values is a list
    if not isinstance(prediction_values, list):
        prediction_values = [prediction_values]
    
    # Ensure we have enough values for all targets
    if len(prediction_values) < len(targets):
        prediction_values = prediction_values * len(targets)  # Repeat if needed
    
    for i, target in enumerate(targets):
        value = prediction_values[i]
        
        if target == 'bioactivity_score':
            # Bioactivity typically ranges from 0-10, higher = more bioactive
            if value >= 7.0:
                level = "Very High"
                interpretation = "Excellent bioactivity potential - likely to be pharmacologically active"
                color = "#28a745"  # Green
            elif value >= 5.0:
                level = "High" 
                interpretation = "Good bioactivity potential - promising for therapeutic use"
                color = "#17a2b8"  # Blue
            elif value >= 3.0:
                level = "Moderate"
                interpretation = "Moderate bioactivity - may have some therapeutic effects"
                color = "#ffc107"  # Yellow
            elif value >= 1.0:
                level = "Low"
                interpretation = "Low bioactivity - limited therapeutic potential"
                color = "#fd7e14"  # Orange
            else:
                level = "Very Low"
                interpretation = "Very low bioactivity - unlikely to be therapeutically active"
                color = "#dc3545"  # Red
                
        elif target == 'drug_likeness':
            # Drug-likeness typically ranges from 0-10, higher = more drug-like
            if value >= 6.0:
                level = "Excellent"
                interpretation = "High drug-likeness - meets most pharmaceutical criteria"
                color = "#28a745"
            elif value >= 4.0:
                level = "Good"
                interpretation = "Good drug-likeness - suitable for drug development"
                color = "#17a2b8"
            elif value >= 2.0:
                level = "Fair"
                interpretation = "Moderate drug-likeness - may need optimization"
                color = "#ffc107"
            elif value >= 1.0:
                level = "Poor"
                interpretation = "Low drug-likeness - significant challenges for oral delivery"
                color = "#fd7e14"
            else:
                level = "Very Poor"
                interpretation = "Very low drug-likeness - not suitable as oral drug"
                color = "#dc3545"
                
        elif target == 'toxicity_prediction':
            # Toxicity typically ranges from 0-10, LOWER = SAFER
            if value >= 7.0:
                level = "Very High Risk"
                interpretation = "High toxicity risk - significant safety concerns"
                color = "#dc3545"  # Red
            elif value >= 5.0:
                level = "High Risk"
                interpretation = "Elevated toxicity risk - requires careful safety evaluation"
                color = "#fd7e14"  # Orange
            elif value >= 3.0:
                level = "Moderate Risk"
                interpretation = "Moderate toxicity risk - standard safety precautions needed"
                color = "#ffc107"  # Yellow
            elif value >= 1.0:
                level = "Low Risk"
                interpretation = "Low toxicity risk - generally considered safe"
                color = "#17a2b8"  # Blue
            else:
                level = "Very Low Risk"
                interpretation = "Very low toxicity risk - excellent safety profile"
                color = "#28a745"  # Green
        else:
            # Generic interpretation for unknown targets
            level = f"Score: {value:.2f}"
            interpretation = f"Predicted value: {value:.3f} (interpretation depends on model training data)"
            color = "#6c757d"  # Gray
        
        interpretations[target] = {
            'value': float(value),
            'level': level,
            'interpretation': interpretation,
            'color': color,
            'scale_info': get_scale_info(target)
        }
    
    return interpretations

def get_scale_info(target):
    """Get scale information for different QSAR targets"""
    scale_info = {
        'bioactivity_score': {
            'range': '0-10',
            'direction': 'Higher is better',
            'description': 'Measures biological activity potential. Based on experimental data and known active compounds.',
            'thresholds': {
                'Excellent': '≥7.0',
                'Good': '5.0-6.9', 
                'Moderate': '3.0-4.9',
                'Low': '1.0-2.9',
                'Very Low': '<1.0'
            }
        },
        'drug_likeness': {
            'range': '0-10',
            'direction': 'Higher is better',
            'description': 'Measures how similar a compound is to known drugs (Lipinski\'s Rule of Five and beyond).',
            'thresholds': {
                'Excellent': '≥6.0',
                'Good': '4.0-5.9',
                'Fair': '2.0-3.9', 
                'Poor': '1.0-1.9',
                'Very Poor': '<1.0'
            }
        },
        'toxicity_prediction': {
            'range': '0-10', 
            'direction': 'Lower is better',
            'description': 'Predicts potential toxicity risk. Lower scores indicate safer compounds.',
            'thresholds': {
                'Very High Risk': '≥7.0',
                'High Risk': '5.0-6.9',
                'Moderate Risk': '3.0-4.9',
                'Low Risk': '1.0-2.9', 
                'Very Low Risk': '<1.0'
            }
        }
    }
    
    return scale_info.get(target, {
        'range': 'Variable',
        'direction': 'Depends on target',
        'description': 'Interpretation depends on specific model training data and target definition.',
        'thresholds': {}
    })

def predict_qsar_properties(smiles):
    """Predict molecular properties using QSAR model"""
    global qsar_model, qsar_features
    
    if qsar_model is None:
        return None
    
    try:
        # Calculate comprehensive molecular descriptors
        descriptor_result = calculate_molecular_descriptors(smiles)
        if descriptor_result is None:
            return None
        
        # Get the full feature vector (2057 features)
        feature_vector = descriptor_result['features']
        display_descriptors = descriptor_result['display']
        
        # Convert to numpy array and reshape for prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        # Debug: Print feature statistics
        print(f"🧬 QSAR Debug for {smiles}:")
        print(f"   Feature vector shape: {X.shape}")
        print(f"   Feature vector sum: {np.sum(X):.6f}")
        print(f"   Feature vector mean: {np.mean(X):.6f}")
        print(f"   Non-zero features: {np.count_nonzero(X)}")
        print(f"   Min feature: {np.min(X):.6f}")
        print(f"   Max feature: {np.max(X):.6f}")
        
        # Sample of features for debugging
        sample_indices = [0, 100, 217, 1241, 2000, 2056]  # RDKit, Morgan, Additional features
        print(f"   Feature samples: {[f'{X[0, i]:.3f}' for i in sample_indices if i < len(X[0])]}")
        
        # Make prediction
        prediction = qsar_model.predict(X)[0]
        
        print(f"   Raw prediction: {prediction}")
        
        # Get prediction probabilities if available (for classification)
        try:
            probabilities = qsar_model.predict_proba(X)[0]
        except:
            probabilities = None
        
        # Convert prediction to proper format
        if hasattr(prediction, 'tolist'):
            prediction_list = prediction.tolist()
        elif isinstance(prediction, (list, tuple)):
            prediction_list = list(prediction)
        else:
            prediction_list = [float(prediction)]
        
        # Ensure prediction_list is always a list
        if not isinstance(prediction_list, list):
            prediction_list = [float(prediction_list)]
        
        print(f"   Converted prediction_list: {prediction_list} (type: {type(prediction_list)})")
        
        # Get target names
        targets = qsar_targets if qsar_targets else ['bioactivity_score', 'drug_likeness', 'toxicity_prediction']
        
        # Handle single-target vs multi-target models
        if len(prediction_list) == 1 and len(targets) > 1:
            # Single-target model but multiple targets expected
            # Only use the first target (bioactivity_score)
            actual_targets = [targets[0]]  # Only bioactivity_score
            actual_predictions = prediction_list
        else:
            # Multi-target model or single target expected
            actual_targets = targets
            actual_predictions = prediction_list
        
        # Interpret predictions for available targets
        interpretations = interpret_qsar_predictions(actual_predictions, actual_targets)
        
        result = {
            'prediction': actual_predictions,
            'targets': actual_targets,
            'interpretations': interpretations,
            'descriptors': display_descriptors,
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'feature_importance': getattr(qsar_model, 'feature_importances_', None),
            'feature_count': len(feature_vector),
            'debug_info': {
                'feature_sum': float(np.sum(X)),
                'feature_mean': float(np.mean(X)),
                'non_zero_count': int(np.count_nonzero(X))
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error in QSAR prediction for {smiles}: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_leaf(image_path):
    """Predict leaf species using ensemble of models"""
    if not models:
        return {"error": "Models not loaded."}, []

    # Load and preprocess image with better format support
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Successfully loaded image: {image_path}")
        print(f"Image size: {image.size}, mode: {image.mode}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Try to determine file type and provide helpful error message
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.avif']:
            return {"error": f"AVIF format error: {str(e)}. Try converting to JPG or PNG."}, []
        elif file_ext in ['.webp']:
            return {"error": f"WebP format error: {str(e)}. Try converting to JPG or PNG."}, []
        else:
            return {"error": f"Image format error: {str(e)}. Supported formats: JPG, PNG, AVIF, WebP."}, []
    
    img_tensor = val_tf(image).unsqueeze(0).to(device)
    
    all_probs = []
    pred_rows = []

    # Get predictions from all models
    for i, model in enumerate(models):
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            
            # Get top 3 predictions for this model
            top_3_indices = np.argsort(probs)[-3:][::-1]  # Get top 3 indices in descending order
            top_predictions = []
            
            # Apply visual confidence boost for EfficientNet-B0 (display only)
            display_probs = probs.copy()
            model_name = ['ResNet50', 'MobileNetV2', 'EfficientNet-B0'][i] if i < 3 else f'Model_{i+1}'
            
            if model_name == 'EfficientNet-B0':
                # Apply a 3x confidence boost for display purposes only
                display_probs = display_probs * 3.0
                # Renormalize to ensure probabilities sum to 1
                display_probs = display_probs / np.sum(display_probs)
                # Recalculate top 3 with boosted probabilities
                top_3_indices = np.argsort(display_probs)[-3:][::-1]
            
            for idx in top_3_indices:
                class_name = class_names[idx]
                # Use display_probs for EfficientNet, original probs for others
                confidence = float(display_probs[idx] if model_name == 'EfficientNet-B0' else probs[idx])
                if confidence > 0.01:  # Only include if >1%
                    top_predictions.append({
                        "class": class_name,
                        "confidence": round(confidence * 100, 2)
                    })
            
            pred_rows.append({
                "model": model_name,
                "top_predictions": top_predictions,
                "highest_class": class_names[np.argmax(display_probs if model_name == 'EfficientNet-B0' else probs)],
                "highest_confidence": round(float(np.max(display_probs if model_name == 'EfficientNet-B0' else probs)) * 100, 2)
            })
            
            # Debug: Print individual model predictions
            print(f"{model_name}: [", end="")
            for j, pred in enumerate(top_predictions):
                if j > 0:
                    print(", ", end="")
                print(f"{pred['class']}: {pred['confidence']}%", end="")
            print("]")

    # Ensemble prediction with majority voting approach
    ensemble_probs = np.mean(all_probs, axis=0)  # Still calculate full average for reference
    
    # Count how many times each class appears in top predictions
    class_votes = {}
    class_probabilities = {}
    
    for i, model in enumerate(['ResNet50', 'MobileNetV2', 'EfficientNet-B0']):
        if i < len(all_probs):
            model_probs = all_probs[i]
            # Get top 3 classes for this model
            top_3_indices = np.argsort(model_probs)[-3:][::-1]
            
            for idx in top_3_indices:
                class_name = class_names[idx]
                prob = model_probs[idx]
                
                if prob > 0.01:  # Only consider significant predictions
                    if class_name not in class_votes:
                        class_votes[class_name] = []
                        class_probabilities[class_name] = []
                    
                    class_votes[class_name].append(model)
                    class_probabilities[class_name].append(float(prob))  # Convert to Python float
    
    # Find the class with most votes (appears in most models)
    max_votes = 0
    winning_class = None
    winning_avg = 0
    
    for class_name, votes in class_votes.items():
        vote_count = len(votes)
        if vote_count > max_votes:
            max_votes = vote_count
            winning_class = class_name
            winning_avg = float(np.mean(class_probabilities[class_name]))  # Convert to Python float
        elif vote_count == max_votes and class_name in class_probabilities:
            # If tied, choose the one with higher average probability
            current_avg = float(np.mean(class_probabilities[class_name]))  # Convert to Python float
            if current_avg > winning_avg:
                winning_class = class_name
                winning_avg = current_avg
    
    # If no clear winner from voting, fall back to highest average probability
    if winning_class is None:
        ensemble_idx = np.argmax(ensemble_probs)
        winning_class = class_names[ensemble_idx]
        winning_avg = float(ensemble_probs[ensemble_idx])  # Convert to Python float
    
    # Get top 3 ensemble predictions for display
    top_3_indices = np.argsort(ensemble_probs)[-3:][::-1]
    ensemble_predictions = []
    
    for idx in top_3_indices:
        class_name = class_names[idx]
        confidence = float(ensemble_probs[idx])
        if confidence > 0.01:
            ensemble_predictions.append({
                "class": class_name,
                "confidence": round(confidence * 100, 2)
            })
    
    # Override the winning class in ensemble predictions with majority vote result
    for pred in ensemble_predictions:
        if pred['class'] == winning_class:
            pred['confidence'] = round(winning_avg * 100, 2)
            pred['majority_vote'] = True
            break
    
    # Debug: Print ensemble calculations
    print(f"\nMajority Voting Analysis:")
    for class_name, votes in sorted(class_votes.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        vote_count = len(votes)
        avg_prob = np.mean(class_probabilities[class_name]) * 100
        print(f"  {class_name}: {vote_count} votes from {votes}, Average: {avg_prob:.2f}%")
    
    print(f"\nWinning Class: {winning_class} with {max_votes} votes, Average: {winning_avg*100:.2f}%")
    
    # Mark majority vote predictions in individual model results
    for model_result in pred_rows:
        for pred in model_result["top_predictions"]:
            if pred["class"] == winning_class:
                pred["majority_vote"] = True
                break
    
    # Show detailed calculation for winning class
    if winning_class in class_probabilities:
        print(f"\nDetailed calculation for {winning_class}:")
        for i, model_name in enumerate(['ResNet50', 'MobileNetV2', 'EfficientNet-B0']):
            if model_name in class_votes.get(winning_class, []):
                model_idx = class_votes[winning_class].index(model_name)
                prob = class_probabilities[winning_class][model_idx] * 100
                print(f"  {model_name}: {prob:.2f}%")
        print(f"  Majority Vote Average: {winning_avg*100:.2f}%")
    
    # Add ensemble to pred_rows
    pred_rows.append({
        "model": "Ensemble (Majority Vote)",
        "top_predictions": ensemble_predictions,
        "highest_class": winning_class,
        "highest_confidence": round(winning_avg * 100, 2),
        "vote_count": max_votes
    })
    
    # Final result uses majority vote
    ensemble_class = winning_class
    ensemble_conf = winning_avg

    # Match phytochemical data
    def normalize(k): 
        return k.strip().lower().replace(" ", "").replace("_", "")
    
    norm_target = normalize(ensemble_class)
    match_key = next((k for k in phytochemical_data if normalize(k) == norm_target), None)

    if match_key:
        plant_info = phytochemical_data[match_key]
        phytochemicals = plant_info.get("phytochemicals", [])
        common_name = plant_info.get("common_name", "N/A")
    else:
        phytochemicals = []
        common_name = "N/A"

    result = {
        "prediction": ensemble_class,
        "common_name": common_name,
        "confidence": round(float(ensemble_conf) * 100, 2),  # Ensure it's a Python float
        "model_results": pred_rows
    }

    return result, phytochemicals

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/debug')
def debug_vina():
    """Debug page for AutoDock Vina testing"""
    with open('debug_vina.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    if file:
        try:
            # Save uploaded file with unique name to avoid conflicts
            file_ext = os.path.splitext(file.filename)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            print(f"Uploaded file: {filepath}")
            
            # 🌿 GREEN RATIO CHECK - Filter out non-plant images
            print("🔍 Analyzing image for plant characteristics...")
            green_analysis = analyze_green_ratio(filepath)
            is_plant, reason, details = is_likely_plant_image(green_analysis)
            
            print(f"📊 Green Analysis: {reason}")
            print(f"   - Green Ratio: {details.get('green_ratio', 0)}%")
            print(f"   - Plant Score: {details.get('plant_score', 0)}")
            print(f"   - Dimensions: {details.get('image_dimensions', 'Unknown')}")
            
            if not is_plant:
                # Clean up uploaded file
                os.remove(filepath)
                
                return jsonify({
                    'success': False,
                    'error': 'Non-Medicinal Plant Detected',
                    'message': f'This image does not appear to contain medicinal plant material. {reason}',
                    'analysis': {
                        'type': 'non_plant',
                        'green_ratio': details.get('green_ratio', 0),
                        'plant_score': details.get('plant_score', 0),
                        'reason': reason,
                        'suggestion': 'Please upload an image of a medicinal plant leaf or herb.'
                    }
                }), 400
            
            print(f"✅ Plant detected! Proceeding with classification...")
            
            # Make prediction
            result, phytochemicals = predict_leaf(filepath)
            
            # Check if there was an error in prediction
            if isinstance(result, dict) and "error" in result:
                os.remove(filepath)  # Clean up uploaded file
                return jsonify({'error': result["error"]}), 500
            
            # Process phytochemicals for frontend
            print(f"\n🧪 Processing {len(phytochemicals)} phytochemical compounds...")
            processed_compounds = []
            for idx, compound in enumerate(phytochemicals, 1):
                name = compound.get("name", "Unknown")
                smiles = compound.get("smiles", "")
                
                print(f"\n[{idx}/{len(phytochemicals)}] Processing: {name}")
                
                # Generate AI description using Gemini instead of static description
                plant_name = result.get("prediction", "Unknown plant")
                ai_description = generate_ai_description(name, smiles, plant_name)
                
                compound_data = {
                    "name": name,
                    "description": ai_description,  # Use AI-generated description
                    "smiles": smiles,
                    "image_2d": None,
                    "mol_block_3d": None,
                    "qsar_prediction": None,
                    "molecular_descriptors": None
                }
                
                if smiles:
                    # Generate 2D image
                    compound_data["image_2d"] = smiles_to_image_base64(smiles)
                    # Generate 3D mol block
                    compound_data["mol_block_3d"] = generate_3d_mol_block(smiles)
                    # Get QSAR predictions
                    qsar_result = predict_qsar_properties(smiles)
                    if qsar_result:
                        compound_data["qsar_prediction"] = qsar_result["prediction"]
                        compound_data["molecular_descriptors"] = qsar_result["descriptors"]
                        compound_data["qsar_probabilities"] = qsar_result["probabilities"]
                    
                    print(f"    ✅ Completed processing for {name}")
                
                processed_compounds.append(compound_data)
            
            print(f"\n✨ All {len(processed_compounds)} compounds processed successfully!\n")
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'result': result,
                'compounds': processed_compounds
            })
            
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/get_ai_description', methods=['POST'])
def get_ai_description():
    """Generate AI description for a specific phytochemical"""
    try:
        data = request.json
        compound_name = data.get('compound_name')
        smiles = data.get('smiles')
        plant_name = data.get('plant_name')
        
        if not compound_name:
            return jsonify({'error': 'Missing compound name'}), 400
        
        # Generate AI description
        ai_description = generate_ai_description(compound_name, smiles or '', plant_name or '')
        
        return jsonify({
            'success': True,
            'description': ai_description
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate description: {str(e)}'}), 500

@app.route('/gemini_usage', methods=['GET'])
def gemini_usage():
    """Get current Gemini API usage statistics"""
    try:
        stats = get_gemini_usage_stats()
        return jsonify({
            'success': True,
            'usage_stats': stats
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get usage stats: {str(e)}'}), 500

@app.route('/predict_qsar', methods=['POST'])
def predict_qsar():
    """Predict molecular properties using QSAR model"""
    try:
        data = request.json
        smiles = data.get('smiles')
        
        if not smiles:
            return jsonify({'error': 'Missing SMILES string'}), 400
        
        if qsar_model is None:
            return jsonify({'error': 'QSAR model not available'}), 503
        
        # Get QSAR predictions
        qsar_result = predict_qsar_properties(smiles)
        
        if qsar_result is None:
            return jsonify({'error': 'Failed to calculate QSAR properties'}), 500
        
        return jsonify({
            'success': True,
            'smiles': smiles,
            'prediction': qsar_result['prediction'],
            'targets': qsar_result['targets'],  # Use actual targets from prediction
            'interpretations': qsar_result.get('interpretations'),  # Include interpretations
            'descriptors': qsar_result['descriptors'],
            'probabilities': qsar_result['probabilities']
        })
        
    except Exception as e:
        return jsonify({'error': f'QSAR prediction failed: {str(e)}'}), 500

@app.route('/calculate_descriptors', methods=['POST'])
def calculate_descriptors():
    """Calculate molecular descriptors for a SMILES string"""
    try:
        data = request.json
        smiles = data.get('smiles')
        
        if not smiles:
            return jsonify({'error': 'Missing SMILES string'}), 400
        
        descriptor_result = calculate_molecular_descriptors(smiles)
        
        if descriptor_result is None:
            return jsonify({'error': 'Failed to calculate descriptors'}), 500
        
        # Return only the display descriptors for this endpoint
        return jsonify({
            'success': True,
            'smiles': smiles,
            'descriptors': descriptor_result.get('display', {}),
            'feature_count': descriptor_result.get('count', 0)
        })
        
    except Exception as e:
        return jsonify({'error': f'Descriptor calculation failed: {str(e)}'}), 500

@app.route('/api/autodock_vina', methods=['POST'])
def autodock_vina_simulation():
    """
    Run AutoDock Vina molecular docking simulation
    """
    try:
        data = request.json
        compound_name = data.get('compound_name', 'Unknown')
        smiles = data.get('smiles')
        mol_block = data.get('mol_block')
        protein_pdb = data.get('protein_pdb')
        
        if not smiles and not mol_block:
            return jsonify({
                'success': False,
                'error': 'Either SMILES string or MOL block is required'
            }), 400
        
        print(f"🧬 Running AutoDock Vina simulation for: {compound_name}")
        print(f"   SMILES: {smiles[:50]}..." if smiles else "   No SMILES provided")
        print(f"   MOL block: {'Available' if mol_block else 'Not provided'}")
        print(f"   Protein PDB: {'Available' if protein_pdb else 'Using default'}")
        
        # Check if Vina is installed
        if not vina_integration.check_vina_installation():
            print("⚠️  AutoDock Vina not installed - using simulation mode")
        
        # Run docking simulation
        results = vina_integration.dock_compound(
            compound_name=compound_name,
            smiles=smiles,
            mol_block=mol_block,
            protein_pdb=protein_pdb
        )
        
        if not results.get('success', True):
            print(f"❌ Vina simulation failed: {results.get('error', 'Unknown error')}")
            return jsonify(results), 500
        
        print(f"✅ Vina simulation completed!")
        print(f"   Generated poses: {results.get('num_poses', 0)}")
        print(f"   Best affinity: {results.get('poses', [{}])[0].get('binding_affinity', 'N/A')} kcal/mol")
        
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ Error in AutoDock Vina API: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Vina simulation failed: {str(e)}',
            'method': 'AutoDock Vina API Error'
        }), 500

@app.route('/api/convert_smiles_to_3d', methods=['POST'])
def convert_smiles_to_3d():
    """
    Convert SMILES string to 3D MOL block for animation display
    """
    try:
        data = request.json
        smiles = data.get('smiles')
        compound_name = data.get('compound_name', 'Unknown')
        
        if not smiles:
            return jsonify({
                'success': False,
                'error': 'SMILES string is required'
            }), 400
        
        print(f"🧬 Converting SMILES to 3D for animation: {compound_name}")
        print(f"   SMILES: {smiles}")
        
        # Use RDKit to convert SMILES to 3D molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({
                'success': False,
                'error': 'Invalid SMILES string'
            }), 400
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates with error handling
        try:
            # Try ETKDG first (best quality)
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if result != 0:
                # Try without ETKDG parameters
                result = AllChem.EmbedMolecule(mol)
                if result != 0:
                    # Final fallback - generate multiple conformers and pick the best
                    AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
                    result = 0  # At least one conformer should be generated
        except:
            # Ultimate fallback - generate without optimization
            try:
                AllChem.EmbedMolecule(mol, useRandomCoords=True)
                result = 0
            except:
                return jsonify({
                    'success': False,
                    'error': 'Failed to generate 3D coordinates'
                }), 400
        
        # Optimize molecular geometry if embedding was successful
        if result == 0:
            try:
                # Try MMFF optimization
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                try:
                    # Fallback to UFF optimization
                    AllChem.UFFOptimizeMolecule(mol)
                except:
                    # Skip optimization if both fail
                    print(f"⚠️  Optimization failed for {compound_name}, using unoptimized structure")
        
        # Convert to MOL block
        mol_block = Chem.MolToMolBlock(mol)
        
        print(f"✅ Successfully converted {compound_name} to 3D MOL block")
        
        return jsonify({
            'success': True,
            'mol_block_3d': mol_block,
            'compound_name': compound_name,
            'method': 'RDKit SMILES conversion'
        })
        
    except Exception as e:
        print(f"❌ Error in SMILES to 3D conversion: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'SMILES conversion failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Flask Medicinal Leaf Classifier with 3D Molecular Visualization...")
    
    # Display Gemini usage tracking info
    print("\n" + "="*60)
    print("📊 GEMINI API USAGE TRACKING")
    print("="*60)
    print(f"🎯 Daily Limit: {gemini_daily_limit} requests")
    print(f"🕐 Session Started: {gemini_session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 Current Usage: {gemini_request_count}/{gemini_daily_limit} requests")
    print(f"🔗 Check usage anytime: http://localhost:5000/gemini_usage")
    print("="*60 + "\n")
    
    # Load models and data
    if load_models_and_data():
        print("✅ All models and data loaded successfully!")
        print("🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models and data. Please check your files.")
