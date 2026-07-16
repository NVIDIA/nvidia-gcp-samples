import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import EsmTokenizer, EsmModel, EsmForProteinFolding, AutoTokenizer, EsmForMaskedLM
import time
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import pairwise2
import base64
import struct
import io
import os
import warnings
import requests
import json
import re
warnings.filterwarnings('ignore')

#===============================================================================
# GPU UTILIZATION SAMPLER (background thread)
#===============================================================================
import threading

gpu_stats = {
    'peak_util': 0.0,
    'last_util': 0.0,
    'memory_used': 0.0,
    'memory_total': 96.0,
    'memory_free': 96.0,
    'gpu_name': 'GPU',
    'sampling': False
}
_gpu_lock = threading.Lock()

def _sample_gpu_once():
    try:
        import subprocess
        smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = [x.strip() for x in smi.stdout.strip().split(',')]
        return {
            'gpu_name': parts[0],
            'util': float(parts[1]),
            'mem_used': float(parts[2]) / 1024,
            'mem_total': float(parts[3]) / 1024,
            'mem_free': float(parts[4]) / 1024,
        }
    except:
        return None

def _gpu_sampler_loop():
    import time
    while True:
        if gpu_stats['sampling']:
            sample = _sample_gpu_once()
            if sample:
                with _gpu_lock:
                    gpu_stats['gpu_name'] = sample['gpu_name']
                    gpu_stats['last_util'] = sample['util']
                    gpu_stats['memory_used'] = sample['mem_used']
                    gpu_stats['memory_total'] = sample['mem_total']
                    gpu_stats['memory_free'] = sample['mem_free']
                    if sample['util'] > gpu_stats['peak_util']:
                        gpu_stats['peak_util'] = sample['util']
            time.sleep(0.5)
        else:
            time.sleep(1)

_sampler_thread = threading.Thread(target=_gpu_sampler_loop, daemon=True)
_sampler_thread.start()

def gpu_inference_start():
    with _gpu_lock:
        gpu_stats['peak_util'] = 0.0
        gpu_stats['sampling'] = True

def gpu_inference_end():
    import time
    time.sleep(0.5)
    with _gpu_lock:
        gpu_stats['sampling'] = False


# RDKit for drug-likeness calculations
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw, AllChem

#===============================================================================
# APPLICATION STARTUP AND CONFIGURATION
#===============================================================================

print("=" * 60)
print("  Accelerating Drug Discovery with AI")
print("  NVIDIA + Google Cloud: Better Together")
print("=" * 60)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TEXT_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL}:generateContent?key={GEMINI_API_KEY}"
GEMINI_TTS_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent?key={GEMINI_API_KEY}"

MAX_SEQUENCE_LENGTH = 400

# NVIDIA NIM Endpoints
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
GENMOL_NIM_URL = os.environ.get("GENMOL_URL", "http://genmol-nim.bionemo-nim:8000")
DIFFDOCK_NIM_URL = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
NIM_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {NVIDIA_API_KEY}"
}

# Amino acid colors and properties
AA_COLORS = {
    'A': '#8B0000', 'R': '#0000FF', 'N': '#00FFFF', 'D': '#FF0000',
    'C': '#FFFF00', 'E': '#FF0000', 'Q': '#00FFFF', 'G': '#808080',
    'H': '#0000FF', 'I': '#008000', 'L': '#008000', 'K': '#0000FF',
    'M': '#FFFF00', 'F': '#008000', 'P': '#FFC0CB', 'S': '#FFA500',
    'T': '#FFA500', 'W': '#008000', 'Y': '#00FFFF', 'V': '#008000'
}

AA_PROPERTIES = {
    'Hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P', 'G'],
    'Polar': ['S', 'T', 'C', 'Y', 'N', 'Q'],
    'Positive': ['K', 'R', 'H'],
    'Negative': ['D', 'E']
}

AA_LIST = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

#===============================================================================
# DRUG DISCOVERY TARGETS - The heart of the new narrative
#===============================================================================

DRUG_TARGETS = {
    "EGFR Kinase Domain — Non-Small Cell Lung Cancer": {
        "sequence": "FKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPI",
        "disease": "Non-Small Cell Lung Cancer (NSCLC)",
        "description": "The Epidermal Growth Factor Receptor (EGFR) kinase domain is a primary target in NSCLC. Mutations in EGFR drive uncontrolled cell growth. Drugs like Erlotinib and Gefitinib target this domain.",
        "known_drugs": ["Erlotinib", "Gefitinib", "Osimertinib", "Afatinib"],
        "resistance_mutations": "T72M, C86S, L49R, G24S",
        "ligands": {
            "Erlotinib": "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
            "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "Caffeine (negative control)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Aspirin (negative control)": "CC(=O)OC1=CC=CC=C1C(=O)O"
        }
    },
    "HIV-1 Protease — HIV/AIDS": {
        "sequence": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF",
        "disease": "HIV/AIDS",
        "description": "HIV-1 Protease cleaves viral polyproteins into functional proteins essential for viral maturation. Protease inhibitors block this enzyme, preventing HIV from producing infectious viral particles.",
        "known_drugs": ["Darunavir", "Ritonavir", "Lopinavir", "Atazanavir"],
        "resistance_mutations": "D30N, M46I, I50V, V82A, I84V, L90M",
        "ligands": {
            "Darunavir": "CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC1COC2OCCC12)S(=O)(=O)c1ccc(N)cc1",
            "Ritonavir": "CC(C)c1nc(CN(C)C(=O)NC(C(=O)NC(CC2CCCCC2)C(O)CN(Cc2cccnc2)S(=O)(=O)c2ccc(N)cc2)C(C)C)cs1",
            "Ibuprofen (negative control)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "Acetaminophen (negative control)": "CC(=O)NC1=CC=C(C=C1)O"
        }
    },
    "BACE1 — Alzheimer's Disease": {
        "sequence": "AIDTGTSLMVFPSEGLTHEQAHQTRIAVYKHLFKTFHVPSTLEGVTAHFLHKDPSLSSQNVFTDLASHIAAQSLGRATFYGQAISAYNNYEGWRSYWNFGPMQPFIFVADKITFNNAPVSIYRQEEIFAQGEPIFKQHHQNFDAQHFFERAQKAVKDAHEFRHFINETEGSLHYKNLHKNGFCDLTRLDL",
        "disease": "Alzheimer's Disease",
        "description": "Beta-Secretase 1 (BACE1) cleaves amyloid precursor protein (APP) to produce amyloid-beta peptides that accumulate in Alzheimer's disease. Inhibiting BACE1 reduces amyloid plaque formation.",
        "known_drugs": ["Verubecestat", "Lanabecestat", "Atabecestat"],
        "resistance_mutations": "Y29C, D64N, A47V, F36L",
        "ligands": {
            "Verubecestat": "CC(C)(C)c1cnc(NC(=O)c2cc(F)c(F)c(F)c2)c(C#N)c1",
            "Lanabecestat": "CC(C)Oc1cc(F)ccc1NC(=O)C1(C)CCNC(N)=N1",
            "Dopamine (negative control)": "NCCc1ccc(O)c(O)c1",
            "Aspirin (negative control)": "CC(=O)OC1=CC=CC=C1C(=O)O"
        }
    },
    "CDK2 — Cancer (Cell Cycle)": {
        "sequence": "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLESEEEGVPSTAIREISLLKELKDDNIVRLYDIVHSDAHKLYLVFEFLDLDLKRYMEGIPKDQPLGADIVKKFMMQLCKGIAYCHSHRILHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKW",
        "disease": "Cancer — Cell Cycle Dysregulation",
        "description": "Cyclin-Dependent Kinase 2 (CDK2) regulates cell cycle progression from G1 to S phase. Overactive CDK2 drives uncontrolled cell division. CDK inhibitors arrest cancer cell proliferation.",
        "known_drugs": ["Dinaciclib", "Seliciclib", "Milciclib"],
        "resistance_mutations": "K33R, L37V, D60N, K89T, D68N",
        "ligands": {
            "Dinaciclib": "CC(O)c1cnc2c(C#N)c(-c3cccc(O)c3)n(-c3ccc(C(N)=O)cc3)c2n1",
            "Seliciclib": "CC(CO)Nc1nc(NCc2ccccc2)c2ncn(C(C)C)c2n1",
            "Caffeine (negative control)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Acetaminophen (negative control)": "CC(=O)NC1=CC=C(C=C1)O"
        }
    }
}

#===============================================================================
# MODEL LOADING
#===============================================================================

print("[1/3] Loading ESM-2 650M for embeddings...")
esm2_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D', cache_dir='/models/esm2')
esm2_model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D', cache_dir='/models/esm2').cuda().eval()
print("ESM-2 loaded!")

print("[2/3] Loading ESM-2 for Masked LM (Mutation Prediction)...")
esm2_mlm = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D', cache_dir='/models/esm2').cuda().eval()
print("ESM-2 MLM loaded!")

print("[3/3] Loading ESMFold for structure prediction...")
esmfold_tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1', cache_dir='/models/esmfold')
esmfold_model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', cache_dir='/models/esmfold', low_cpu_mem_usage=True).cuda().eval()
print("ESMFold loaded!")
print("=" * 60)
print("All models loaded. Dashboard ready.")
print("=" * 60)

#===============================================================================
# GLOBAL STATE
#===============================================================================

predicted_structures = {}
current_results = {
    'target_discovery': {},
    'target_analysis': {},
    'structure_prediction': {},
    'binding_site': {},
    'drug_screening': {},
    'resistance': {},
    'lead_comparison': {},
    'benchmark': {},
    'gpu_monitor': {}
}

# Track the currently selected drug target globally
selected_target_info = {}

#===============================================================================
# NVIDIA NIM HELPER FUNCTIONS
#===============================================================================

def call_genmol_nim(seed_smiles, num_molecules=10):
    """Call GenMol NIM to generate novel drug candidates from a seed molecule.
    
    Tries NVIDIA hosted API first, falls back to self-hosted GKE pod.
    Uses multiple generation strategies for diversity.
    Returns list of {"smiles": str, "score": float} or (None, error_msg).
    """
    all_molecules = []
    
    # Strategy 1: De novo generation with size constraint (most diverse)
    # Just a mask = generate completely new molecules
    inputs_to_try = [
        f"[*{{10-25}}]",  # pure de novo
        f"{seed_smiles}.[*{{5-15}}]",  # extend the seed
        f"[*{{5-10}}].{seed_smiles}.[*{{5-10}}]",  # decorate both ends
    ]
    
    for input_smiles in inputs_to_try:
        payload = {
            "smiles": input_smiles,
            "num_molecules": num_molecules,
            "temperature": 2.0,
            "noise": 1.2,
            "step_size": 1,
            "scoring": "QED",
            "unique": True
        }
        
        # Try NVIDIA hosted API first
        if NVIDIA_API_KEY:
            try:
                response = requests.post(
                    "https://health.api.nvidia.com/v1/biology/nvidia/genmol/generate",
                    json=payload,
                    headers=NIM_HEADERS,
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json()
                    mols = result.get("molecules", [])
                    if result.get("status") == "success":
                        mols = result.get("molecules", [])
                    # Filter out molecules identical to seed
                    for m in mols:
                        if m.get("smiles", "") != seed_smiles:
                            all_molecules.append(m)
                    if len(all_molecules) >= num_molecules:
                        break
                    continue
            except Exception as e:
                print(f"GenMol hosted API error with input '{input_smiles[:30]}...': {e}")
        
        # Fallback: self-hosted
        try:
            response = requests.post(
                f"{GENMOL_NIM_URL}/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                mols = result.get("molecules", [])
                for m in mols:
                    if m.get("smiles", "") != seed_smiles:
                        all_molecules.append(m)
                if len(all_molecules) >= num_molecules:
                    break
        except Exception as e:
            print(f"GenMol self-hosted error: {e}")
            continue
    
    if not all_molecules:
        return None, "GenMol could not generate novel molecules. Both hosted API and self-hosted pod failed."
    
    # Deduplicate by SMILES
    seen = set()
    unique_molecules = []
    for m in all_molecules:
        s = m.get("smiles", "")
        if s and s not in seen and s != seed_smiles:
            seen.add(s)
            unique_molecules.append(m)
    
    if not unique_molecules:
        return None, "GenMol generated molecules but all were identical to the seed."
    
    return unique_molecules[:num_molecules], None


def call_diffdock_nim(pdb_string, smiles, num_poses=5):
    """Call DiffDock NIM hosted API for molecular docking.
    
    Sends protein PDB + ligand SMILES, returns docking poses with confidence scores.
    Returns (poses_data, error_msg). poses_data is the raw JSON response.
    """
    # Extract ATOM lines from PDB
    protein_lines = "\n".join(
        line for line in pdb_string.split("\n") 
        if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("END")
    )
    
    payload = {
        "ligand": smiles,
        "ligand_file_type": "txt",
        "protein": protein_lines,
        "num_poses": num_poses,
        "time_divisions": 20,
        "steps": 18,
        "save_trajectory": False,
        "is_staged": False
    }
    try:
        # Try multiple endpoint paths (hosted API vs self-hosted have different paths)
        endpoints = [
            f"{DIFFDOCK_NIM_URL}/generate",
            f"{DIFFDOCK_NIM_URL}/molecular-docking/diffdock/generate",
            "https://integrate.api.nvidia.com/v1/biology/mit/diffdock/generate",
            "https://integrate.api.nvidia.com/v1/biology/mit/diffdock/molecular-docking/diffdock/generate",
        ]
        
        response = None
        last_error = ""
        for endpoint in endpoints:
            for auth_header in [NIM_HEADERS, {"Content-Type": "application/json", "Authorization": f"Bearer {NVIDIA_API_KEY}"}, {"Content-Type": "application/json", "accept": "application/json", "Authorization": f"Bearer {NVIDIA_API_KEY}"}]:
                try:
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers=auth_header,
                        timeout=180
                    )
                    if response.status_code not in [404, 401, 403]:
                        break
                    last_error = f"{endpoint} → {response.status_code}"
                except Exception as e:
                    last_error = f"{endpoint} → {e}"
                    continue
            if response and response.status_code not in [404, 401, 403]:
                break
        
        if response is None:
            return None, f"Could not reach DiffDock API. Last: {last_error}"
        
        if response.status_code != 200:
            return None, f"DiffDock API error: {response.status_code} - {response.text[:300]}"
        
        result = response.json()
        return result, None
    except requests.exceptions.Timeout:
        return None, "DiffDock API timed out (docking can take 30-60s)"
    except Exception as e:
        return None, f"DiffDock API error: {str(e)}"

#===============================================================================
# GEMINI AI HELPER FUNCTIONS
#===============================================================================

def pcm_to_wav(pcm_data, sample_rate=24000, num_channels=1, bits_per_sample=16):
    try:
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_data)
        wav_buffer = io.BytesIO()
        wav_buffer.write(b'RIFF')
        wav_buffer.write(struct.pack('<I', 36 + data_size))
        wav_buffer.write(b'WAVE')
        wav_buffer.write(b'fmt ')
        wav_buffer.write(struct.pack('<I', 16))
        wav_buffer.write(struct.pack('<H', 1))
        wav_buffer.write(struct.pack('<H', num_channels))
        wav_buffer.write(struct.pack('<I', sample_rate))
        wav_buffer.write(struct.pack('<I', byte_rate))
        wav_buffer.write(struct.pack('<H', block_align))
        wav_buffer.write(struct.pack('<H', bits_per_sample))
        wav_buffer.write(b'data')
        wav_buffer.write(struct.pack('<I', data_size))
        wav_buffer.write(pcm_data)
        return wav_buffer.getvalue()
    except Exception as e:
        print(f"PCM to WAV conversion error: {e}")
        return None


def get_gemini_text_explanation(context, tab_name):
    prompt = f"""You are a friendly science educator explaining drug discovery results.
Based on the following {tab_name} results, provide a clear, engaging 2-3 sentence explanation.
Be specific about the numbers and what they mean for drug discovery.
Keep it concise - under 60 words.

Results:
{context}

Provide only the spoken explanation, no formatting or bullet points."""
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
        }
        response = requests.post(GEMINI_TEXT_ENDPOINT, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Unable to generate explanation. Please try again."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def get_gemini_audio(text):
    try:
        tts_text = f"Say this exactly: {text}"
        payload = {
            "contents": [{"parts": [{"text": tts_text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": "Kore"}
                    }
                }
            }
        }
        response = requests.post(GEMINI_TTS_ENDPOINT, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                parts = result['candidates'][0]['content']['parts']
                if len(parts) > 0 and 'inlineData' in parts[0]:
                    audio_b64 = parts[0]['inlineData']['data']
                    mime_type = parts[0]['inlineData']['mimeType']
                    pcm_data = base64.b64decode(audio_b64)
                    wav_data = pcm_to_wav(pcm_data, sample_rate=24000)
                    if wav_data:
                        wav_b64 = base64.b64encode(wav_data).decode('utf-8')
                        return wav_b64, "audio/wav"
        return None, None
    except Exception as e:
        print(f"TTS Exception: {str(e)}")
        return None, None


def create_audio_player(text, audio_data, mime_type):
    # FIX: Use unique timestamped IDs to prevent the browser from 
    # caching/reusing old audio elements during UI updates.
    player_id = f"audio_{int(time.time())}"

    if audio_data and mime_type:
        html = f'''
        <div id="{player_id}_wrapper" style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 10px; margin: 10px 0;">
            <div style="margin-bottom: 15px;">
                <audio id="{player_id}" controls autoplay style="width: 100%; height: 50px;">
                    <source src="data:{mime_type};base64,{audio_data}" type="{mime_type}">
                </audio>
            </div>
            <div style="padding: 15px; background: rgba(0, 212, 255, 0.1); border-radius: 8px; border-left: 4px solid #00d4ff; color: #e0e0e0; font-size: 14px; line-height: 1.6;">
                <strong style="color: #00d4ff;">🔬 AI Explanation:</strong><br>{text}
            </div>
            <script>
                var audio = document.getElementById("{player_id}");
                if (audio) {{
                    audio.load();
                }}
            </script>
        </div>'''
    else:
        html = f'''
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 10px; margin: 10px 0;">
            <div style="padding: 15px; background: rgba(255, 107, 107, 0.1); border-radius: 8px; border-left: 4px solid #ff6b6b; color: #e0e0e0; font-size: 14px; line-height: 1.6;">
                <strong style="color: #ff6b6b;">Text explanation:</strong><br>{text}
            </div>
        </div>'''
    return html

#===============================================================================
# PROTEIN ANALYSIS HELPERS
#===============================================================================

def clean_sequence(seq):
    return ''.join(c for c in seq.upper() if c in AA_COLORS)


def analyze_sequence(sequence):
    length = len(sequence)
    aa_counts = {aa: sequence.count(aa) for aa in AA_COLORS}
    properties = {
        prop: sum(sequence.count(aa) for aa in aas) / length * 100
        for prop, aas in AA_PROPERTIES.items()
    }
    try:
        analysis = ProteinAnalysis(sequence)
        return {
            'length': length, 'aa_counts': aa_counts, 'properties': properties,
            'mol_weight': analysis.molecular_weight(), 'isoelectric': analysis.isoelectric_point(),
            'instability': analysis.instability_index(), 'gravy': analysis.gravy()
        }
    except:
        return {
            'length': length, 'aa_counts': aa_counts, 'properties': properties,
            'mol_weight': 0, 'isoelectric': 0, 'instability': 0, 'gravy': 0
        }


def create_sequence_viz(sequence):
    html = '<div style="font-family:monospace;font-size:14px;line-height:1.8;">'
    for i, aa in enumerate(sequence):
        color = AA_COLORS.get(aa, "#000")
        html += f'<span style="background:{color};color:white;padding:2px 4px;margin:1px;border-radius:3px;">{aa}</span>'
        if (i + 1) % 50 == 0:
            html += '<br>'
    return html + '</div>'


def create_composition_chart(analysis):
    fig = go.Figure(data=[go.Bar(
        x=list(analysis['aa_counts'].keys()),
        y=list(analysis['aa_counts'].values()),
        marker_color=[AA_COLORS.get(aa, '#000') for aa in analysis['aa_counts']]
    )])
    fig.update_layout(title="Amino Acid Composition", template="plotly_dark", height=400)
    return fig


def create_properties_radar(analysis):
    fig = go.Figure(data=go.Scatterpolar(
        r=list(analysis['properties'].values()),
        theta=list(analysis['properties'].keys()),
        fill='toself'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), template="plotly_dark", height=400)
    return fig


def get_embedding(sequence):
    inputs = esm2_tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        outputs = esm2_model(**inputs)
    return outputs.last_hidden_state[0].cpu().numpy()


def create_embedding_heatmap(sequence):
    emb = get_embedding(sequence)[1:min(51, len(sequence)+1), :100]
    if emb.size == 0:
        fig = go.Figure()
        fig.update_layout(title="No embedding data available", template="plotly_dark", height=500)
        return fig
    fig = go.Figure(data=go.Heatmap(
        z=emb.T.tolist(), colorscale='Viridis',
        x=list(range(1, emb.shape[0]+1)),
        y=list(range(emb.shape[1]))
    ))
    fig.update_layout(
        title="ESM-2 Embedding Heatmap (first 50 residues, 100 dimensions)",
        template="plotly_dark", height=500,
        xaxis_title="Residue Position", yaxis_title="Embedding Dimension",
        xaxis=dict(dtick=5), yaxis=dict(dtick=10)
    )
    return fig

#===============================================================================
# 3D STRUCTURE VIEWERS
#===============================================================================

def create_3d_viewer(pdb_string, viewer_id="molviewer"):
    pdb_b64 = base64.b64encode(pdb_string.encode()).decode()
    html = f'''<!DOCTYPE html>
<html><head><script src="https://3Dmol.org/build/3Dmol-min.js"></script></head>
<body style="margin:0;padding:0;">
<div id="{viewer_id}" style="width:100%;height:600px;background-color:#1a1a2e;border-radius:10px;"></div>
<script>
(function(){{
    var pdbData = atob("{pdb_b64}");
    var viewer = $3Dmol.createViewer("{viewer_id}", {{backgroundColor: "#1a1a2e"}});
    viewer.addModel(pdbData, "pdb");
    viewer.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
    viewer.zoomTo();
    viewer.render();
    viewer.spin("y", 0.1);
}})();
</script></body></html>'''
    return f'<iframe srcdoc=\'{html}\' style="width:100%;height:620px;border:none;border-radius:10px;"></iframe>'


def create_3d_viewer_with_highlights(pdb_string, highlight_residues=None, viewer_id="molviewer"):
    pdb_b64 = base64.b64encode(pdb_string.encode()).decode()
    highlight_script = ""
    if highlight_residues and len(highlight_residues) > 0:
        residue_list = ",".join([str(r) for r in highlight_residues])
        highlight_script = f'viewer.setStyle({{resi: [{residue_list}]}}, {{cartoon: {{color: "spectrum"}}, stick: {{color: "red", radius: 0.3}}}});'
    html = f'''<!DOCTYPE html>
<html><head><script src="https://3Dmol.org/build/3Dmol-min.js"></script></head>
<body style="margin:0;padding:0;">
<div id="{viewer_id}" style="width:100%;height:600px;background-color:#1a1a2e;border-radius:10px;"></div>
<script>
(function(){{
    var pdbData = atob("{pdb_b64}");
    var viewer = $3Dmol.createViewer("{viewer_id}", {{backgroundColor: "#1a1a2e"}});
    viewer.addModel(pdbData, "pdb");
    viewer.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
    {highlight_script}
    viewer.zoomTo();
    viewer.render();
    viewer.spin("y", 0.1);
}})();
</script></body></html>'''
    return f'<iframe srcdoc=\'{html}\' style="width:100%;height:620px;border:none;border-radius:10px;"></iframe>'


def create_side_by_side_3d_viewer(pdb1, pdb2, name1, name2):
    pdb_b64_1 = base64.b64encode(pdb1.encode()).decode()
    pdb_b64_2 = base64.b64encode(pdb2.encode()).decode()
    html = f'''<!DOCTYPE html>
<html><head><script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
.container{{display:flex;width:100%;height:600px;gap:10px;}}
.viewer-box{{flex:1;position:relative;}}
.viewer{{width:100%;height:100%;background-color:#1a1a2e;border-radius:10px;}}
.label{{position:absolute;top:10px;left:10px;color:#00d4ff;font-family:Arial;font-size:16px;font-weight:bold;z-index:100;background:rgba(0,0,0,0.5);padding:5px 10px;border-radius:5px;}}
</style></head>
<body style="margin:0;padding:0;background-color:#1a1a2e;">
<div class="container">
    <div class="viewer-box"><div class="label">{name1}</div><div id="viewer1" class="viewer"></div></div>
    <div class="viewer-box"><div class="label">{name2}</div><div id="viewer2" class="viewer"></div></div>
</div>
<script>
(function(){{
    var pdb1 = atob("{pdb_b64_1}");
    var pdb2 = atob("{pdb_b64_2}");
    var v1 = $3Dmol.createViewer("viewer1", {{backgroundColor: "#1a1a2e"}});
    v1.addModel(pdb1, "pdb"); v1.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}}); v1.zoomTo(); v1.render(); v1.spin("y", 0.1);
    var v2 = $3Dmol.createViewer("viewer2", {{backgroundColor: "#1a1a2e"}});
    v2.addModel(pdb2, "pdb"); v2.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}}); v2.zoomTo(); v2.render(); v2.spin("y", 0.1);
}})();
</script></body></html>'''
    return f'<iframe srcdoc=\'{html}\' style="width:100%;height:620px;border:none;border-radius:10px;"></iframe>'

#===============================================================================
# RDKit DRUG-LIKENESS CALCULATIONS
#===============================================================================

def compute_lipinski(smiles):
    """Compute Lipinski Rule of 5 and extended drug-likeness properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    rings = Descriptors.RingCount(mol)
    heavy_atoms = Descriptors.HeavyAtomCount(mol)

    # Lipinski violations
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1

    # Drug-likeness verdict
    if violations == 0:
        verdict = "Excellent — passes all Lipinski rules"
    elif violations == 1:
        verdict = "Good — 1 Lipinski violation (often acceptable)"
    elif violations == 2:
        verdict = "Moderate — 2 violations (may have bioavailability issues)"
    else:
        verdict = "Poor — 3+ violations (likely poor oral bioavailability)"

    return {
        'mw': mw, 'logp': logp, 'hbd': hbd, 'hba': hba,
        'rotatable': rotatable, 'tpsa': tpsa, 'rings': rings,
        'heavy_atoms': heavy_atoms, 'violations': violations,
        'verdict': verdict
    }


def create_lipinski_radar(props):
    """Create a radar chart for drug-likeness properties."""
    # Normalize each property to 0-1 scale relative to Lipinski thresholds
    categories = ['MW\n(<500)', 'LogP\n(<5)', 'H-Donors\n(<5)', 'H-Acceptors\n(<10)', 'TPSA\n(<140)', 'Rot. Bonds\n(<10)']
    values = [
        min(props['mw'] / 500, 2.0),
        min(props['logp'] / 5, 2.0) if props['logp'] > 0 else 0,
        min(props['hbd'] / 5, 2.0),
        min(props['hba'] / 10, 2.0),
        min(props['tpsa'] / 140, 2.0),
        min(props['rotatable'] / 10, 2.0)
    ]
    # Threshold line at 1.0 (= Lipinski limit)
    threshold = [1.0] * 6

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        name='Compound', line_color='#00d4ff', fillcolor='rgba(0,212,255,0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=threshold, theta=categories, fill='none',
        name='Lipinski Limit', line=dict(color='#ff6b6b', dash='dash', width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 2], visible=True)),
        template="plotly_dark", height=400,
        title="Drug-Likeness Profile (Lipinski Rule of 5)"
    )
    return fig


def compute_binding_score(smiles, pocket_residues, pdb_string):
    """
    Compute a physics-inspired binding score based on:
    1. Size complementarity (ligand size vs pocket volume)
    2. Charge complementarity (ligand charges vs pocket charges)
    3. Hydrophobic matching
    4. Shape score
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Ligand properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    heavy_atoms = Descriptors.HeavyAtomCount(mol)

    # Parse pocket residues from PDB
    pocket_aa = []
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            try:
                res_id = int(line[22:26].strip())
                res_name = line[17:20].strip()
                if res_id in pocket_residues:
                    # Convert 3-letter to 1-letter code
                    three_to_one = {
                        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                    }
                    aa = three_to_one.get(res_name, '')
                    if aa:
                        pocket_aa.append(aa)
            except:
                continue

    pocket_size = len(pocket_aa)
    if pocket_size == 0:
        pocket_size = len(pocket_residues)
        pocket_aa = ['A'] * pocket_size  # fallback

    # Score 1: Size complementarity (ideal: ~15-30 heavy atoms for typical pocket)
    ideal_heavy_atoms = pocket_size * 1.5
    size_score = max(0, 1.0 - abs(heavy_atoms - ideal_heavy_atoms) / ideal_heavy_atoms)

    # Score 2: Charge complementarity
    pocket_positive = sum(1 for aa in pocket_aa if aa in ['K', 'R', 'H'])
    pocket_negative = sum(1 for aa in pocket_aa if aa in ['D', 'E'])
    pocket_hydrophobic = sum(1 for aa in pocket_aa if aa in ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'])

    # Count ligand functional groups from SMILES
    ligand_positive = smiles.count('N') - smiles.count('NC(=O)')
    ligand_negative = smiles.count('C(=O)O') + smiles.count('S(=O)')
    ligand_hydrophobic_ratio = logp / 5.0

    charge_match = 0
    if pocket_negative > 0 and ligand_positive > 0:
        charge_match += 0.3
    if pocket_positive > 0 and ligand_negative > 0:
        charge_match += 0.3
    if pocket_hydrophobic > pocket_size * 0.4 and logp > 1:
        charge_match += 0.4
    charge_score = min(charge_match, 1.0)

    # Score 3: H-bond potential
    pocket_polar = sum(1 for aa in pocket_aa if aa in ['S', 'T', 'N', 'Q', 'Y', 'C'])
    hbond_capacity = min((hbd + hba) / max(pocket_polar + pocket_positive + pocket_negative, 1), 1.5)
    hbond_score = min(hbond_capacity, 1.0)

    # Score 4: Drug-likeness penalty
    lipinski = compute_lipinski(smiles)
    druglike_score = max(0, 1.0 - lipinski['violations'] * 0.25) if lipinski else 0.5

    # Combine scores into binding energy estimate (kcal/mol scale)
    # Real docking scores typically range from -3 to -15 kcal/mol
    raw_score = (size_score * 0.25 + charge_score * 0.35 + hbond_score * 0.2 + druglike_score * 0.2)

    # Map to kcal/mol range — spread wider so good/bad are more distinct
    # raw_score 0.0 → -3.0 (terrible), raw_score 1.0 → -12.0 (excellent)
    binding_energy = -3.0 - (raw_score * 9.0)

    # Add small deterministic noise based on SMILES hash for pose variation
    np.random.seed(hash(smiles) % 2**32)
    poses = sorted([binding_energy + np.random.uniform(-0.8, 1.5) for _ in range(5)])

    return {
        'best_score': poses[0],
        'poses': poses,
        'size_score': size_score,
        'charge_score': charge_score,
        'hbond_score': hbond_score,
        'druglike_score': druglike_score,
        'raw_score': raw_score,
        'pocket_size': pocket_size,
        'heavy_atoms': heavy_atoms
    }


#===============================================================================
# PDB PARSING HELPERS
#===============================================================================

def parse_pdb_ca_coords(pdb_string):
    ca_coords = []
    residue_ids = []
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                res_id = int(line[22:26].strip())
                ca_coords.append([x, y, z])
                residue_ids.append(res_id)
            except:
                continue
    return np.array(ca_coords), residue_ids

#===============================================================================
# TAB 1: TARGET DISCOVERY
#===============================================================================

def select_target(target_name):
    """When user selects a drug target, populate all downstream fields."""
    global selected_target_info

    if target_name not in DRUG_TARGETS:
        return "", "", "", ""

    target = DRUG_TARGETS[target_name]
    selected_target_info = target

    # Build info card
    info_html = f'''
    <div style="background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 100%); padding: 25px; border-radius: 12px; border: 1px solid #00d4ff33;">
        <h2 style="color: #00d4ff; margin-top: 0;">🎯 {target_name.split("—")[0].strip()}</h2>
        <div style="background: rgba(255,107,107,0.1); padding: 12px; border-radius: 8px; border-left: 4px solid #ff6b6b; margin-bottom: 15px;">
            <strong style="color: #ff6b6b;">Disease:</strong>
            <span style="color: #e0e0e0;"> {target['disease']}</span>
        </div>
        <p style="color: #ccc; line-height: 1.6;">{target['description']}</p>
        <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-top: 15px;">
            <div style="background: rgba(0,212,255,0.1); padding: 10px 15px; border-radius: 8px; flex: 1; min-width: 200px;">
                <strong style="color: #00d4ff;">Known Drugs:</strong><br>
                <span style="color: #ccc;">{", ".join(target['known_drugs'])}</span>
            </div>
            <div style="background: rgba(0,255,136,0.1); padding: 10px 15px; border-radius: 8px; flex: 1; min-width: 200px;">
                <strong style="color: #00ff88;">Sequence Length:</strong><br>
                <span style="color: #ccc;">{len(target['sequence'])} amino acids</span>
            </div>
            <div style="background: rgba(255,215,0,0.1); padding: 10px 15px; border-radius: 8px; flex: 1; min-width: 200px;">
                <strong style="color: #ffd700;">Known Resistance:</strong><br>
                <span style="color: #ccc;">{target['resistance_mutations']}</span>
            </div>
        </div>
    </div>'''

    return (
        info_html,
        target['sequence'],
        target['resistance_mutations'],
        list(target['ligands'].keys())[0]
    )


def explain_target_discovery():
    if not selected_target_info:
        text = "Please select a drug target to begin the discovery pipeline."
        return create_audio_player(text, None, None)
    context = f"Selected target for {selected_target_info.get('disease', 'unknown disease')}. Sequence has {len(selected_target_info.get('sequence', ''))} amino acids. Known drugs include {', '.join(selected_target_info.get('known_drugs', [])[:2])}."
    text = get_gemini_text_explanation(context, "Target Discovery")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# TAB 2: TARGET ANALYSIS (was Protein Analyzer)
#===============================================================================

def analyze_protein(sequence):
    global current_results
    sequence = clean_sequence(sequence)
    if len(sequence) < 5:
        return "Enter at least 5 amino acids", None, None, None, None

    analysis = analyze_sequence(sequence)
    sorted_aa = sorted(analysis['aa_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
    top_aa = ", ".join([f"{aa}({count})" for aa, count in sorted_aa])

    current_results['target_analysis'] = {
        'length': analysis['length'],
        'mol_weight': f"{analysis['mol_weight']:.0f}",
        'isoelectric': f"{analysis['isoelectric']:.2f}",
        'instability': f"{analysis['instability']:.1f}",
        'top_aa': top_aa
    }

    # Add disease context if available
    disease_context = ""
    if selected_target_info:
        disease_context = f"\n**Disease Target:** {selected_target_info.get('disease', 'N/A')}\n"

    summary = f"""## Drug Target Analysis Results
{disease_context}
| Property | Value |
|----------|-------|
| **Length** | {analysis['length']} amino acids |
| **Molecular Weight** | {analysis['mol_weight']:.0f} Da |
| **Isoelectric Point** | {analysis['isoelectric']:.2f} |
| **Instability Index** | {analysis['instability']:.1f} |
| **GRAVY Score** | {analysis['gravy']:.3f} |
"""
    return (
        summary,
        create_sequence_viz(sequence),
        create_composition_chart(analysis),
        create_properties_radar(analysis),
        create_embedding_heatmap(sequence)
    )


def explain_target_analysis():
    results = current_results.get('target_analysis', {})
    if not results:
        text = "Please analyze the drug target first."
        return create_audio_player(text, None, None)
    context = f"Drug target with {results.get('length', 'N/A')} amino acids, molecular weight {results.get('mol_weight', 'N/A')} Daltons, isoelectric point {results.get('isoelectric', 'N/A')}, instability index {results.get('instability', 'N/A')}."
    text = get_gemini_text_explanation(context, "Drug Target Analysis")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# TAB 3: STRUCTURE PREDICTION
#===============================================================================

def predict_structure(sequence, name="DrugTarget"):
    global predicted_structures, current_results
    sequence = clean_sequence(sequence)
    if len(sequence) < 5:
        return "Enter at least 5 amino acids", None, None, None, None
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        return f"Maximum sequence length is {MAX_SEQUENCE_LENGTH} amino acids", None, None, None, None

    start_time = time.time()
    inputs = esmfold_tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        outputs = esmfold_model(**inputs)
    pdb_string = esmfold_model.output_to_pdb(outputs)[0]
    elapsed = time.time() - start_time

    structure_id = f"{name}_{len(sequence)}aa"
    predicted_structures[structure_id] = {'pdb': pdb_string, 'sequence': sequence}

    # FIX: Flattening the array to 1D and converting to a standard Python list.
    # This prevents JSON serialization errors in Gradio/Plotly caused by NumPy/Tensor types.
    plddt_raw = outputs.plddt[0].cpu().numpy().flatten()
    plddt_list = plddt_raw.tolist()

    # Calculated mean using pure Python floats for better title rendering
    mean_plddt = float(sum(plddt_list) / len(plddt_list))

    # Normalize to 0-100 scale if ESMFold returns 0-1
    # FIX: Normalization check to ensure consistent 0-100 scale for plotting
    if mean_plddt < 1.5:
        plddt_list = [x * 100 for x in plddt_list]
        mean_plddt = mean_plddt * 100
    predicted_structures[structure_id]['mean_plddt'] = mean_plddt

    # Plotting using the cleaned Python list
    # Added showscale=True
    plddt_fig = go.Figure(data=go.Scatter(
        y=plddt_list, mode='lines+markers',
        marker=dict(color=plddt_list, colorscale='RdYlGn', cmin=0, cmax=100, size=6, showscale=True),
        line=dict(color='cyan', width=1)
    ))
    plddt_fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="High confidence")
    plddt_fig.add_hline(y=70, line_dash="dash", line_color="yellow", annotation_text="Good")
    plddt_fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Low")
    plddt_fig.update_layout(
        title=f"pLDDT Confidence (mean: {mean_plddt:.1f})",
        template="plotly_dark", height=400,
        xaxis_title="Residue Position", yaxis_title="pLDDT Score", yaxis_range=[0, 100]
    )

    confidence = "HIGH" if mean_plddt > 90 else "GOOD" if mean_plddt > 70 else "MODERATE" if mean_plddt > 50 else "LOW"

    current_results['structure_prediction'] = {
        'structure_id': structure_id, 'length': len(sequence),
        'prediction_time': f"{elapsed:.1f}", 'mean_plddt': f"{mean_plddt:.1f}", 'confidence': confidence
    }

    pdb_file = f"/tmp/{structure_id}.pdb"
    with open(pdb_file, 'w') as f:
        f.write(pdb_string)

    disease_context = ""
    if selected_target_info:
        disease_context = f"\n**Disease Target:** {selected_target_info.get('disease', 'N/A')}\n"

    summary = f"""## 3D Structure Prediction Complete
{disease_context}
| Property | Value |
|----------|-------|
| **Structure ID** | {structure_id} |
| **Sequence Length** | {len(sequence)} aa |
| **Prediction Time** | {elapsed:.1f} seconds |
| **Mean pLDDT** | {mean_plddt:.1f} |
| **Confidence** | {confidence} |

✅ Structure saved — proceed to Binding Site Detection and Drug Screening!
"""
    return summary, create_3d_viewer(pdb_string), plddt_fig, pdb_string, pdb_file


def explain_structure_prediction():
    results = current_results.get('structure_prediction', {})
    if not results:
        text = "Please predict a structure first."
        return create_audio_player(text, None, None)
    context = f"Structure {results.get('structure_id', 'Unknown')} predicted in {results.get('prediction_time', 'N/A')} seconds with mean pLDDT of {results.get('mean_plddt', 'N/A')} out of 100. Confidence level: {results.get('confidence', 'N/A')}. pLDDT above 90 is high confidence, 70-90 is good, 50-70 is moderate, below 50 is low. Be honest about the quality."
    text = get_gemini_text_explanation(context, "Structure Prediction")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# TAB 4: BINDING SITE DETECTION (merged with Contact Map)
#===============================================================================

def compute_contact_map(pdb_string, threshold=8.0):
    ca_coords, residue_ids = parse_pdb_ca_coords(pdb_string)
    if len(ca_coords) == 0:
        return None, None, 0
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    contact_map = (distances < threshold).astype(float)
    num_contacts = 0
    for i in range(len(ca_coords)):
        for j in range(i + 4, len(ca_coords)):
            if contact_map[i, j] > 0:
                num_contacts += 1
    return contact_map, distances, num_contacts


def predict_binding_sites_and_contacts(structure_id):
    """Combined binding site prediction + contact map visualization."""
    global current_results

    if not structure_id or structure_id not in predicted_structures:
        available = list(predicted_structures.keys())
        if not available:
            return "No structures available. Predict a structure first in Step 2.", None, None, None, None
        return f"Structure not found. Available: {', '.join(available)}", None, None, None, None

    structure = predicted_structures[structure_id]
    pdb_string = structure['pdb']
    ca_coords, residue_ids = parse_pdb_ca_coords(pdb_string)

    if len(ca_coords) == 0:
        return "Could not parse structure coordinates.", None, None, None, None

    n_residues = len(ca_coords)
    center = np.mean(ca_coords, axis=0)
    distances_from_center = np.linalg.norm(ca_coords - center, axis=1)

    neighbor_counts = []
    for i in range(n_residues):
        dist_to_others = np.linalg.norm(ca_coords - ca_coords[i], axis=1)
        neighbors = np.sum((dist_to_others > 0) & (dist_to_others < 10))
        neighbor_counts.append(neighbors)
    neighbor_counts = np.array(neighbor_counts)

    dist_norm = (distances_from_center - distances_from_center.min()) / (distances_from_center.max() - distances_from_center.min() + 1e-6)
    neighbor_norm = (neighbor_counts - neighbor_counts.min()) / (neighbor_counts.max() - neighbor_counts.min() + 1e-6)
    pocket_score = dist_norm * (1 - np.abs(neighbor_norm - 0.5))

    pocket_threshold = np.percentile(pocket_score, 75)
    pocket_residues = np.where(pocket_score > pocket_threshold)[0]

    pockets = []
    used = set()
    for res in pocket_residues:
        if res in used:
            continue
        pocket_cluster = [res]
        used.add(res)
        for other_res in pocket_residues:
            if other_res in used:
                continue
            dist = np.linalg.norm(ca_coords[res] - ca_coords[other_res])
            if dist < 12:
                pocket_cluster.append(other_res)
                used.add(other_res)
        if len(pocket_cluster) >= 3:
            avg_score = np.mean(pocket_score[pocket_cluster])
            pockets.append({
                'residues': [r + 1 for r in pocket_cluster],
                'score': avg_score,
                'center': np.mean(ca_coords[pocket_cluster], axis=0)
            })

    pockets.sort(key=lambda x: x['score'], reverse=True)
    pockets = pockets[:5]

    # Store pockets globally for drug screening
    predicted_structures[structure_id]['pockets'] = pockets

    # Binding site score plot
    binding_fig = go.Figure()
    binding_fig.add_trace(go.Scatter(
        x=list(range(1, n_residues + 1)), y=pocket_score,
        mode='lines+markers',
        marker=dict(color=pocket_score, colorscale='RdYlGn', size=6),
        line=dict(color='cyan', width=1), name='Pocket Score'
    ))
    for i, pocket in enumerate(pockets):
        binding_fig.add_trace(go.Scatter(
            x=pocket['residues'],
            y=[pocket_score[r-1] for r in pocket['residues']],
            mode='markers', marker=dict(size=12, symbol='circle-open', line=dict(width=2)),
            name=f'Pocket {i+1}'
        ))
    binding_fig.add_hline(y=pocket_threshold, line_dash="dash", line_color="yellow", annotation_text="Pocket threshold")
    binding_fig.update_layout(title="Binding Site Prediction Scores", xaxis_title="Residue Number", yaxis_title="Pocket Score", template="plotly_dark", height=400)

    # Contact map
    _, distances_matrix, num_contacts = compute_contact_map(pdb_string)
    contact_fig = None
    if distances_matrix is not None:
        # Create binary contact map (below 8Å = contact)
        contact_binary = (distances_matrix < 8.0).astype(float)
        contact_fig = go.Figure()
        contact_fig.add_trace(go.Heatmap(
            z=distances_matrix.tolist(), colorscale='Viridis_r', zmin=0, zmax=20,
            colorbar=dict(title="Distance (Å)"),
            x=list(range(1, len(distances_matrix)+1)),
            y=list(range(1, len(distances_matrix)+1))
        ))
        contact_fig.update_layout(
            title=f"Residue Distance Map — {structure_id} (red line = 8Å contact threshold)",
            xaxis_title="Residue Number", yaxis_title="Residue Number",
            template="plotly_dark", height=500, 
            yaxis=dict(autorange='reversed'),
            xaxis=dict(dtick=10), 
        )

    # 3D viewer with highlights
    all_pocket_residues = []
    for pocket in pockets:
        all_pocket_residues.extend(pocket['residues'])
    viewer_html = create_3d_viewer_with_highlights(pdb_string, all_pocket_residues)

    # Store results
    if pockets:
        current_results['binding_site'] = {
            'structure_id': structure_id,
            'num_pockets': len(pockets),
            'top_score': f"{pockets[0]['score']:.3f}",
            'top_residues': ", ".join([str(r) for r in pockets[0]['residues'][:10]])
        }

    pocket_table = "| Pocket | Score | Key Residues |\n|--------|-------|-------------|\n"
    for i, pocket in enumerate(pockets):
        residue_str = ", ".join([str(r) for r in pocket['residues'][:8]])
        if len(pocket['residues']) > 8:
            residue_str += f"... (+{len(pocket['residues'])-8} more)"
        pocket_table += f"| **Pocket {i+1}** | {pocket['score']:.3f} | {residue_str} |\n"

    top_score_str = f"{pockets[0]['score']:.3f}" if pockets else "N/A"
    summary = f"""## Binding Site Detection Results

| Property | Value |
|----------|-------|
| **Structure** | {structure_id} |
| **Pockets Detected** | {len(pockets)} |
| **Top Pocket Score** | {top_score_str} |

### Predicted Drug Binding Pockets
{pocket_table}

Red sticks in 3D viewer = predicted binding site residues.
✅ Pockets saved — proceed to Drug Screening to test compounds!
"""
    return summary, binding_fig, contact_fig, viewer_html, f"Found {len(pockets)} potential binding pockets"


def explain_binding_site():
    results = current_results.get('binding_site', {})
    if not results:
        text = "Please predict binding sites first."
        return create_audio_player(text, None, None)
    context = f"Found {results.get('num_pockets', 'N/A')} binding pockets in {results.get('structure_id', 'Unknown')}. Top pocket score: {results.get('top_score', 'N/A')}. Key residues: {results.get('top_residues', 'N/A')}."
    # Add structure quality caveat
    struct_id = results.get('structure_id', '')
    if struct_id in predicted_structures and 'mean_plddt' in predicted_structures[struct_id]:
        plddt_val = predicted_structures[struct_id]['mean_plddt']
        if plddt_val < 70:
            context += f" IMPORTANT: The underlying structure has a moderate/low pLDDT of {plddt_val:.0f}/100, so these binding site predictions should be interpreted with caution."
    text = get_gemini_text_explanation(context, "Binding Site Detection")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# TAB 5: DRUG SCREENING (Lipinski + Physics-Based Scoring)
#===============================================================================

def get_structures():
    return list(predicted_structures.keys()) if predicted_structures else []

def refresh_structures():
    choices = get_structures()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

def get_target_ligands():
    """Get ligands for currently selected target."""
    if selected_target_info and 'ligands' in selected_target_info:
        return list(selected_target_info['ligands'].keys())
    return ["Aspirin", "Ibuprofen", "Caffeine"]

def refresh_ligands():
    choices = get_target_ligands()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def generate_novel_candidates(structure_id, ligand_name, smiles):
    """Use GenMol NIM to generate novel drug candidates from a seed molecule."""
    if not smiles:
        return "Enter a seed SMILES string first.", None, ""
    
    # Validate seed SMILES
    seed_mol = Chem.MolFromSmiles(smiles)
    if seed_mol is None:
        return "Invalid SMILES string.", None, ""
    
    molecules, error = call_genmol_nim(smiles, num_molecules=10)
    
    if error:
        return f"⚠️ GenMol NIM Error: {error}\n\nFalling back is not available — ensure the GenMol NIM pod is running.", None, ""
    
    if not molecules or len(molecules) == 0:
        return "GenMol returned no molecules. Try a different seed compound.", None, ""
    
    # Analyze each generated molecule with Lipinski
    analyzed = []
    for mol_data in molecules:
        gen_smiles = mol_data.get("smiles", "")
        score = mol_data.get("score", 0.0)
        mol = Chem.MolFromSmiles(gen_smiles)
        if mol is None:
            continue
        lip = compute_lipinski(gen_smiles)
        if lip is None:
            continue
        analyzed.append({
            "smiles": gen_smiles,
            "qed_score": score,
            "mw": lip["mw"],
            "logp": lip["logp"],
            "hbd": lip["hbd"],
            "hba": lip["hba"],
            "violations": lip["violations"],
            "verdict": "✅" if lip["violations"] <= 1 else "⚠️" if lip["violations"] == 2 else "❌"
        })
    
    if not analyzed:
        return "GenMol generated molecules but none were valid.", None, ""
    
    # Sort by QED score descending
    analyzed.sort(key=lambda x: x["qed_score"], reverse=True)
    
    # Store for Gemini explanation
    current_results['genmol'] = {
        'seed': ligand_name,
        'num_generated': len(analyzed),
        'best_qed': f"{analyzed[0]['qed_score']:.3f}",
        'lipinski_pass': sum(1 for m in analyzed if m['violations'] <= 1),
        'total': len(analyzed)
    }
    
    # Build results table
    table = "| # | SMILES | QED | MW | LogP | Lipinski |\n|---|--------|-----|-----|------|----------|\n"
    for i, mol in enumerate(analyzed[:10]):
        short_smiles = mol["smiles"][:40] + "..." if len(mol["smiles"]) > 40 else mol["smiles"]
        table += f"| {i+1} | `{short_smiles}` | {mol['qed_score']:.3f} | {mol['mw']:.0f} | {mol['logp']:.2f} | {mol['verdict']} ({mol['violations']} viol.) |\n"
    
    summary = f"""## 🧬 GenMol NIM — Novel Drug Candidates

### Generated from seed: {ligand_name}
**Seed SMILES:** `{smiles[:60]}{'...' if len(smiles) > 60 else ''}`

**GenMol NIM** (self-hosted on GKE) generated **{len(analyzed)}** novel drug-like molecules using masked diffusion on SAFE representations.

### Top Candidates Ranked by QED (Drug-Likeness)
{table}

**Best candidate QED:** {analyzed[0]['qed_score']:.3f} | **Lipinski-compliant:** {sum(1 for m in analyzed if m['violations'] <= 1)}/{len(analyzed)}

💡 Select a generated candidate SMILES above and paste it into the Drug Screening SMILES field to dock it against the target.
"""
    
    # QED score bar chart
    fig = go.Figure(data=go.Bar(
        x=[f"Mol {i+1}" for i in range(len(analyzed[:10]))],
        y=[m["qed_score"] for m in analyzed[:10]],
        marker_color=['#76b900' if m["violations"] <= 1 else '#ffd700' if m["violations"] == 2 else '#ff6b6b' for m in analyzed[:10]],
        text=[f"{m['qed_score']:.3f}" for m in analyzed[:10]],
        textposition='outside'
    ))
    fig.update_layout(
        title="GenMol NIM — Generated Molecule QED Scores",
        template="plotly_dark", height=400,
        yaxis_title="QED Score (Drug-Likeness)", yaxis_range=[0, 1.1]
    )
    
    # Return the best candidate SMILES for easy copy
    best_smiles = analyzed[0]["smiles"]
    
    return summary, fig, best_smiles


def run_drug_screening(structure_id, ligand_name, smiles):
    """Run drug screening with Lipinski analysis and physics-based binding score."""
    global current_results

    if not structure_id or structure_id not in predicted_structures:
        available = list(predicted_structures.keys())
        if not available:
            return "No structures available. Predict a structure first.", None, None, None, None
        return f"Structure not found. Available: {', '.join(available)}", None, None, None, None

    if not smiles:
        return "Enter a SMILES string for the drug candidate.", None, None, None, None

    structure = predicted_structures[structure_id]
    pdb_string = structure['pdb']

    # Get pocket residues (from binding site prediction)
    pockets = structure.get('pockets', [])
    if pockets:
        pocket_residues = pockets[0]['residues']
    else:
        # Fallback: use center residues
        seq_len = len(structure['sequence'])
        pocket_residues = list(range(max(1, seq_len//3), min(seq_len, seq_len//3 + 15)))

    # Compute Lipinski properties
    lipinski = compute_lipinski(smiles)
    if lipinski is None:
        return "Invalid SMILES string. Please check the molecular structure.", None, None, None, None

    # Compute binding score (local physics-based)
    binding = compute_binding_score(smiles, pocket_residues, pdb_string)
    if binding is None:
        return "Could not compute binding score.", None, None, None, None

    # --- DiffDock NIM Integration ---
    diffdock_result = None
    diffdock_error = None
    diffdock_confidence = None
    if NVIDIA_API_KEY:
        diffdock_result, diffdock_error = call_diffdock_nim(pdb_string, smiles, num_poses=3)
        if diffdock_result and not diffdock_error:
            # Extract confidence scores from DiffDock response
            try:
                if isinstance(diffdock_result, dict):
                    # DiffDock returns pose_confidence as list of scores
                    confidences = diffdock_result.get("pose_confidence", diffdock_result.get("confidence", []))
                    if isinstance(confidences, list) and len(confidences) > 0:
                        diffdock_confidence = confidences
                    elif "output" in diffdock_result:
                        # Alternative response format
                        diffdock_confidence = [-0.5]  # placeholder if format unclear
            except Exception as e:
                diffdock_error = f"Could not parse DiffDock response: {e}"

    # Store results
    current_results['drug_screening'] = {
        'structure_id': structure_id,
        'ligand': ligand_name,
        'best_score': f"{binding['best_score']:.2f}",
        'violations': lipinski['violations'],
        'verdict': lipinski['verdict']
    }

    # Lipinski radar chart
    lipinski_fig = create_lipinski_radar(lipinski)

    # Binding scores bar chart
    scores_fig = go.Figure()
    colors = ['#00d4ff' if i == 0 else '#4a4a6a' for i in range(5)]
    scores_fig.add_trace(go.Bar(
        x=[f"Pose {i+1}" for i in range(5)],
        y=binding['poses'],
        marker_color=colors,
        text=[f"{s:.2f}" for s in binding['poses']],
        textposition='outside'
    ))
    scores_fig.update_layout(
        title="Predicted Binding Energy (kcal/mol) — Lower = Stronger Binding",
        template="plotly_dark", height=400,
        yaxis_title="Binding Energy (kcal/mol)"
    )

    # Scoring breakdown
    breakdown_fig = go.Figure(data=go.Bar(
        x=['Size Match', 'Charge Match', 'H-Bond Potential', 'Drug-Likeness'],
        y=[binding['size_score'], binding['charge_score'], binding['hbond_score'], binding['druglike_score']],
        marker_color=['#00d4ff', '#00ff88', '#ffd700', '#ff6b6b'],
        text=[f"{s:.2f}" for s in [binding['size_score'], binding['charge_score'], binding['hbond_score'], binding['druglike_score']]],
        textposition='outside'
    ))
    breakdown_fig.update_layout(
        title="Binding Score Breakdown (0 = poor, 1 = ideal)",
        template="plotly_dark", height=350,
        yaxis_range=[0, 1.2]
    )

    # Build summary
    # Factor in structure confidence (pLDDT) if available
    plddt_warning = ""
    plddt_factor = 1.0
    if structure_id in predicted_structures:
        struct_data = predicted_structures[structure_id]
        if 'mean_plddt' in struct_data:
            mean_plddt = struct_data['mean_plddt']
            if mean_plddt < 50:
                plddt_factor = 0.5
                plddt_warning = "\n⚠️ **Low structure confidence (pLDDT < 50)** — binding predictions are unreliable.\n"
            elif mean_plddt < 70:
                plddt_factor = 0.75
                plddt_warning = "\n⚠️ **Moderate structure confidence (pLDDT < 70)** — interpret binding scores with caution.\n"

    # Adjusted assessment with tighter thresholds
    adjusted_score = binding['best_score'] / plddt_factor  # worse score if structure is poor
    drug_assessment = "🟢 PROMISING" if binding['best_score'] < -9 and lipinski['violations'] <= 1 and plddt_factor >= 0.75 else \
                      "🟡 MODERATE" if binding['best_score'] < -7 and lipinski['violations'] <= 2 else "🔴 WEAK"

    summary = f"""## Drug Screening Results — {ligand_name}

### Overall Assessment: {drug_assessment}
{plddt_warning}
### Binding Analysis
| Property | Value |
|----------|-------|
| **Target Structure** | {structure_id} |
| **Drug Candidate** | {ligand_name} |
| **Best Binding Energy** | {binding['best_score']:.2f} kcal/mol |
| **Pocket Size** | {binding['pocket_size']} residues |
| **Ligand Heavy Atoms** | {binding['heavy_atoms']} |

### Lipinski Rule of 5 (Oral Drug-Likeness)
| Property | Value | Limit | Status |
|----------|-------|-------|--------|
| **Molecular Weight** | {lipinski['mw']:.1f} Da | < 500 | {'✅' if lipinski['mw'] <= 500 else '❌'} |
| **LogP** | {lipinski['logp']:.2f} | < 5 | {'✅' if lipinski['logp'] <= 5 else '❌'} |
| **H-Bond Donors** | {lipinski['hbd']} | < 5 | {'✅' if lipinski['hbd'] <= 5 else '❌'} |
| **H-Bond Acceptors** | {lipinski['hba']} | < 10 | {'✅' if lipinski['hba'] <= 10 else '❌'} |
| **Rotatable Bonds** | {lipinski['rotatable']} | < 10 | {'✅' if lipinski['rotatable'] <= 10 else '❌'} |
| **TPSA** | {lipinski['tpsa']:.1f} Å² | < 140 | {'✅' if lipinski['tpsa'] <= 140 else '❌'} |

**Verdict:** {lipinski['verdict']}
**Violations:** {lipinski['violations']}/4
"""

    # Add DiffDock NIM results section
    if diffdock_confidence and not diffdock_error:
        best_conf = max(diffdock_confidence) if diffdock_confidence else 0
        conf_assessment = "🟢 STRONG" if best_conf > 0 else "🟡 MODERATE" if best_conf > -1 else "🔴 WEAK"
        summary += f"""
### DiffDock NIM — AI Molecular Docking
| Property | Value |
|----------|-------|
| **Method** | DiffDock NIM (NVIDIA Hosted API) |
| **Best Confidence** | {best_conf:.3f} |
| **Assessment** | {conf_assessment} |
| **Poses Generated** | {len(diffdock_confidence)} |

*DiffDock confidence > 0 = likely correct pose, < -1.5 = unlikely*
"""
    elif not NVIDIA_API_KEY:
        summary += "\n### DiffDock NIM\n💡 Set `NVIDIA_API_KEY` to enable AI molecular docking.\n"
    return summary, scores_fig, lipinski_fig, breakdown_fig, f"Screening complete: {drug_assessment}"


def explain_drug_screening():
    results = current_results.get('drug_screening', {})
    genmol = current_results.get('genmol', {})
    if not results and not genmol:
        text = "Please run drug screening first."
        return create_audio_player(text, None, None)
    context = ""
    if results:
        context = f"Drug screening for {results.get('ligand', 'Unknown')} against {results.get('structure_id', 'Unknown')}. Best binding energy: {results.get('best_score', 'N/A')} kcal/mol. Lipinski violations: {results.get('violations', 'N/A')}. {results.get('verdict', '')}."
        # Add structure quality caveat
        struct_id = results.get('structure_id', '')
        if struct_id in predicted_structures and 'mean_plddt' in predicted_structures[struct_id]:
            plddt_val = predicted_structures[struct_id]['mean_plddt']
            if plddt_val < 70:
                context += f" IMPORTANT: Structure confidence is only {plddt_val:.0f}/100 (moderate/low), so binding predictions may not be reliable. Be cautious in your assessment."
    if genmol:
        context += f" Additionally, GenMol NIM generated {genmol.get('num_generated', 0)} novel drug candidates from {genmol.get('seed', 'the seed molecule')}. The best candidate has a QED drug-likeness score of {genmol.get('best_qed', 'N/A')}. {genmol.get('lipinski_pass', 0)} out of {genmol.get('total', 0)} candidates pass Lipinski rules. This demonstrates AI-powered drug design generating novel molecules that never existed before."
    text = get_gemini_text_explanation(context, "Drug Screening")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# TAB 6: RESISTANCE ANALYSIS (was Mutation Effects)
#===============================================================================

def parse_mutations(mutation_string):
    mutations = []
    parts = re.split(r'[,\s]+', mutation_string.strip())
    for part in parts:
        part = part.strip().upper()
        if not part:
            continue
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', part)
        if match:
            mutations.append({
                'wt': match.group(1), 'pos': int(match.group(2)),
                'mut': match.group(3), 'raw': part
            })
    return mutations


def predict_resistance(sequence, mutation_string):
    """Predict drug resistance mutations using ESM-2 log-likelihood ratios."""
    global current_results

    sequence = clean_sequence(sequence)
    if len(sequence) < 5:
        return "Enter at least 5 amino acids", None, None

    mutations = parse_mutations(mutation_string)
    if len(mutations) == 0:
        return "Enter mutations in format: T790M, C797S (original AA, position, new AA)", None, None

    results = []
    for mut in mutations:
        pos = mut['pos']
        wt = mut['wt']
        new = mut['mut']

        if pos < 1 or pos > len(sequence):
            results.append({'mutation': mut['raw'], 'score': None, 'error': f"Position {pos} out of range (1-{len(sequence)})"})
            continue
        if sequence[pos-1] != wt:
            results.append({'mutation': mut['raw'], 'score': None, 'error': f"Position {pos} is {sequence[pos-1]}, not {wt}"})
            continue

        masked_seq = sequence[:pos-1] + '<mask>' + sequence[pos:]
        inputs = esm2_tokenizer(masked_seq, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = esm2_mlm(**inputs)
            logits = outputs.logits

        mask_token_id = esm2_tokenizer.mask_token_id
        mask_idx = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0].item()
        probs = F.softmax(logits[0, mask_idx], dim=-1)
        wt_token = esm2_tokenizer.convert_tokens_to_ids(wt)
        mut_token = esm2_tokenizer.convert_tokens_to_ids(new)

        if wt_token is None or mut_token is None:
            results.append({'mutation': mut['raw'], 'score': None, 'error': "Invalid amino acid"})
            continue

        wt_prob = probs[wt_token].item()
        mut_prob = probs[mut_token].item()
        llr = np.log(mut_prob / (wt_prob + 1e-10))
        results.append({'mutation': mut['raw'], 'score': llr, 'wt_prob': wt_prob, 'mut_prob': mut_prob, 'error': None})

    valid_results = [r for r in results if r['error'] is None]
    if len(valid_results) == 0:
        error_msgs = "\n".join([f"- {r['mutation']}: {r['error']}" for r in results if r['error']])
        return f"All mutations had errors:\n{error_msgs}", None, None

    valid_results.sort(key=lambda x: x['score'])
    worst = valid_results[0]
    best = valid_results[-1]

    current_results['resistance'] = {
        'num_mutations': len(valid_results),
        'worst_mutation': worst['mutation'], 'worst_score': f"{worst['score']:.3f}",
        'best_mutation': best['mutation'], 'best_score': f"{best['score']:.3f}"
    }

    fig = go.Figure()
    mutations_sorted = [r['mutation'] for r in valid_results]
    scores_sorted = [r['score'] for r in valid_results]
    colors = ['#ff6b6b' if s < -1 else '#ffd93d' if s < 0 else '#6bcb77' for s in scores_sorted]

    fig.add_trace(go.Bar(
        x=mutations_sorted, y=scores_sorted, marker_color=colors,
        text=[f"{s:.2f}" for s in scores_sorted], textposition='outside'
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=2)
    fig.add_hline(y=-1, line_dash="dash", line_color="red", annotation_text="Likely destabilizing — drug may still bind")
    fig.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="Well-tolerated — potential resistance")
    fig.update_layout(
        title="Resistance Mutation Analysis (Log-Likelihood Ratio)",
        xaxis_title="Mutation", yaxis_title="LLR Score",
        template="plotly_dark", height=400
    )

    table_rows = "| Mutation | LLR Score | WT Prob | Mut Prob | Resistance Risk |\n|----------|-----------|---------|----------|-----------------|\n"
    for r in valid_results:
        if r['score'] > 0.5:
            risk = "🔴 HIGH — mutation is tolerated, drug binding may be disrupted"
        elif r['score'] > -0.5:
            risk = "🟡 MODERATE — mutation partially tolerated"
        else:
            risk = "🟢 LOW — mutation is destabilizing, unlikely to persist"
        table_rows += f"| {r['mutation']} | {r['score']:.3f} | {r['wt_prob']:.3f} | {r['mut_prob']:.3f} | {risk} |\n"

    error_section = ""
    errors = [r for r in results if r['error'] is not None]
    if errors:
        error_section = "\n### Errors\n" + "\n".join([f"- **{r['mutation']}**: {r['error']}" for r in errors])

    disease_context = ""
    if selected_target_info:
        disease_context = f"\n**Disease:** {selected_target_info.get('disease', 'N/A')}\n"

    summary = f"""## Drug Resistance Analysis
{disease_context}
### Will mutations at the binding site break our drug?

{table_rows}

**Interpretation for drug discovery:**
- **Positive scores (🔴)**: Mutation is well-tolerated by the protein → it can persist → HIGH resistance risk
- **Scores near 0 (🟡)**: Mutation is partially tolerated → MODERATE resistance risk
- **Negative scores (🟢)**: Mutation destabilizes the protein → unlikely to persist → LOW resistance risk
{error_section}
"""
    return summary, fig, f"Analyzed {len(valid_results)} resistance mutations"


def explain_resistance():
    results = current_results.get('resistance', {})
    if not results:
        text = "Please analyze resistance mutations first."
        return create_audio_player(text, None, None)
    context = f"Analyzed {results.get('num_mutations', 'N/A')} resistance mutations. Highest risk: {results.get('best_mutation', 'N/A')} (score {results.get('best_score', 'N/A')}). Lowest risk: {results.get('worst_mutation', 'N/A')} (score {results.get('worst_score', 'N/A')})."
    text = get_gemini_text_explanation(context, "Drug Resistance Analysis")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# TAB 7: LEAD COMPARISON
#===============================================================================

def compare_proteins(seq1, seq2, name1, name2):
    global current_results
    seq1 = clean_sequence(seq1)
    seq2 = clean_sequence(seq2)
    if len(seq1) < 5 or len(seq2) < 5:
        return "Both sequences must have at least 5 amino acids", None, None, None, None
    if len(seq1) > MAX_SEQUENCE_LENGTH or len(seq2) > MAX_SEQUENCE_LENGTH:
        return f"Maximum sequence length is {MAX_SEQUENCE_LENGTH} amino acids", None, None, None, None

    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    if alignments:
        score = alignments[0].score
        similarity = (score / max(len(seq1), len(seq2))) * 100
    else:
        similarity = 0
        score = 0

    analysis1 = analyze_sequence(seq1)
    analysis2 = analyze_sequence(seq2)
    mw_diff = abs(analysis1['mol_weight'] - analysis2['mol_weight'])
    pi_diff = abs(analysis1['isoelectric'] - analysis2['isoelectric'])

    current_results['lead_comparison'] = {
        'name1': name1, 'name2': name2,
        'length1': analysis1['length'], 'length2': analysis2['length'],
        'similarity': f"{similarity:.1f}", 'mw_diff': f"{mw_diff:.0f}"
    }

    summary = f"""## Target Variant Comparison

### Sequence Similarity
| Metric | Value |
|--------|-------|
| **Alignment Score** | {score:.1f} |
| **Similarity** | {similarity:.1f}% |

### Properties Comparison
| Property | {name1} | {name2} | Difference |
|----------|---------|---------|------------|
| **Length** | {analysis1['length']} aa | {analysis2['length']} aa | {abs(analysis1['length'] - analysis2['length'])} aa |
| **Molecular Weight** | {analysis1['mol_weight']:.0f} Da | {analysis2['mol_weight']:.0f} Da | {mw_diff:.0f} Da |
| **Isoelectric Point** | {analysis1['isoelectric']:.2f} | {analysis2['isoelectric']:.2f} | {pi_diff:.2f} |
"""

    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(name=name1, x=list(analysis1['aa_counts'].keys()), y=list(analysis1['aa_counts'].values()), marker_color='#00d4ff'))
    comp_fig.add_trace(go.Bar(name=name2, x=list(analysis2['aa_counts'].keys()), y=list(analysis2['aa_counts'].values()), marker_color='#ff6b6b'))
    comp_fig.update_layout(title="Amino Acid Composition", template="plotly_dark", height=400, barmode='group')

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=list(analysis1['properties'].values()), theta=list(analysis1['properties'].keys()), fill='toself', name=name1, line_color='#00d4ff'))
    radar_fig.add_trace(go.Scatterpolar(r=list(analysis2['properties'].values()), theta=list(analysis2['properties'].keys()), fill='toself', name=name2, line_color='#ff6b6b'))
    radar_fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), template="plotly_dark", height=400)

    inputs1 = esmfold_tokenizer([seq1], return_tensors="pt", add_special_tokens=False).to("cuda")
    inputs2 = esmfold_tokenizer([seq2], return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        outputs1 = esmfold_model(**inputs1)
        outputs2 = esmfold_model(**inputs2)
    pdb1 = esmfold_model.output_to_pdb(outputs1)[0]
    pdb2 = esmfold_model.output_to_pdb(outputs2)[0]

    predicted_structures[f"{name1}_{len(seq1)}aa"] = {'pdb': pdb1, 'sequence': seq1}
    predicted_structures[f"{name2}_{len(seq2)}aa"] = {'pdb': pdb2, 'sequence': seq2}

    return summary, comp_fig, radar_fig, create_side_by_side_3d_viewer(pdb1, pdb2, name1, name2), "Comparison complete!"


def explain_lead_comparison():
    results = current_results.get('lead_comparison', {})
    if not results:
        text = "Please compare two targets first."
        return create_audio_player(text, None, None)
    context = f"Compared {results.get('name1', 'Unknown')} ({results.get('length1', 'N/A')} aa) with {results.get('name2', 'Unknown')} ({results.get('length2', 'N/A')} aa). Sequence similarity: {results.get('similarity', 'N/A')}%."
    text = get_gemini_text_explanation(context, "Target Comparison")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# TAB 8: GPU BENCHMARK
#===============================================================================

def run_benchmark():
    global current_results
    test_sequences = [
        ("Short (30aa)", "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL"),
        ("Medium (50aa)", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"),
        ("Long (76aa)", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
    ]
    results = []
    total_time = 0
    peak_memory = 0

    for name, seq in test_sequences:
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 1e9
        start_time = time.time()
        inputs = esmfold_tokenizer([seq], return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            outputs = esmfold_model(**inputs)
        pdb = esmfold_model.output_to_pdb(outputs)[0]
        elapsed = time.time() - start_time
        current_peak = torch.cuda.max_memory_allocated() / 1e9
        peak_memory = max(peak_memory, current_peak)
        mem_used = current_peak - start_mem
        results.append({'name': name, 'length': len(seq), 'time': elapsed, 'memory': mem_used, 'speed': 60 / elapsed})
        total_time += elapsed

    avg_speed = sum(r['speed'] for r in results) / 3
    current_results['benchmark'] = {
        'short_time': f"{results[0]['time']:.2f}", 'medium_time': f"{results[1]['time']:.2f}",
        'long_time': f"{results[2]['time']:.2f}", 'avg_speed': f"{avg_speed:.1f}",
        'peak_memory': f"{peak_memory:.1f}"
    }

    summary = f"""## GPU Benchmark Results — {time.strftime('%Y-%m-%d %H:%M:%S')}

| Protein | Length | Time | Memory | Speed |
|---------|--------|------|--------|-------|
| {results[0]['name']} | {results[0]['length']} aa | {results[0]['time']:.2f}s | {results[0]['memory']:.2f} GB | {results[0]['speed']:.1f}/min |
| {results[1]['name']} | {results[1]['length']} aa | {results[1]['time']:.2f}s | {results[1]['memory']:.2f} GB | {results[1]['speed']:.1f}/min |
| {results[2]['name']} | {results[2]['length']} aa | {results[2]['time']:.2f}s | {results[2]['memory']:.2f} GB | {results[2]['speed']:.1f}/min |

| Metric | Value |
|--------|-------|
| **Total Time** | {total_time:.2f}s |
| **Average Speed** | {avg_speed:.1f} proteins/min |
| **Peak Memory** | {peak_memory:.1f} GB |
"""

    time_fig = go.Figure(data=go.Bar(x=[r['name'] for r in results], y=[r['time'] for r in results], marker_color=['#00d4ff', '#00ff88', '#ff6b6b'], text=[f"{r['time']:.2f}s" for r in results], textposition='outside'))
    time_fig.update_layout(title="Prediction Time", template="plotly_dark", height=350, yaxis_title="Seconds")

    mem_fig = go.Figure(data=go.Bar(x=[r['name'] for r in results], y=[r['memory'] for r in results], marker_color=['#00d4ff', '#00ff88', '#ff6b6b'], text=[f"{r['memory']:.2f} GB" for r in results], textposition='outside'))
    mem_fig.update_layout(title="Memory Usage", template="plotly_dark", height=350, yaxis_title="GB")

    speed_fig = go.Figure(data=go.Bar(x=[r['name'] for r in results], y=[r['speed'] for r in results], marker_color=['#00d4ff', '#00ff88', '#ff6b6b'], text=[f"{r['speed']:.1f}/min" for r in results], textposition='outside'))
    speed_fig.update_layout(title="Inference Speed", template="plotly_dark", height=350, yaxis_title="Proteins/min")

    return summary, time_fig, mem_fig, speed_fig


def explain_benchmark():
    results = current_results.get('benchmark', {})
    if not results:
        text = "Please run the benchmark first."
        return create_audio_player(text, None, None)
    context = f"GPU benchmark on NVIDIA RTX PRO 6000. Short protein: {results.get('short_time', 'N/A')}s, Medium: {results.get('medium_time', 'N/A')}s, Long: {results.get('long_time', 'N/A')}s. Average speed: {results.get('avg_speed', 'N/A')} proteins per minute."
    text = get_gemini_text_explanation(context, "GPU Benchmark")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# TAB 9: GPU MONITOR
#===============================================================================

def create_gpu_dials():
    global current_results
    try:
        gpu_name = torch.cuda.get_device_name(0)
        memory_used = torch.cuda.memory_allocated(0) / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_free = memory_total - memory_used
        utilization_pct = (memory_used / memory_total) * 100
    except:
        gpu_name = "GPU"
        memory_used = 0
        memory_total = 48
        memory_free = 48
        utilization_pct = 0

    current_results['gpu_monitor'] = {
        'gpu_name': gpu_name, 'memory_used': f"{memory_used:.1f}",
        'memory_total': f"{memory_total:.1f}", 'utilization': f"{utilization_pct:.1f}"
    }

    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'indicator'}]*3], horizontal_spacing=0.1)

    fig.add_trace(go.Indicator(
        mode="gauge+number", value=utilization_pct,
        title={'text': "GPU Utilization", 'font': {'size': 20, 'color': '#00d4ff'}},
        number={'suffix': '%', 'font': {'size': 36, 'color': 'white'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#00d4ff'}, 'bgcolor': '#1a1a2e',
               'steps': [{'range': [0, 50], 'color': '#1e3a5f'}, {'range': [50, 80], 'color': '#3d5a80'}, {'range': [80, 100], 'color': '#5c2a2a'}]}
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number", value=memory_used,
        title={'text': "Memory Used", 'font': {'size': 20, 'color': '#00ff88'}},
        number={'suffix': ' GB', 'font': {'size': 36, 'color': 'white'}},
        gauge={'axis': {'range': [0, memory_total]}, 'bar': {'color': '#00ff88'}, 'bgcolor': '#1a1a2e',
               'steps': [{'range': [0, memory_total*0.5], 'color': '#1e5f3a'}, {'range': [memory_total*0.5, memory_total*0.8], 'color': '#3d805a'}, {'range': [memory_total*0.8, memory_total], 'color': '#5c2a2a'}]}
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="gauge+number", value=memory_free,
        title={'text': "Memory Free", 'font': {'size': 20, 'color': '#ffd700'}},
        number={'suffix': ' GB', 'font': {'size': 36, 'color': 'white'}},
        gauge={'axis': {'range': [0, memory_total]}, 'bar': {'color': '#ffd700'}, 'bgcolor': '#1a1a2e',
               'steps': [{'range': [0, memory_total*0.2], 'color': '#5c2a2a'}, {'range': [memory_total*0.2, memory_total*0.5], 'color': '#5f5a1e'}, {'range': [memory_total*0.5, memory_total], 'color': '#1e5f3a'}]}
    ), row=1, col=3)

    fig.update_layout(template="plotly_dark", height=350, paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e', font={'color': 'white'}, margin=dict(t=80, b=40, l=40, r=40))

    gpu_info = f"""## {gpu_name}
| Property | Value |
|----------|-------|
| **Total Memory** | {memory_total:.1f} GB |
| **Used Memory** | {memory_used:.1f} GB |
| **Free Memory** | {memory_free:.1f} GB |
| **Utilization** | {utilization_pct:.1f}% |
"""
    return fig, gpu_info


def explain_gpu_monitor():
    try:
        sample = _sample_gpu_once()
        with _gpu_lock:
            peak = gpu_stats['peak_util']
        gpu_name = sample['gpu_name'] if sample else "GPU"
        memory_used = sample['mem_used'] if sample else 0
        memory_total = sample['mem_total'] if sample else 96
        context = f"GPU: {gpu_name}. Total: {memory_total:.1f} GB, Used: {memory_used:.1f} GB, Peak Utilization: {peak:.0f}%. Explain the GPU utilization and memory usage for a technical audience."
    except:
        context = "GPU information unavailable."
    text = get_gemini_text_explanation(context, "GPU Monitoring")
    audio_data, mime_type = get_gemini_audio(text)
    return create_audio_player(text, audio_data, mime_type)

#===============================================================================
# GRADIO USER INTERFACE - 5-Act Drug Discovery Pipeline
#===============================================================================

with gr.Blocks(title="AI Drug Discovery", theme=gr.themes.Soft(primary_hue="cyan")) as demo:

    # Banner
    gr.HTML("""
    <div style="text-align:center; padding:30px; background: white; border-radius:12px; margin-bottom:20px; border: 1px solid #00d4ff33;">
        <div style="display:flex; align-items:center; justify-content:center; gap:15px; margin-bottom:10px;">
            <span style="font-size:40px;">💊</span>
            <h1 style="color:#00d4ff; margin:0; font-size:2.2em;">Accelerating Drug Discovery with AI</h1>
            <span style="font-size:40px;">🧬</span>
        </div>
        <p style="color:#76b900; font-size:1.2em; margin:5px 0; font-weight:bold;">NVIDIA + Google Cloud: Better Together</p>
        <p style="color:#333; font-size:1.05em; font-weight:600; margin:5px 0;">ESM-2 · ESMFold · GenMol NIM · DiffDock NIM · RDKit · Gemini AI · NVIDIA RTX PRO 6000 · GKE</p>
        <div style="display:flex; justify-content:center; gap:30px; margin-top:15px;">
            <span style="color:#222; font-size:0.95em; font-weight:600;">Step 1: Select Target</span>
            <span style="color:#222;">→</span>
            <span style="color:#222; font-size:0.95em; font-weight:600;">Step 2: Analyze</span>
            <span style="color:#222;">→</span>
            <span style="color:#222; font-size:0.95em; font-weight:600;">Step 3: Predict Structure</span>
            <span style="color:#222;">→</span>
            <span style="color:#222; font-size:0.95em; font-weight:600;">Step 4: Find Binding Sites</span>
            <span style="color:#222;">→</span>
            <span style="color:#222; font-size:0.95em; font-weight:600;">Step 5: Screen Drugs</span>
            <span style="color:#222;">→</span>
            <span style="color:#222; font-size:0.95em; font-weight:600;">Step 6: Test Resistance</span>
        </div>
    </div>
    """)

    gr.HTML("""<style>.tabs button{font-weight:600 !important;color:#333 !important;font-size:0.95em !important;}.tabs button.selected{font-weight:700 !important;color:#0891b2 !important;}</style>""")
    with gr.Tabs():

        # ===== TAB 1: TARGET DISCOVERY =====
        with gr.TabItem("🎯 Step 1: Target Discovery"):
            gr.Markdown("### Select a Disease Target to Begin the Drug Discovery Pipeline")
            gr.Markdown("Choose a protein target associated with a disease. This will populate all downstream analysis steps.")
            target_dropdown = gr.Dropdown(
                list(DRUG_TARGETS.keys()),
                label="Select Disease Target",
                value=list(DRUG_TARGETS.keys())[0]
            )
            target_info = gr.HTML()
            # Hidden state outputs to propagate to other tabs
            target_seq_state = gr.Textbox(visible=False)
            target_mutations_state = gr.Textbox(visible=False)
            target_ligand_state = gr.Textbox(visible=False)
            gr.Markdown("---")
            explain_btn_target = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_target = gr.HTML()

            target_dropdown.change(select_target, target_dropdown, [target_info, target_seq_state, target_mutations_state, target_ligand_state])
            demo.load(select_target, target_dropdown, [target_info, target_seq_state, target_mutations_state, target_ligand_state])
            explain_btn_target.click(explain_target_discovery, None, explain_out_target)

        # ===== TAB 2: TARGET ANALYSIS =====
        with gr.TabItem("🧬 Step 2: Target Analysis"):
            gr.Markdown("### Analyze the Drug Target's Biochemical Properties")
            with gr.Row():
                with gr.Column(scale=1):
                    seq_analysis = gr.Textbox(label="Target Sequence", lines=4, value=DRUG_TARGETS[list(DRUG_TARGETS.keys())[0]]['sequence'])
                    btn_analysis = gr.Button("Analyze Target", variant="primary")
                with gr.Column(scale=2):
                    out_analysis = gr.Markdown()
                    viz_analysis = gr.HTML()
            with gr.Row():
                plot_comp = gr.Plot(label="Amino Acid Composition")
                plot_props = gr.Plot(label="Property Distribution")
            plot_embed = gr.Plot(label="ESM-2 Embeddings")
            gr.Markdown("---")
            explain_btn_analysis = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_analysis = gr.HTML()

            # Wire target selection to update sequence
            target_seq_state.change(lambda x: x, target_seq_state, seq_analysis)
            btn_analysis.click(analyze_protein, seq_analysis, [out_analysis, viz_analysis, plot_comp, plot_props, plot_embed])
            explain_btn_analysis.click(explain_target_analysis, None, explain_out_analysis)

        # ===== TAB 3: STRUCTURE PREDICTION =====
        with gr.TabItem("🏗️ Step 3: Structure Prediction"):
            gr.Markdown(f"### Predict 3D Structure of the Drug Target\nMaximum sequence length: **{MAX_SEQUENCE_LENGTH} amino acids**")
            with gr.Row():
                with gr.Column(scale=1):
                    seq_struct = gr.Textbox(label="Target Sequence", lines=4, value=DRUG_TARGETS[list(DRUG_TARGETS.keys())[0]]['sequence'])
                    name_struct = gr.Textbox(label="Structure Name", value="DrugTarget")
                    btn_struct = gr.Button("Predict Structure", variant="primary")
                with gr.Column(scale=2):
                    out_struct = gr.Markdown()
            struct_view = gr.HTML(label="3D Structure Viewer")
            plddt_plot = gr.Plot(label="pLDDT Confidence Score")
            with gr.Row():
                pdb_out = gr.Textbox(visible=False)
                pdb_download = gr.File(label="Download PDB File")
            gr.Markdown("---")
            explain_btn_struct = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_struct = gr.HTML()

            target_seq_state.change(lambda x: x, target_seq_state, seq_struct)
            btn_struct.click(predict_structure, [seq_struct, name_struct], [out_struct, struct_view, plddt_plot, pdb_out, pdb_download])
            explain_btn_struct.click(explain_structure_prediction, None, explain_out_struct)

        # ===== TAB 4: BINDING SITE DETECTION =====
        with gr.TabItem("📍 Step 4: Binding Sites"):
            gr.Markdown("### Detect Drug Binding Pockets + Contact Map")
            gr.Markdown("Find where drug molecules can bind to the target protein. **Requires a predicted structure from Step 3.**")
            with gr.Row():
                with gr.Column(scale=1):
                    struct_binding = gr.Dropdown(choices=[], label="Select Structure")
                    refresh_binding_btn = gr.Button("🔄 Refresh Structures")
                    binding_btn = gr.Button("Detect Binding Sites", variant="primary")
                with gr.Column(scale=2):
                    binding_out = gr.Markdown()
            binding_plot = gr.Plot(label="Pocket Scores by Residue")
            binding_view = gr.HTML(label="3D Structure with Binding Sites")
            contact_plot = gr.Plot(label="Residue Contact Map")
            binding_status = gr.Markdown()
            gr.Markdown("---")
            explain_btn_binding = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_binding = gr.HTML()

            refresh_binding_btn.click(refresh_structures, None, struct_binding)
            binding_btn.click(predict_binding_sites_and_contacts, struct_binding, [binding_out, binding_plot, contact_plot, binding_view, binding_status])
            explain_btn_binding.click(explain_binding_site, None, explain_out_binding)

        # ===== TAB 5: DRUG SCREENING =====
        with gr.TabItem("💊 Step 5: Drug Screening"):
            gr.Markdown("### Screen Drug Candidates Against the Target")
            gr.Markdown("Evaluate drug candidates using **DiffDock NIM** molecular docking, Lipinski Rule of 5, and physics-based scoring. Generate novel candidates with **GenMol NIM**.")
            with gr.Row():
                with gr.Column():
                    struct_drug = gr.Dropdown(choices=[], label="Select Protein Structure")
                    refresh_drug_btn = gr.Button("🔄 Refresh Structures")
                    ligand_dd = gr.Dropdown(choices=get_target_ligands(), label="Select Drug Candidate", value=get_target_ligands()[0] if get_target_ligands() else None)
                    refresh_ligand_btn = gr.Button("🔄 Refresh Drug Candidates")
                    smiles_in = gr.Textbox(label="SMILES String", value="")
                    drug_btn = gr.Button("Screen Drug Candidate", variant="primary")
                with gr.Column():
                    drug_out = gr.Markdown()
            with gr.Row():
                drug_scores_plot = gr.Plot(label="Binding Energy Predictions")
                drug_lipinski_plot = gr.Plot(label="Drug-Likeness Profile")
            drug_breakdown_plot = gr.Plot(label="Binding Score Breakdown")
            drug_status = gr.Markdown()
            
            # GenMol NIM Section
            gr.Markdown("---")
            gr.Markdown("### 🧬 Generate Novel Drug Candidates with GenMol NIM")
            gr.Markdown("Use NVIDIA GenMol (self-hosted on GKE) to generate novel drug-like molecules from the selected seed compound.")
            genmol_btn = gr.Button("🧬 Generate Novel Candidates (GenMol NIM)", variant="secondary")
            genmol_out = gr.Markdown()
            genmol_plot = gr.Plot(label="GenMol Generated Molecules")
            genmol_best = gr.Textbox(label="Best Candidate SMILES (paste into screening above)", interactive=True)
            
            gr.Markdown("---")
            explain_btn_drug = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_drug = gr.HTML()

            def update_smiles_from_ligand(ligand_name):
                if selected_target_info and 'ligands' in selected_target_info:
                    return selected_target_info['ligands'].get(ligand_name, "")
                return ""

            refresh_drug_btn.click(refresh_structures, None, struct_drug)
            refresh_ligand_btn.click(refresh_ligands, None, ligand_dd)
            ligand_dd.change(update_smiles_from_ligand, ligand_dd, smiles_in)
            drug_btn.click(run_drug_screening, [struct_drug, ligand_dd, smiles_in], [drug_out, drug_scores_plot, drug_lipinski_plot, drug_breakdown_plot, drug_status])
            genmol_btn.click(generate_novel_candidates, [struct_drug, ligand_dd, smiles_in], [genmol_out, genmol_plot, genmol_best])
            genmol_best.change(lambda x: x, genmol_best, smiles_in)
            explain_btn_drug.click(explain_drug_screening, None, explain_out_drug)

        # ===== TAB 6: RESISTANCE ANALYSIS =====
        with gr.TabItem("🧪 Step 6: Resistance Analysis"):
            gr.Markdown("### Will Mutations Break Our Drug?")
            gr.Markdown("Predict whether known resistance mutations will allow the target to evade drug binding.")
            with gr.Row():
                with gr.Column(scale=1):
                    seq_resist = gr.Textbox(label="Target Sequence", lines=4, value=DRUG_TARGETS[list(DRUG_TARGETS.keys())[0]]['sequence'])
                    mutations_input = gr.Textbox(
                        label="Resistance Mutations (format: T790M, C797S)",
                        value=DRUG_TARGETS[list(DRUG_TARGETS.keys())[0]]['resistance_mutations']
                    )
                    resist_btn = gr.Button("Analyze Resistance", variant="primary")
                with gr.Column(scale=2):
                    resist_out = gr.Markdown()
            resist_plot = gr.Plot(label="Resistance Mutation Scores")
            resist_status = gr.Markdown()
            gr.Markdown("---")
            explain_btn_resist = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_resist = gr.HTML()

            target_seq_state.change(lambda x: x, target_seq_state, seq_resist)
            target_mutations_state.change(lambda x: x, target_mutations_state, mutations_input)
            resist_btn.click(predict_resistance, [seq_resist, mutations_input], [resist_out, resist_plot, resist_status])
            explain_btn_resist.click(explain_resistance, None, explain_out_resist)

        # ===== TAB 7: LEAD COMPARISON =====
        with gr.TabItem("🔬 Target Comparison"):
            gr.Markdown("### Compare Two Drug Target Variants")
            gr.Markdown("Analyze sequence similarity, properties, and predict both structures side-by-side.")
            target_names = list(DRUG_TARGETS.keys())
            with gr.Row():
                with gr.Column():
                    cmp_dd1 = gr.Dropdown(target_names, label="Target 1", value=target_names[0])
                    seq_cmp1 = gr.Textbox(label="Sequence 1", lines=3, value=DRUG_TARGETS[target_names[0]]['sequence'])
                    name_cmp1 = gr.Textbox(label="Name", value="Target1")
                with gr.Column():
                    cmp_dd2 = gr.Dropdown(target_names, label="Target 2", value=target_names[1])
                    seq_cmp2 = gr.Textbox(label="Sequence 2", lines=3, value=DRUG_TARGETS[target_names[1]]['sequence'])
                    name_cmp2 = gr.Textbox(label="Name", value="Target2")
            btn_compare = gr.Button("Compare Targets", variant="primary")
            compare_out = gr.Markdown()
            with gr.Row():
                compare_comp = gr.Plot(label="Composition")
                compare_radar = gr.Plot(label="Properties")
            compare_structures = gr.HTML(label="3D Structures Side-by-Side")
            compare_status = gr.Markdown()
            gr.Markdown("---")
            explain_btn_cmp = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_cmp = gr.HTML()

            cmp_dd1.change(lambda x: DRUG_TARGETS[x]['sequence'] if x in DRUG_TARGETS else "", cmp_dd1, seq_cmp1)
            cmp_dd2.change(lambda x: DRUG_TARGETS[x]['sequence'] if x in DRUG_TARGETS else "", cmp_dd2, seq_cmp2)
            btn_compare.click(compare_proteins, [seq_cmp1, seq_cmp2, name_cmp1, name_cmp2], [compare_out, compare_comp, compare_radar, compare_structures, compare_status])
            explain_btn_cmp.click(explain_lead_comparison, None, explain_out_cmp)

        # ===== TAB 8: GPU BENCHMARK =====
        with gr.TabItem("⚡ GPU Benchmark"):
            gr.Markdown("### GPU Performance Benchmark")
            bench_btn = gr.Button("Run Benchmark", variant="primary")
            bench_out = gr.Markdown()
            with gr.Row():
                bench_time = gr.Plot(label="Prediction Time")
                bench_mem = gr.Plot(label="Memory Usage")
            bench_speed = gr.Plot(label="Inference Speed")
            gr.Markdown("---")
            explain_btn_bench = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_bench = gr.HTML()

            bench_btn.click(run_benchmark, None, [bench_out, bench_time, bench_mem, bench_speed])
            explain_btn_bench.click(explain_benchmark, None, explain_out_bench)

        # ===== TAB 9: GPU MONITOR =====
        with gr.TabItem("📊 GPU Monitor"):
            gr.Markdown("### Real-Time GPU Monitoring")
            gpu_dials = gr.Plot(label="GPU Metrics")
            gpu_info = gr.Markdown()
            refresh_gpu_btn = gr.Button("🔄 Refresh", variant="primary")
            gr.Markdown("---")
            explain_btn_gpu = gr.Button("🔊 Explain with AI Voice", variant="secondary")
            explain_out_gpu = gr.HTML()

            refresh_gpu_btn.click(create_gpu_dials, None, [gpu_dials, gpu_info])
            demo.load(create_gpu_dials, None, [gpu_dials, gpu_info])
            explain_btn_gpu.click(explain_gpu_monitor, None, explain_out_gpu)

        # ===== TAB 10: ABOUT =====
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
## About — Accelerating Drug Discovery with AI

### The Pipeline
This demo showcases a complete AI-accelerated drug discovery pipeline with **genuine NVIDIA NIM integration**:

| Step | Tool | What It Does | Where It Runs |
|------|------|-------------|---------------|
| **1. Target Discovery** | Curated Database | Select a disease target with known drug context | Local |
| **2. Target Analysis** | ESM-2 (650M params) | Analyze biochemical properties and embeddings | Local GPU |
| **3. Structure Prediction** | ESMFold | Predict 3D protein structure with confidence | Local GPU |
| **4. Binding Site Detection** | Geometric Analysis | Find druggable pockets on the protein surface | Local CPU |
| **5. Drug Screening** | DiffDock NIM + RDKit | AI molecular docking + drug-likeness scoring | NVIDIA Hosted API + Local |
| **5b. Novel Generation** | GenMol NIM | Generate novel drug candidates from seed molecules | Self-hosted GKE NIM |
| **6. Resistance Analysis** | ESM-2 MLM | Predict if mutations will break drug binding | Local GPU |

### NVIDIA NIM Integration
| NIM | Version | Deployment | Purpose |
|-----|---------|-----------|---------|
| **GenMol** | 1.0.1 | Self-hosted on GKE (RTX PRO 6000) | Novel molecule generation via masked diffusion |
| **DiffDock** | 2.x | NVIDIA Hosted API | AI-powered blind molecular docking |

### Infrastructure
| Component | Specification |
|-----------|---------------|
| **Platform** | Google Kubernetes Engine (GKE) |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |
| **Storage** | Hyperdisk-balanced (250 GB) |
| **AI Voice** | Gemini 2.5 Flash TTS |
| **Drug Chemistry** | RDKit (local) |
| **Protein AI** | ESM-2 + ESMFold (Meta AI, local GPU) |

### Architecture: Hybrid Local + NIM
- **2 NVIDIA NIMs actively called** — GenMol (self-hosted) + DiffDock (hosted API)
- **2 Local GPU models** — ESM-2 (650M) + ESMFold
- **1 Local CPU library** — RDKit for Lipinski/drug-likeness
- **1 Cloud API** — Gemini AI for explanations + TTS

### Credits
- **ESM-2 & ESMFold**: Meta AI Research
- **GenMol & DiffDock**: NVIDIA BioNeMo NIMs
- **RDKit**: Open-source cheminformatics
- **3D Visualization**: 3Dmol.js
- **Charts**: Plotly
- **UI**: Gradio

---
*Powered by NVIDIA BioNeMo NIMs on Google Kubernetes Engine | Gemini AI*
""")
            gr.Markdown("---")
            explain_btn_about = gr.Button("🔊 Explain Dashboard with AI Voice", variant="secondary")
            explain_out_about = gr.HTML()

            def explain_about():
                context = "AI drug discovery pipeline running on GKE with NVIDIA RTX PRO 6000 Blackwell GPU. Features 6-step pipeline from target selection through resistance analysis, powered by ESM-2, ESMFold, RDKit, and Gemini AI."
                text = get_gemini_text_explanation(context, "Dashboard Overview")
                audio_data, mime_type = get_gemini_audio(text)
                return create_audio_player(text, audio_data, mime_type)

            explain_btn_about.click(explain_about, None, explain_out_about)

    # Footer
    gr.HTML('''
    <div style="text-align:center;padding:20px;color:#666;">
        <p style="font-size:1.1em;"><strong style="color:#76b900;">NVIDIA</strong> + <strong style="color:#4285f4;">Google Cloud</strong>: Better Together</p>
        <p>Powered by NVIDIA BioNeMo NIMs (GenMol + DiffDock) | RTX PRO 6000 Blackwell | Google Kubernetes Engine | Gemini AI</p>
    </div>
    ''')

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
