"""
Resonate Web Interface - Streamlit app for live music reconstruction

Provides user-friendly interface for:
- Audio file upload and analysis
- Processing configuration
- Before/after comparison with A/B toggle
- Quality metrics visualization
- Restoration settings
- Stem regeneration (Phase 4 - JASCO AI)
- Stem export
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, fields

# Add parent directory to path to import audio_engine
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import tempfile
import os

from audio_engine.pipeline import AudioPipeline, PipelineConfig, PipelineMode
from logging_config import get_logger
logger = get_logger()
from audio_engine.metrics import QualityMetrics, analyze_quality
from audio_engine.profiling.quality_detector import QualityDetector, StemQualityReport, DamageLevel
from audio_engine.generation.stem_regenerator import StemRegenerator, RegenerationPlan, RegenerationRegion, RegenerationSummary
from audio_engine.generation.blender import Blender
from audio_engine.mixing import StemMixer

# Setup logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Resonate - Live Music Reconstruction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FFFFFF;
        margin-top: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1DB95420;
        border: 1px solid #1DB954;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #2D2D2D;
        text-align: center;
    }
    .ab-toggle {
        text-align: center;
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #1DB954;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def plot_waveform(audio: np.ndarray, sample_rate: int, title: str = "Waveform") -> go.Figure:
    """Create an interactive waveform plot using Plotly."""
    # Downsample for visualization if needed
    max_points = 5000
    if len(audio) > max_points:
        step = len(audio) // max_points
        audio_plot = audio[::step]
        time_plot = np.arange(len(audio_plot)) * step / sample_rate
    else:
        audio_plot = audio
        time_plot = np.arange(len(audio)) / sample_rate
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_plot,
        y=audio_plot,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(29, 185, 84, 0.2)',
        line=dict(color='#1DB954', width=1),
        name='Waveform'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
        height=150,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def plot_spectrogram(audio: np.ndarray, sample_rate: int, title: str = "Spectrogram") -> go.Figure:
    """Create a spectrogram visualization using Plotly."""
    import librosa
    import librosa.display
    
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        colorscale='Viridis',
        showscale=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Frequency",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark"
    )
    
    return fig


def prepare_analysis_audio(audio: np.ndarray, sample_rate: int, duration_s: float) -> np.ndarray:
    """Prepare audio for metrics analysis (mono + truncated to reduce compute)."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if duration_s and duration_s > 0:
        max_samples = int(sample_rate * duration_s)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    return audio.astype(np.float32)


def get_damage_level_color(damage_level: DamageLevel) -> str:
    """Get color indicator for damage level."""
    colors = {
        DamageLevel.GOOD: "🟢",
        DamageLevel.MINOR: "🟡",
        DamageLevel.MODERATE: "🟠",
        DamageLevel.SEVERE: "🔴",
        DamageLevel.CRITICAL: "⛔",
    }
    return colors.get(damage_level, "⚪")


def get_damage_level_color_hex(damage_level: DamageLevel) -> str:
    """Get hex color for damage level."""
    colors = {
        DamageLevel.GOOD: "#1DB954",
        DamageLevel.MINOR: "#FFD700",
        DamageLevel.MODERATE: "#FF8C00",
        DamageLevel.SEVERE: "#DC143C",
        DamageLevel.CRITICAL: "#8B0000",
    }
    return colors.get(damage_level, "#808080")


def analyze_stem_quality(stem_name: str, audio: np.ndarray, sample_rate: int,
                         artifact_metrics: Optional[Dict] = None) -> StemQualityReport:
    """
    Analyze the quality of a stem and return a quality report.
    
    Args:
        stem_name: Name of the stem (e.g., "vocals", "drums")
        audio: Audio samples (float32, range [-1, 1])
        sample_rate: Sample rate in Hz
        artifact_metrics: Optional artifact scores from QualityMetrics
        
    Returns:
        StemQualityReport with damage assessment
    """
    try:
        detector = QualityDetector()
        report = detector.analyze(audio, sample_rate, artifact_metrics)
        logger.info(f"Quality analysis for {stem_name}: {report.damage_level.value}")
        return report
    except Exception as e:
        logger.error(f"Quality analysis failed for {stem_name}: {e}")
        # Return a default "good" report on error
        return StemQualityReport(
            damage_level=DamageLevel.GOOD,
            confidence=0.0,
            needs_regeneration=False,
        )


def regenerate_stem(stem_name: str, audio: np.ndarray, sample_rate: int,
                    style_description: str = "",
                    blend_ratio: float = 0.5,
                    musical_profile: Optional[Dict[str, Any]] = None,
                    use_bssr: bool = True) -> Tuple[np.ndarray, RegenerationSummary]:
    """
    Regenerate a stem using JASCO AI.
    
    Args:
        stem_name: Name of the stem to regenerate
        audio: Original audio samples
        sample_rate: Sample rate in Hz
        style_description: Optional style description for JASCO conditioning
        blend_ratio: Blend ratio (0-1) original vs regenerated
        musical_profile: Musical profile for conditioning
        use_bssr: Whether to use BSSR pipeline for long audio
        
    Returns:
        Tuple of (regenerated audio, summary)
    """
    try:
        logger.info(f"Starting regeneration for stem: {stem_name}")
        
        # Create regenerator
        regenerator = StemRegenerator(use_bssr=use_bssr)
        
        # Create plan
        plan = regenerator.create_regeneration_plan(
            audio=audio,
            stem_name=stem_name,
            sample_rate=sample_rate,
            musical_profile=musical_profile or {},
        )
        
        # Execute regeneration
        regenerated_audio, summary = regenerator.execute_regeneration_plan(plan)
        
        # Apply blending if needed
        if blend_ratio < 1.0:
            blender = Blender()
            regenerated_audio = blender.blend_with_loudness_match(
                original=audio,
                regenerated=regenerated_audio,
                start_time=0.0,
                end_time=len(audio) / sample_rate,
                fade_duration=0.1,
                sample_rate=sample_rate,
            )
        
        logger.info(f"Regeneration complete for {stem_name}: {summary.regions_regenerated} regions")
        return regenerated_audio, summary
        
    except Exception as e:
        logger.error(f"Regeneration failed for {stem_name}: {e}")
        # Return original on failure
        summary = RegenerationSummary(
            stem_name=stem_name,
            original_duration=len(audio) / sample_rate,
            regenerated_duration=len(audio) / sample_rate,
            regions_regenerated=0,
            total_regenerated_time=0.0,
            percent_regenerated=0.0,
            use_whole_stem=False,
            regions=[],
            success=False,
            error=str(e),
        )
        return audio.copy(), summary


def display_quality_report(report: StemQualityReport, stem_name: str = "") -> None:
    """
    Display a quality report in the Streamlit UI.
    
    Args:
        report: StemQualityReport to display
        stem_name: Optional stem name for display
    """
    # Header with damage level
    damage_emoji = get_damage_level_color(report.damage_level)
    damage_color = get_damage_level_color_hex(report.damage_level)
    
    st.markdown(f"**{damage_emoji} Quality Assessment:** <span style='color:{damage_color}'>{report.damage_level.value.upper()}</span>", 
                unsafe_allow_html=True)
    
    # Confidence
    st.caption(f"Confidence: {report.confidence:.0%}")
    
    # Component scores
    st.markdown("##### Damage Scores")
    
    # Create columns for scores
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Clipping", f"{report.clipping_score:.2f}")
        st.progress(min(report.clipping_score, 1.0))
    
    with c2:
        st.metric("Distortion", f"{report.distortion_score:.2f}")
        st.progress(min(report.distortion_score, 1.0))
    
    with c3:
        st.metric("Noise", f"{report.noise_score:.2f}")
        st.progress(min(report.noise_score, 1.0))
    
    c4, c5, c6 = st.columns(3)
    
    with c4:
        st.metric("Artifacts", f"{report.artifact_score:.2f}")
        st.progress(min(report.artifact_score, 1.0))
    
    with c5:
        st.metric("Spectral", f"{report.spectral_score:.2f}")
        st.progress(min(report.spectral_score, 1.0))
    
    # SNR if available
    if report.snr_db is not None:
        with c6:
            st.metric("SNR", f"{report.snr_db:.1f} dB")
    
    # Regeneration recommendation
    if report.needs_regeneration:
        st.warning(f"⚠️ **Regeneration Recommended** - {len(report.regenerate_regions)} regions identified")
    else:
        st.success("✅ No regeneration needed")


def display_regeneration_plan(plan: RegenerationPlan) -> None:
    """
    Display a regeneration plan summary in the Streamlit UI.
    
    Args:
        plan: RegenerationPlan to display
    """
    st.markdown("##### 📋 Regeneration Plan")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Stem", plan.stem_name)
    
    with col2:
        st.metric("Regions to Regenerate", len(plan.regions))
    
    with col3:
        mode = "Whole Stem" if plan.use_whole_stem else "Selective"
        st.metric("Mode", mode)
    
    # Region details
    if plan.regions:
        st.markdown("##### 🗺️ Regeneration Regions")
        
        for i, region in enumerate(plan.regions):
            col_reg1, col_reg2, col_reg3 = st.columns([1, 2, 1])
            
            with col_reg1:
                damage_emoji = get_damage_level_color(region.damage_level)
                st.write(f"{damage_emoji} Region {i+1}")
            
            with col_reg2:
                duration = region.end_time - region.start_time
                st.write(f"⏱️ {region.start_time:.1f}s → {region.end_time:.1f}s ({duration:.1f}s)")
            
            with col_reg3:
                st.write(f"📊 {region.confidence:.0%} confidence")
        
        # Timeline visualization
        if plan.regions:
            st.markdown("##### 📊 Timeline Overview")
            audio_duration = len(plan.original_audio) / plan.sample_rate
            
            # Create a simple timeline
            timeline_fig = go.Figure()
            
            # Add audio waveform in background (simplified)
            timeline_fig.add_trace(go.Scatter(
                x=[0, audio_duration],
                y=[0, 0],
                mode='lines',
                line=dict(color='gray', width=2),
                name='Audio Duration',
                fill='tozeroy',
                fillcolor='rgba(128, 128, 128, 0.1)',
            ))
            
            # Add regeneration regions
            for i, region in enumerate(plan.regions):
                color = get_damage_level_color_hex(region.damage_level)
                timeline_fig.add_trace(go.Scatter(
                    x=[region.start_time, region.end_time],
                    y=[1, 1],
                    mode='markers',
                    marker=dict(size=20, color=color, symbol='square'),
                    name=f"Region {i+1}: {region.damage_level.value}",
                ))
            
            timeline_fig.update_layout(
                title="Regeneration Regions",
                xaxis_title="Time (s)",
                yaxis=dict(showticklabels=False, range=[-0.5, 1.5]),
                height=100,
                margin=dict(l=20, r=20, t=30, b=20),
                template="plotly_dark",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            st.plotly_chart(timeline_fig, width='stretch', key="timeline")
    else:
        st.info("No regeneration regions identified.")


def display_regeneration_summary(summary: RegenerationSummary) -> None:
    """
    Display a regeneration summary after completion.
    
    Args:
        summary: RegenerationSummary to display
    """
    st.markdown("##### ✅ Regeneration Complete")
    
    if summary.success:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Regions Regenerated", summary.regions_regenerated)
        
        with col2:
            st.metric("Time Regenerated", f"{summary.total_regenerated_time:.1f}s")
        
        with col3:
            st.metric("Percent Regenerated", f"{summary.percent_regenerated:.1f}%")
        
        with col4:
            mode = "Whole Stem" if summary.use_whole_stem else "Selective"
            st.metric("Mode", mode)
        
        st.success(f"Successfully regenerated {summary.regions_regenerated} region(s) for {summary.stem_name}")
    else:
        st.error(f"Regeneration failed: {summary.error}")


def display_stem_quality_cards(quality_reports: Dict[str, StemQualityReport]) -> None:
    """
    Display quality assessment cards for all stems.
    
    Args:
        quality_reports: Dictionary mapping stem names to quality reports
    """
    st.markdown("##### 🎯 Stem Quality Assessment")
    
    for stem_name, report in quality_reports.items():
        damage_emoji = get_damage_level_color(report.damage_level)
        damage_color = get_damage_level_color_hex(report.damage_level)
        
        with st.expander(f"{damage_emoji} {stem_name.title()}: {report.damage_level.value.upper()}", 
                        expanded=report.needs_regeneration):
            
            # Quick metrics
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric("Clipping", f"{report.clipping_score:.0%}")
                
            with m2:
                st.metric("Distortion", f"{report.distortion_score:.0%}")
                
            with m3:
                st.metric("Artifacts", f"{report.artifact_score:.0%}")
            
            # Regeneration status
            if report.needs_regeneration:
                st.warning(f"⚠️ Needs regeneration: {len(report.regenerate_regions)} regions")
                
                # Show regions
                if report.regenerate_regions:
                    st.markdown("**Regions to regenerate:**")
                    for i, (start, end) in enumerate(report.regenerate_regions):
                        st.text(f"  • {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
            else:
                st.success("✅ Quality is acceptable")


def main():
    """Main Streamlit application."""
    
    # Initialize session state for A/B comparison
    if 'ab_mode' not in st.session_state:
        st.session_state.ab_mode = 'processed'  # 'original' or 'processed'
    
    # Header
    st.markdown('<div class="main-header">🎵 Resonate - Live Music Reconstruction</div>', 
                unsafe_allow_html=True)
    st.markdown("Transform phone recordings into studio-quality audio using ML source separation and DSP enhancement.")
    
    # Sidebar - Configuration
    with st.sidebar:
        # ===== PRE-CONDITIONING SECTION =====
        st.subheader("🎚️ Pre-Conditioning")
        st.caption("Clean input audio BEFORE separation (critical for phone recordings)")
        
        enable_precondition = st.toggle(
            "Enable Pre-Conditioning",
            value=True,
            help="Highly recommended ON for phone recordings. Cleans audio before separation."
        )
        
        precondition_noise_strength = 0.5
        precondition_declip = True
        precondition_dynamics = True
        
        if enable_precondition:
            precondition_noise_strength = st.slider(
                "Noise Reduction Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher = more noise removed, but may affect music. 0.5 is balanced."
            )
            
            precondition_declip = st.toggle(
                "De-Clipping",
                value=True,
                help="Repair distorted peaks from phone AGC saturation"
            )
            
            precondition_dynamics = st.toggle(
                "Dynamics Restoration",
                value=True,
                help="Restore dynamic range compressed by phone AGC"
            )
        
        st.divider()

        st.header("⚙️ Processing Settings")
        
        # Processing mode
        mode = st.selectbox(
            "Processing Mode",
            ["render", "preview"],
            index=0,
            help="Preview: Fast, lower quality. Render: Slow, best quality."
        )
        
        # Enhancement intensity
        intensity = st.slider(
            "Enhancement Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="How much enhancement to apply to stems."
        )
        
        # Restoration settings
        st.subheader("🔊 Restoration Settings")
        enable_frequency = st.toggle(
            "Frequency Restoration",
            value=True,
            help="Extend frequency response from phone-limited range."
        )
        frequency_intensity = st.slider(
            "Frequency Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Intensity of frequency restoration."
        )
        
        enable_dereverb = st.toggle(
            "Dereverberation",
            value=True,
            help="Remove venue acoustics from recording."
        )
        dereverb_intensity = st.slider(
            "Dereverberation Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Intensity of dereverberation."
        )

        # Neural polishing (MBD)
        st.subheader("✨ Neural Polishing (MBD)")
        enable_mbd = st.toggle(
            "Enable Neural Polishing (MBD)",
            value=False,
            help="Optional high-quality artifact reduction using AudioCraft MBD"
        )
        mbd_intensity = st.slider(
            "Polishing Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Blend amount (0=no change, 1=full polish). ~30s overhead.",
            disabled=not enable_mbd
        )

        st.caption("💡 Use when artifacts persist after restoration")
        
        # Stem Regeneration (Phase 4)
        st.subheader("🧬 Stem Regeneration (Phase 4)")
        st.caption("AI-powered regeneration for heavily damaged stems using JASCO")
        
        # Regeneration enabled toggle
        enable_regeneration = st.toggle(
            "Enable Stem Regeneration",
            value=False,
            help="Regenerate heavily damaged stems using JASCO AI"
        )
        
        if enable_regeneration:
            # Show regeneration settings
            st.markdown("**Regeneration Settings**")
            
            # Style description
            style_description = st.text_input(
                "Style Description",
                value="",
                placeholder="e.g., funky rock, 1970s soul, acoustic folk",
                help="Optional text to guide JASCO regeneration style"
            )
            
            # Blend ratio
            blend_ratio = st.slider(
                "Blend Ratio (Original vs Regenerated)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0 = all original, 1 = all regenerated, 0.5 = equal blend"
            )
            
            st.caption(f"📊 Current blend: {int(blend_ratio*100)}% regenerated / {int((1-blend_ratio)*100)}% original")
            
            # Regeneration progress indicator
            st.markdown("**Progress Tracking**")
            show_progress = st.toggle("Show regeneration progress", value=True)
            
        # Taylor's Version Rerecording
        st.subheader("🎙️ Taylor's Version Rerecording")
        st.caption("Full AI rerecording for studio quality")
        
        taylors_version = st.toggle(
            "Enable Taylor's Version Mode",
            value=False,
            help="100% AI regeneration of all stems (no original audio blended)"
        )
        
        tv_similarity_target = 0.85
        tv_timbre_match = True
        tv_articulation_match = True
        
        if taylors_version:
            tv_similarity_target = st.slider(
                "Similarity Target",
                min_value=0.7,
                max_value=0.95,
                value=0.85,
                help="How closely to match the original (0.85 = very close)"
            )
            
            tv_max_iterations = st.number_input(
                "Max Refinement Iterations",
                min_value=1,
                max_value=5,
                value=1,
                help="Attempt to improve similarity (1 = single pass)"
            )
            
            tv_timbre_match = st.toggle("Timbre Matching", value=True)
            tv_articulation_match = st.toggle("Articulation Matching", value=True)
        
        # BSSR Long-Form Generation
        st.subheader("🎵 Long-Form Generation (BSSR)")
        st.caption("For songs > 30 seconds")
        
        use_bssr = st.toggle(
            "Enable BSSR Pipeline",
            value=True,
            help="Beat-Synchronous Stem-Sequential Regeneration for seamless full-length songs. Required for tracks > 30s."
        )
        
        # Output settings
        st.subheader("📤 Output Settings")
        output_format = st.selectbox(
            "Output Format",
            ["wav", "flac", "mp3"],
            index=0,
            help="WAV: Best quality. FLAC: Compressed lossless. MP3: Smallest file."
        )
        
        target_loudness = st.number_input(
            "Target Loudness (LUFS)",
            value=-14.0,
            step=1.0,
            help="Streaming platforms use -14 LUFS."
        )

        # Quality analysis settings
        st.subheader("📊 Quality Analysis")
        run_quality_analysis = st.toggle(
            "Run Quality Analysis",
            value=True,
            help="Disable if analysis is slow or causes the app to stall."
        )
        analysis_seconds = st.number_input(
            "Analysis Duration (seconds)",
            value=30,
            min_value=5,
            max_value=120,
            step=5,
            help="Analyze only the first N seconds to reduce memory/CPU."
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Settings"):
            separation_model = st.selectbox(
                "Separation Model",
                ["htdemucs_ft", "htdemucs"],
                index=0,
                help="htdemucs_ft: Better quality, slower. htdemucs: Faster, good quality."
            )
            
            mix_mode = st.selectbox(
                "Mix Mode",
                ["enhanced", "natural"],
                index=0,
                help="Enhanced: Optimized for clarity. Natural: Preserve original balance."
            )
        
        st.divider()
        
        # Info
        st.info("💡 **Tip**: Start with Preview mode to test settings, then use Render for final output.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Audio")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Upload a phone recording of a live performance."
        )
        
        if uploaded_file is not None:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Display audio info
            st.audio(uploaded_file)
            
            # Load and analyze
            try:
                from audio_engine.ingest import AudioIngest
                ingest = AudioIngest()
                buffer = ingest.load(tmp_path)
                
                st.markdown("### 📊 Audio Analysis")
                meta = buffer.metadata
                meta_dict = meta.to_dict()
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Duration", meta_dict['duration_formatted'])
                m2.metric("Sample Rate", f"{meta.sample_rate} Hz")
                m3.metric("Channels", meta.channels)
                
                if buffer.analysis:
                    analysis = buffer.analysis
                    st.markdown("#### Quality Metrics")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("SNR", f"{analysis.snr_db:.1f} dB")
                    a2.metric("Spectral Centroid", f"{analysis.spectral_centroid_hz:.0f} Hz")
                    a3.metric("Frequency Range", f"{analysis.frequency_range_hz[0]:.0f}-{analysis.frequency_range_hz[1]:.0f} Hz")
                    
                    if analysis.is_clipped:
                        st.warning(f"⚠️ Audio has clipping ({analysis.clipping_percent:.2f}%)")
                    
                    # Original waveform
                    st.markdown("#### Original Waveform")
                    orig_waveform = plot_waveform(buffer.audio, buffer.sample_rate, "Original Audio")
                    st.plotly_chart(orig_waveform, width='stretch', key="orig_waveform")
                
            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
    
    with col2:
        st.subheader("🎛️ Processing")
        
        if uploaded_file is None:
            st.info("👆 Upload an audio file to begin processing")
        else:
            # Process button with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if st.button("🚀 Process Audio", type="primary"):
                status_text.text("Initializing pipeline...")
                progress_bar.progress(10)

                try:
                    # Create pipeline
                    mode_enum = PipelineMode.RENDER if mode == "render" else PipelineMode.PREVIEW
                    # Build config kwargs and filter to known PipelineConfig fields to prevent unexpected-kw errors
                    config_kwargs = {
                        'mode': mode_enum,
                        'enhancement_intensity': intensity,
                        'separation_model': separation_model,
                        'mix_mode': mix_mode,
                        'output_format': output_format,
                        'target_loudness_lufs': target_loudness,
                        'frequency_restoration': enable_frequency,
                        'frequency_intensity': frequency_intensity,
                        'dereverberation': enable_dereverb,
                        'dereverb_intensity': dereverb_intensity,
                        'enable_mbd_polish': enable_mbd,
                        'mbd_intensity': mbd_intensity,
                        'enable_preconditioning': enable_precondition,
                        'precondition_noise_reduction': enable_precondition,
                        'precondition_noise_strength': precondition_noise_strength,
                        'precondition_declip': precondition_declip,
                        'precondition_dynamics': precondition_dynamics,
                    }

                    valid_keys = {f.name for f in fields(PipelineConfig)}
                    filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_keys}

                    config = PipelineConfig(**filtered_kwargs)

                    pipeline = AudioPipeline(config)

                    status_text.text("Processing audio... This may take a few minutes.")
                    progress_bar.progress(30)

                    # Process
                    result = pipeline.process(tmp_path)
                    progress_bar.progress(90)

                    if result.success:
                        logger.info("✅ Pipeline completed successfully!")
                        logger.info(f"📁 Output file: {result.output_file}")
                        status_text.text("Finalizing... Loading audio files")
                        progress_bar.progress(92)

                        # Load original audio for comparison
                        logger.info("📖 Loading original audio...")
                        t0 = time.time()
                        original_audio, original_sr = sf.read(tmp_path)
                        logger.info(f"   Original loaded: {len(original_audio)} samples ({time.time()-t0:.1f}s)")
                        
                        status_text.text("Finalizing... Loading processed audio")
                        progress_bar.progress(94)
                        
                        logger.info("📖 Loading processed audio...")
                        t0 = time.time()
                        if result.output_file and os.path.exists(result.output_file):
                            processed_audio, processed_sr = sf.read(result.output_file)
                            logger.info(f"   Processed loaded: {len(processed_audio)} samples ({time.time()-t0:.1f}s)")
                        else:
                            logger.error(f"❌ Output file not found: {result.output_file}")
                            st.error(f"Output file not found: {result.output_file}")
                            processed_audio = original_audio  # Fallback
                            processed_sr = original_sr
                        
                        # Store in session state for persistence across reruns
                        st.session_state.processed_audio = processed_audio
                        st.session_state.processed_sr = processed_sr
                        st.session_state.original_audio = original_audio
                        st.session_state.original_sr = original_sr
                        st.session_state.result = result
                        st.session_state.tmp_path = tmp_path
                        st.session_state.processing_complete = True

                        # Analyze quality (can be slow for long audio)
                        if run_quality_analysis:
                            status_text.text("Finalizing... Analyzing original quality")
                            progress_bar.progress(96)
                            
                            logger.info("📊 Analyzing original audio quality...")
                            t0 = time.time()
                            try:
                                original_audio_for_analysis = prepare_analysis_audio(
                                    original_audio, original_sr, analysis_seconds
                                )
                                original_quality = analyze_quality(original_audio_for_analysis, original_sr)
                                logger.info(f"   Original analysis done ({time.time()-t0:.1f}s)")
                            except Exception as e:
                                logger.warning(f"   Original analysis failed: {e}")
                                original_quality = None

                            status_text.text("Finalizing... Analyzing processed quality")
                            progress_bar.progress(98)
                            
                            logger.info("📊 Analyzing processed audio quality...")
                            t0 = time.time()
                            try:
                                processed_audio_for_analysis = prepare_analysis_audio(
                                    processed_audio, processed_sr, analysis_seconds
                                )
                                processed_quality = analyze_quality(processed_audio_for_analysis, processed_sr)
                                logger.info(f"   Processed analysis done ({time.time()-t0:.1f}s)")
                            except Exception as e:
                                logger.warning(f"   Processed analysis failed: {e}")
                                processed_quality = None
                        else:
                            original_quality = None
                            processed_quality = None

                        progress_bar.progress(100)
                        status_text.text("✅ Complete!")
                        logger.info("🎉 Processing complete!")

                        # Success message
                        st.markdown('<div class="success-box">✅ Processing Complete!</div>',
                                    unsafe_allow_html=True)

                        # Processing time
                        st.markdown("#### ⏱️ Processing Time")
                        total_time = result.total_processing_time

                        t1, t2, t3, t4 = st.columns(4)
                        t1.metric("Total", f"{total_time:.1f}s")
                        t2.metric("Separation", f"{result.stage_times.get('separation', 0):.1f}s")
                        t3.metric("Enhancement", f"{result.stage_times.get('enhancement', 0):.1f}s")
                        t4.metric("Restoration", f"{result.stage_times.get('restoration', 0):.1f}s")

                        # A/B Toggle with Regenerated option
                        st.markdown("#### 🔄 A/B Comparison")
                        
                        # Extend A/B mode options if regeneration is available
                        ab_options = ['original', 'processed']
                        if 'regenerated_audio' in st.session_state:
                            ab_options.append('regenerated')
                        
                        col_ab1, col_ab2 = st.columns([1, 3])
                        with col_ab1:
                            if st.button("🔀 Toggle A/B"):
                                current_idx = ab_options.index(st.session_state.ab_mode)
                                next_idx = (current_idx + 1) % len(ab_options)
                                st.session_state.ab_mode = ab_options[next_idx]

                        with col_ab2:
                            mode_labels = {
                                'original': 'Original',
                                'processed': 'Processed',
                                'regenerated': 'Regenerated (JASCO)'
                            }
                            st.markdown(f"**Currently playing:** {mode_labels.get(st.session_state.ab_mode, st.session_state.ab_mode)}")

                        # Show appropriate audio based on A/B mode
                        if st.session_state.ab_mode == 'original':
                            st.audio(tmp_path)
                        elif st.session_state.ab_mode == 'regenerated':
                            if 'regenerated_audio' in st.session_state:
                                # Save regenerated audio to temp file for playback
                                regen_path = tmp_path.replace('.wav', '_regenerated.wav')
                                sf.write(regen_path, st.session_state.regenerated_audio, 44100)
                                st.audio(regen_path)
                            else:
                                st.warning("No regenerated audio available")
                                st.audio(result.output_file)
                        else:
                            st.audio(result.output_file)

                        # Waveform comparison
                        st.markdown("#### 📈 Waveform Comparison")
                        col_wave1, col_wave2 = st.columns(2)

                        with col_wave1:
                            st.markdown("**Original**")
                            orig_fig = plot_waveform(original_audio, 44100, "Original")
                            st.plotly_chart(orig_fig, width='stretch', key="comp_orig_wave")

                        with col_wave2:
                            st.markdown("**Processed**")
                            proc_fig = plot_waveform(processed_audio, 44100, "Processed")
                            st.plotly_chart(proc_fig, width='stretch', key="comp_proc_wave")

                        # Quality metrics comparison
                        st.markdown("#### 📊 Quality Metrics Comparison")

                        if processed_quality and original_quality:
                            # Create comparison metrics
                            snr_improvement = processed_quality.snr_db - original_quality.snr_db
                            loudness_shift = processed_quality.loudness_lufs - original_quality.loudness_lufs

                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("SNR", f"{processed_quality.snr_db:.1f} dB", f"{snr_improvement:+.1f} dB")
                            q2.metric("Loudness", f"{processed_quality.loudness_lufs:.1f} LUFS", f"{loudness_shift:+.1f} LUFS")
                            q3.metric("Spectral Centroid", f"{processed_quality.spectral_centroid_hz:.0f} Hz")
                            q4.metric("Dynamic Range", f"{processed_quality.dynamic_range_db:.1f} dB")

                            # Artifact detection
                            st.markdown("#### 🔍 Artifact Analysis")
                            artifacts = processed_quality.artifacts
                            for artifact_name, score in artifacts.items():
                                st.progress(min(score, 1.0), text=f"{artifact_name.replace('_', ' ').title()}: {score:.2f}")
                        else:
                            st.warning("⚠️ Quality analysis not available")

                        # Output file
                        st.markdown("#### 📤 Output")
                        st.markdown(f"**File:** {result.output_file}")

                        # Download button
                        with open(result.output_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Result",
                                data=f,
                                file_name=Path(result.output_file).name,
                                mime=f"audio/{output_format}"
                            )
                        
                        # Store quality analysis results in session state
                        st.session_state.original_quality = original_quality
                        st.session_state.processed_quality = processed_quality

                    else:
                        progress_bar.progress(100)
                        status_text.text("Failed!")
                        st.error(f"❌ Processing failed: {result.error_message}")

                    # Cleanup
                    pipeline.cleanup()

                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text("Error!")
                    st.error(f"Error during processing: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # =========================================================================
    # STEM REGENERATION SECTION (Phase 4) - OUTSIDE BUTTON BLOCK
    # =========================================================================
    # This section persists across reruns because it uses session state
    if st.session_state.get('processing_complete', False):
        st.markdown("---")
        st.markdown("## 🧬 Stem Regeneration (Phase 4)")
        st.markdown("Use JASCO AI to regenerate heavily damaged stems that restoration couldn't fix.")
        
        # Get stored data from session state
        processed_audio = st.session_state.processed_audio
        processed_sr = st.session_state.processed_sr
        processed_quality = st.session_state.get('processed_quality')
        
        # Initialize session state for regeneration
        if 'quality_reports' not in st.session_state:
            st.session_state.quality_reports = {}
        if 'regeneration_plan' not in st.session_state:
            st.session_state.regeneration_plan = None
        if 'regenerated_audio' not in st.session_state:
            st.session_state.regenerated_audio = None
        if 'stem_to_regenerate' not in st.session_state:
            st.session_state.stem_to_regenerate = 'vocals'
        
        # Regeneration controls
        regen_col1, regen_col2 = st.columns([1, 2])
        
        with regen_col1:
            st.markdown("#### 🎛️ Regeneration Controls")
            
            # Select stem to regenerate
            available_stems = ['vocals', 'drums', 'bass', 'other']
            
            # Find current index from session state
            try:
                current_idx = available_stems.index(st.session_state.stem_to_regenerate)
            except (ValueError, TypeError):
                current_idx = 0
            
            # selectbox returns the VALUE directly (stem name), not index
            stem_to_regenerate = st.selectbox(
                "Select Stem to Regenerate",
                available_stems,
                index=current_idx,
                help="Choose which stem to analyze and potentially regenerate",
                key="stem_selectbox"
            )
            
            # Update session state with the stem NAME
            st.session_state.stem_to_regenerate = stem_to_regenerate
            
            # Analyze quality button
            if st.button("🔍 Analyze Stem Quality", type="secondary"):
                with st.spinner(f"Analyzing {stem_to_regenerate} quality..."):
                    try:
                        # Use processed audio for analysis (from first 30 seconds)
                        analysis_audio = prepare_analysis_audio(
                            processed_audio, processed_sr, analysis_seconds
                        )
                        quality_report = analyze_stem_quality(
                            stem_name=stem_to_regenerate,
                            audio=analysis_audio,
                            sample_rate=processed_sr,
                            artifact_metrics=processed_quality.artifacts if processed_quality else None
                        )
                        st.session_state.quality_reports[stem_to_regenerate] = quality_report
                        st.success(f"Analysis complete for {stem_to_regenerate}")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
        
        with regen_col2:
            st.markdown("#### 📊 Stem Quality Overview")
            
            if st.session_state.quality_reports:
                # Display quality cards
                display_stem_quality_cards(st.session_state.quality_reports)
            else:
                st.info("👆 Click 'Analyze Stem Quality' to assess a stem")
        
        # Regeneration plan and execution
        if stem_to_regenerate in st.session_state.quality_reports:
            report = st.session_state.quality_reports[stem_to_regenerate]
            
            st.markdown("#### 🔧 Regeneration Settings")
            
            rg_col1, rg_col2, rg_col3 = st.columns(3)
            
            # Wrap regeneration inputs in a form so pressing Enter in the Style Description submits
            with st.form(key=f"regen_form_{stem_to_regenerate}"):
                with rg_col1:
                    # Multi-line prompt so Enter inserts a newline and does not submit the form
                    style_description = st.text_area(
                        "Style Description (optional)",
                        value="",
                        placeholder="e.g., funky rock, acoustic. Describe tone and what to preserve (timing, lyrics).",
                        height=120,
                        help="Multi-line prompt: press Enter for new lines (doesn't submit)."
                    )

                with rg_col2:
                    blend_ratio = st.slider(
                        "Blend Ratio",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        help="0 = all original, 1 = all regenerated"
                    )

                with rg_col3:
                    st.metric("Blend", f"{int(blend_ratio*100)}% regenerated")

                # Allow user to force regeneration even when recommendation is to skip
                force_regen = st.checkbox(
                    "Force Regenerate (override recommendation)",
                    value=False,
                    help="Check to regenerate even if the detector does not recommend it"
                )

                # Submit button — prefer explicit click; pressing Enter in text areas will not submit
                submit_regen = st.form_submit_button("🚀 Apply Regeneration")

            # End of form
            # Helpful quick-action: Auto-Repair (safe) preset — conservative settings, snapshot, preview
            auto_repair = st.button("🛠️ Auto-Repair (safe)", help="Apply conservative preconditioning + selective regeneration (Blend 40%) and show preview before applying")


            # Explanatory caption to avoid confusion about metric meaning
            st.caption("Scores are 0–100% (higher = more damaged). Overall recommendation uses a weighted score, so a single high metric doesn't always require regeneration.")

            st.markdown("##### 📋 Regeneration Plan")

            try:
                # Prefer using the isolated enhanced stem if available
                enhanced_stems = None
                if st.session_state.get('result') and getattr(st.session_state['result'], 'enhancement_result', None):
                    enhanced_stems = st.session_state['result'].enhancement_result.stems

                if enhanced_stems and stem_to_regenerate in enhanced_stems:
                    stem_source_audio = enhanced_stems[stem_to_regenerate]
                    source_tag = "enhanced stem"
                else:
                    stem_source_audio = processed_audio
                    source_tag = "processed mix (fallback)"

                st.caption(f"Using {source_tag} for planning and regeneration")

                regenerator = StemRegenerator(use_bssr=use_bssr)
                # Pass user style description into generator config if provided
                if style_description:
                    try:
                        regenerator.generator.config.style_description = style_description
                    except Exception:
                        pass

                plan = regenerator.create_regeneration_plan(
                    audio=stem_source_audio,
                    stem_name=stem_to_regenerate,
                    sample_rate=processed_sr,
                    musical_profile={},
                )
                st.session_state.regeneration_plan = plan

                display_regeneration_plan(plan)

                # Handle Auto-Repair (safe) quick-action
                if auto_repair:
                    st.info("Running Auto-Repair (safe): conservative preconditioning + selective regeneration with Blend 40% (preview only)")
                    if not plan.regions:
                        st.warning("No regions detected — Auto-Repair will not perform whole-stem regeneration automatically. To force a whole-stem regeneration, enable 'Regenerate whole stem' or use the Apply button after a preview.")
                    else:
                        try:
                            plan_to_execute = plan
                            # Conservative blend for Auto-Repair
                            auto_blend = 0.4

                            # Use style description if provided
                            if style_description:
                                try:
                                    regenerator.generator.config.style_description = style_description
                                except Exception:
                                    pass

                            regenerated_stem, summary = regenerator.execute_regeneration_plan(plan_to_execute)

                            # Blend
                            blender = Blender()
                            regenerated_stem = blender.blend_with_loudness_match(
                                original=stem_source_audio,
                                regenerated=regenerated_stem,
                                start_time=0.0,
                                end_time=len(stem_source_audio)/processed_sr,
                                fade_duration=0.1,
                                sample_rate=processed_sr,
                            )
                            final_stem = (1.0 - auto_blend) * stem_source_audio + auto_blend * regenerated_stem

                            # Store candidate preview
                            st.session_state.candidate_regen = {
                                'stem_name': stem_to_regenerate,
                                'final_stem': final_stem,
                                'regenerated_mix': None,
                                'sr': processed_sr,
                                'summary': summary,
                            }

                            if enhanced_stems is not None:
                                temp_stems = dict(enhanced_stems)
                                temp_stems[stem_to_regenerate] = final_stem
                                mixer = StemMixer()
                                mix_res = mixer.mix(temp_stems, processed_sr)
                                st.session_state.candidate_regen['regenerated_mix'] = mix_res.mixed_audio

                            st.success("Auto-Repair complete — preview available. Use Apply to commit or Discard to cancel.")

                        except Exception as e:
                            st.error(f"Auto-Repair failed: {e}")

                # Auto-Repair All: one-click conservative apply across available stems
                # Auto-Repair All controls
                auto_repair_all = st.button("🧰 Auto-Repair All Stems (one-click)", help="Conservatively repair all stems that need it and apply changes. Snapshots are created so you can Undo.")

                # Preference to skip confirmation
                if 'auto_repair_all_auto_apply' not in st.session_state:
                    st.session_state.auto_repair_all_auto_apply = False

                auto_apply_pref = st.checkbox("Auto-apply Auto-Repair All (skip confirmation)", value=st.session_state.auto_repair_all_auto_apply, help="If checked, Auto-Repair All will run and apply changes without asking for further confirmation")
                st.session_state.auto_repair_all_auto_apply = auto_apply_pref

                # Optional aggressive mode: force whole-stem regeneration when no regions detected
                if 'auto_repair_all_aggressive' not in st.session_state:
                    st.session_state.auto_repair_all_aggressive = False
                aggressive_pref = st.checkbox("Aggressive: force whole-stem for stems with no detected regions", value=st.session_state.auto_repair_all_aggressive, help="If checked, stems with no detected regions will be regenerated whole-stem")
                st.session_state.auto_repair_all_aggressive = aggressive_pref

                # QUICK REMASTER: analyze -> repair flagged stems -> mix -> master
                st.markdown("---")
                st.write("### 🎚️ Quick Remaster (Analyze → Remaster)")
                if 'quick_remaster_use_ai' not in st.session_state:
                    st.session_state.quick_remaster_use_ai = True
                quick_use_ai = st.checkbox("Use AI to repair stems if needed", value=st.session_state.quick_remaster_use_ai)
                st.session_state.quick_remaster_use_ai = quick_use_ai

                if 'quick_remaster_aggressive' not in st.session_state:
                    st.session_state.quick_remaster_aggressive = False
                quick_aggr = st.checkbox("Aggressive: allow whole-stem repair when no regions detected", value=st.session_state.quick_remaster_aggressive)
                st.session_state.quick_remaster_aggressive = quick_aggr

                quick_remaster = st.button("🎚️ Quick Remaster (one-click)", help="Analyze stems, repair only those that need it (or all if aggressive), then automatically mix and master the final track")

                if quick_remaster:
                    try:
                        import logging
                        quick_rem_logger = logging.getLogger('resonate.regen')
                        quick_rem_logger.info('Quick Remaster: user initiated')

                        if enhanced_stems is None:
                            st.error('No enhanced stems available to remaster')
                        else:
                            from audio_engine.generation.regeneration_utils import remaster_project

                            st.info('Analyzing stems and preparing remaster...')

                            results = remaster_project(dict(enhanced_stems), processed_sr, use_ai_if_needed=st.session_state.quick_remaster_use_ai, aggressive=st.session_state.quick_remaster_aggressive, auto_blend=0.4)

                            # Store a candidate master in session for preview/download
                            st.session_state.candidate_master = {
                                'mastering': results['mastering'],
                                'mix_result': results['mix_result'],
                                'stems_repaired': results.get('stems_repaired', []),
                                'reports': results.get('reports', {}),
                            }

                            st.success('Remaster complete — preview available below.')

                            # Show summary and allow preview / download
                            if results.get('stems_repaired'):
                                st.markdown(f"**Repaired stems:** {', '.join(results['stems_repaired'])}")
                            else:
                                st.markdown("**Repaired stems:** None")

                            master_res = results['mastering']
                            st.markdown(f"**Master loudness:** {getattr(master_res, 'loudness_lufs', 'N/A')} LUFS")
                            st.markdown(f"**True peak:** {getattr(master_res, 'true_peak_db', 'N/A')} dB")

                            # Allow playback of mastered audio if available
                            try:
                                if hasattr(master_res, 'audio') and master_res.audio is not None:
                                    st.audio(master_res.audio, sample_rate=processed_sr)

                                # Offer a download link if file path was produced
                                if hasattr(master_res, 'file_path'):
                                    st.markdown(f"**Master file:** `{master_res.file_path}`")

                            except Exception as e:
                                st.warning(f"Could not preview mastered audio: {e}")

                    except Exception as e:
                        st.error(f"Quick Remaster failed: {e}")

                if auto_repair_all:
                    st.info("Auto-Repair All queued.")
                    if st.session_state.auto_repair_all_auto_apply:
                        run_all = True
                    else:
                        st.warning("Auto-Repair All will run conservative selective regeneration on all available stems and apply changes. Snapshots are created for undo. This may take a while.")
                        run_all = st.button("Run Auto-Repair All (apply)")

                    if run_all:
                        try:
                            import logging
                            regen_logger = logging.getLogger('resonate.regen')
                            regen_logger.info("Auto-Repair All: user initiated run_all")

                            from audio_engine.generation.regeneration_utils import auto_repair_all_stems

                            if enhanced_stems is None:
                                st.error("No enhanced stems available to auto-repair")
                            else:
                                # Snapshot dict
                                if 'regeneration_snapshots' not in st.session_state:
                                    st.session_state.regeneration_snapshots = {}

                                # Run helper
                                results = auto_repair_all_stems(dict(enhanced_stems), processed_sr, regenerator=regenerator, auto_blend=0.4, require_regions=not st.session_state.auto_repair_all_aggressive)

                                # Commit stems and store snapshots
                                for sname, snap in results['snapshots'].items():
                                    st.session_state.regeneration_snapshots[sname] = snap

                                st.session_state.result.enhancement_result.stems = results['stems']

                                # Recompute mix and store
                                mixer = StemMixer()
                                mix_res = mixer.mix(results['stems'], processed_sr)
                                st.session_state.regenerated_mix = mix_res.mixed_audio
                                st.session_state.regenerated_mix_sr = processed_sr

                                st.success("Auto-Repair All complete — changes applied. Use Undo to revert last run per stem.")

                                # Re-run analysis automatically so the user doesn't need to click Process again
                                try:
                                    for sname, audio in results['stems'].items():
                                        analysis_audio = prepare_analysis_audio(audio, processed_sr, analysis_seconds)
                                        report = analyze_stem_quality(stem_name=sname, audio=analysis_audio, sample_rate=processed_sr)
                                        st.session_state.quality_reports[sname] = report
                                    regen_logger.info("Auto-Repair All: re-analysis complete for all stems")
                                except Exception as e:
                                    regen_logger.exception(f"Auto-Repair All: failed to re-analyze stems: {e}")

                                # Show a short results summary
                                processed = [p['stem'] for p in results['results']['processed']]
                                skipped = results['results']['skipped']
                                errors = results['results']['errors']

                                st.markdown(f"**Processed:** {len(processed)} stem(s): {', '.join(processed) if processed else 'None'}")
                                st.markdown(f"**Skipped (no detected regions):** {len(skipped)} stem(s): {', '.join(skipped) if skipped else 'None'}")
                                if errors:
                                    st.markdown(f"**Errors:** {len(errors)} — see logs for details")

                                # Offer to show logs
                                if st.button("Show recent regeneration logs"):
                                    try:
                                        from logging_config import LOG_FILE
                                        with open(LOG_FILE, 'r') as lf:
                                            lines = lf.readlines()[-200:]
                                        st.text('\n'.join(lines[-50:]))
                                    except Exception as e:
                                        st.error(f"Failed to read log file: {e}")

                        except Exception as e:
                            st.error(f"Auto-Repair All failed: {e}")

                # Regeneration target selection
                target_opts = ["All detected regions"]
                if plan.regions:
                    region_labels = [f"Region {i+1}: {r.start_time:.1f}s – {r.end_time:.1f}s" for i, r in enumerate(plan.regions)]
                    target_opts.extend(region_labels)
                target_opts.append("Whole Stem")
                target_opts.append("Custom (start/end)")

                target_choice = st.selectbox("Target", target_opts)

                # Region-specific style prompt (optional) - only shown when selecting a specific region
                region_style = ""
                selected_idx = None
                custom_start = None
                custom_end = None

                if target_choice.startswith("Region"):
                    selected_idx = int(target_choice.split()[1].rstrip(':')) - 1
                    region_style = st.text_area("Region Style Prompt (optional)", value="", placeholder="e.g., David Guetta Future Rave style...",
                                               height=80)
                elif target_choice == "Custom (start/end)":
                    cs, ce = st.columns(2)
                    with cs:
                        custom_start = st.number_input("Start time (s)", value=0.0, min_value=0.0, step=0.1)
                    with ce:
                        custom_end = st.number_input("End time (s)", value=min(3.0, len(stem_source_audio)/processed_sr), min_value=0.0, step=0.1)

                # If region_style empty, default to global style_description
                if not region_style:
                    region_style = style_description

                # Allow whole-stem override
                regen_whole = st.checkbox("Regenerate whole stem (override)", value=False)

                # Require explicit confirmation when user selects Whole Stem but no regions were detected
                whole_confirm = False
                if target_choice == "Whole Stem" and not plan.regions:
                    st.warning("No regions were detected for selective regeneration. Whole-stem regeneration will replace the entire stem.")
                    whole_confirm = st.checkbox("I confirm whole-stem regeneration (this will replace the entire stem)", value=False)

                # Execute if form submitted
                if submit_regen:
                    regen_progress = st.progress(0)
                    regen_status = st.empty()
                    regen_status.text("Starting regeneration...")
                    regen_progress.progress(10)

                    # Build plan based on selection
                    plan_to_execute = plan

                    # If user requested whole stem, require confirmation when no regions exist
                    if regen_whole or (target_choice == "Whole Stem" and (plan.regions or whole_confirm)):
                        plan_to_execute.use_whole_stem = True
                        duration = len(stem_source_audio) / processed_sr
                        plan_to_execute.regions = [RegenerationRegion(0.0, duration, stem_to_regenerate, plan.quality_report.damage_level, quality_report=plan.quality_report, confidence=plan.quality_report.confidence)]

                    elif target_choice.startswith("Region") and plan.regions:
                        plan_to_execute.regions = [plan.regions[selected_idx]]

                    elif target_choice == "Custom (start/end)":
                        if custom_end is None or custom_end <= custom_start:
                            st.error("Invalid custom region: ensure end > start")
                            plan_to_execute.regions = []
                        else:
                            plan_to_execute.regions = [RegenerationRegion(custom_start, custom_end, stem_to_regenerate, plan.quality_report.damage_level, quality_report=plan.quality_report, confidence=plan.quality_report.confidence)]

                    if not plan_to_execute.regions and not plan_to_execute.use_whole_stem:
                        st.warning("No regions selected for regeneration — use 'Regenerate whole stem' to force a full regeneration or pick a region.")
                        regen_progress.progress(100)
                        regen_status.text("Cancelled: no regions selected")
                    else:
                        try:
                            # Apply region-specific style prompt if provided
                            if region_style:
                                try:
                                    regenerator.generator.config.style_description = region_style
                                except Exception:
                                    pass
                            elif style_description:
                                try:
                                    regenerator.generator.config.style_description = style_description
                                except Exception:
                                    pass

                            # Execute regeneration on isolated stem
                            regenerated_stem, summary = regenerator.execute_regeneration_plan(plan_to_execute)
                            regen_progress.progress(60)

                            # Blend per-stem if requested
                            if blend_ratio < 1.0:
                                blender = Blender()
                                regenerated_stem = blender.blend_with_loudness_match(
                                    original=stem_source_audio,
                                    regenerated=regenerated_stem,
                                    start_time=0.0,
                                    end_time=len(stem_source_audio)/processed_sr,
                                    fade_duration=0.1,
                                    sample_rate=processed_sr,
                                )
                                final_stem = (1.0 - blend_ratio) * stem_source_audio + blend_ratio * regenerated_stem
                            else:
                                final_stem = regenerated_stem

                            regen_progress.progress(80)

                            # Store candidate regeneration (preview) instead of immediately replacing session state
                            st.session_state.candidate_regen = {
                                'stem_name': stem_to_regenerate,
                                'final_stem': final_stem,
                                'regenerated_mix': None,
                                'sr': processed_sr,
                                'summary': summary,
                            }

                            # If enhanced stems available, compute a preview mix but don't commit
                            if enhanced_stems is not None:
                                temp_stems = dict(enhanced_stems)
                                temp_stems[stem_to_regenerate] = final_stem
                                mixer = StemMixer()
                                mix_res = mixer.mix(temp_stems, processed_sr)
                                regenerated_mix = mix_res.mixed_audio
                                st.session_state.candidate_regen['regenerated_mix'] = regenerated_mix

                            regen_progress.progress(100)
                            regen_status.text("✅ Regeneration complete — preview available. Use Apply to commit or Discard to cancel.")

                            # Display results and preview UI
                            display_regeneration_summary(summary)
                            st.markdown("##### 📈 Regenerated Stem Waveform (preview)")
                            regen_fig = plot_waveform(final_stem, processed_sr, "Regenerated Stem (preview)")
                            st.plotly_chart(regen_fig, width='stretch', key="regen_wave_preview")

                            if st.session_state.candidate_regen.get('regenerated_mix') is not None:
                                with tempfile.NamedTemporaryFile(delete=False, suffix="_regenerated_mix_preview.wav") as tmpr:
                                    sf.write(tmpr.name, st.session_state.candidate_regen['regenerated_mix'], processed_sr)
                                    regen_mix_path = tmpr.name

                                st.markdown("##### ▶️ Preview Regenerated Mix (preview)")
                                st.audio(regen_mix_path)

                                with open(regen_mix_path, "rb") as f:
                                    st.download_button(
                                        label="⬇️ Download Regenerated Mix (preview)",
                                        data=f,
                                        file_name=f"regenerated_mix_preview_{stem_to_regenerate}.wav",
                                        mime="audio/wav"
                                    )

                            # Show Apply / Discard buttons
                            apply_col, discard_col = st.columns(2)
                            if apply_col.button("✅ Apply Regeneration"):
                                # Snapshot current stem for undo
                                if 'regeneration_snapshots' not in st.session_state:
                                    st.session_state.regeneration_snapshots = {}
                                prev_stem = enhanced_stems.get(stem_to_regenerate) if enhanced_stems is not None else None
                                st.session_state.regeneration_snapshots[stem_to_regenerate] = prev_stem

                                # Commit candidate into session state
                                if enhanced_stems is not None:
                                    enhanced_stems[stem_to_regenerate] = final_stem
                                    st.session_state.result.enhancement_result.stems = enhanced_stems

                                    # Recompute final mix and save
                                    mixer = StemMixer()
                                    mix_res = mixer.mix(enhanced_stems, processed_sr)
                                    regenerated_mix = mix_res.mixed_audio

                                    st.session_state.regenerated_mix = regenerated_mix
                                    st.session_state.regenerated_mix_sr = processed_sr

                                st.session_state.regenerated_stem = final_stem
                                st.session_state.regeneration_summary = summary

                                st.success("✅ Regeneration applied and mix updated")

                            if discard_col.button("Discard Preview"):
                                st.session_state.pop('candidate_regen', None)
                                st.info("Preview discarded — no changes applied")

                        except Exception as e:
                            regen_progress.progress(100)
                            regen_status.text("❌ Regeneration failed!")
                            st.error(f"Regeneration failed: {e}")

            except Exception as e:
                st.error(f"Failed to create regeneration plan: {e}")

            # Helpful guidance when detector found nothing
            if not plan.regions and not plan.use_whole_stem:
                st.info("No regeneration regions identified. If you want to recreate a poor-quality stem anyway, check 'Regenerate whole stem' or use a custom region.")
        
# Provide Undo / snapshot restore if available for this stem
            if 'regeneration_snapshots' in st.session_state and st.session_state.regeneration_snapshots.get(stem_to_regenerate) is not None:
                if st.button(f"↩️ Undo last regeneration for '{stem_to_regenerate}'"):
                    prev = st.session_state.regeneration_snapshots.get(stem_to_regenerate)
                    if prev is not None and st.session_state.get('result') and getattr(st.session_state['result'], 'enhancement_result', None):
                        stems_dict = st.session_state['result'].enhancement_result.stems
                        stems_dict[stem_to_regenerate] = prev
                        st.session_state.result.enhancement_result.stems = stems_dict
                        mixer = StemMixer()
                        mix_res = mixer.mix(stems_dict, processed_sr)
                        st.session_state.regenerated_mix = mix_res.mixed_audio
                        st.success("Undo applied — previous stem restored and mix recomputed")
                    else:
                        st.error("Undo failed: no prior stem or enhancement result found")

            # Quality comparison (Original vs Regenerated) — supports regenerated stem or full mix
        if st.session_state.get('regenerated_mix') is not None or st.session_state.get('regenerated_stem') is not None:
            st.markdown("---")
            st.markdown("### 📊 Regeneration Quality Comparison")

            try:
                # Choose mix if available, otherwise compare regenerated stem
                regen_audio = None
                regen_label = "Regenerated"

                if st.session_state.get('regenerated_mix') is not None:
                    regen_audio = st.session_state.regenerated_mix
                    regen_label = "Regenerated Mix"
                else:
                    regen_audio = st.session_state.regenerated_stem
                    regen_label = "Regenerated Stem"

                regen_audio_for_analysis = prepare_analysis_audio(
                    regen_audio, st.session_state.get('regenerated_mix_sr', processed_sr), analysis_seconds
                )
                regenerated_quality = analyze_quality(regen_audio_for_analysis, processed_sr)

                # Compare metrics
                if processed_quality and regenerated_quality:
                    comp_col1, comp_col2, comp_col3 = st.columns(3)

                    with comp_col1:
                        snr_diff = regenerated_quality.snr_db - processed_quality.snr_db
                        st.metric(
                            "SNR",
                            f"{regenerated_quality.snr_db:.1f} dB",
                            f"{snr_diff:+.1f} dB"
                        )

                    with comp_col2:
                        dr_diff = regenerated_quality.dynamic_range_db - processed_quality.dynamic_range_db
                        st.metric(
                            "Dynamic Range",
                            f"{regenerated_quality.dynamic_range_db:.1f} dB",
                            f"{dr_diff:+.1f} dB"
                        )

                    with comp_col3:
                        loud_diff = regenerated_quality.loudness_lufs - processed_quality.loudness_lufs
                        st.metric(
                            "Loudness",
                            f"{regenerated_quality.loudness_lufs:.1f} LUFS",
                            f"{loud_diff:+.1f} LUFS"
                        )

                    # Artifact comparison
                    st.markdown("##### 🔍 Artifact Score Comparison")

                    artifact_cols = st.columns(2)

                    with artifact_cols[0]:
                        st.markdown("**Processed Audio**")
                        for artifact_name, score in processed_quality.artifacts.items():
                            st.progress(min(score, 1.0),
                                       text=f"{artifact_name.replace('_', ' ').title()}: {score:.2f}")

                    with artifact_cols[1]:
                        st.markdown(f"**{regen_label}**")
                        for artifact_name, score in regenerated_quality.artifacts.items():
                            st.progress(min(score, 1.0),
                                       text=f"{artifact_name.replace('_', ' ').title()}: {score:.2f}")

                    # Waveform comparison
                    st.markdown("##### 📈 Waveform Comparison")
                    wave_comp_col1, wave_comp_col2 = st.columns(2)

                    with wave_comp_col1:
                        st.markdown("**Processed**")
                        proc_fig = plot_waveform(processed_audio, processed_sr, "Processed")
                        st.plotly_chart(proc_fig, width='stretch', key="comp_proc_regen")


                    with wave_comp_col2:
                        st.markdown(f"**{regen_label}**")
                        regen_fig = plot_waveform(regen_audio, processed_sr, regen_label)
                        st.plotly_chart(regen_fig, width='stretch', key="comp_regen")

            except Exception as e:
                st.error(f"Quality comparison failed: {e}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Resonate - Live Music Reconstruction Pipeline<br>
        Built with Demucs, Pedalboard, and Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
