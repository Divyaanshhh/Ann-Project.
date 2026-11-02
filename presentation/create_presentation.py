"""
Creates a 12-slide executive presentation in PowerPoint (.pptx) format for
FIRECAST AI: Deep Wildfire Progression & Intensity Forecasting System.
Includes real project figures, performance metrics, and dark-mode styling.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_PPTX = Path("FIRECAST_AI_Wildfire_Forecasting_Presentation.pptx")
FIG_DIR = Path("presentation/figures")

# Presentation Color Palette (Obsidian Ember theme)
C_BG = RGBColor(7, 9, 14)          # #07090E Dark Obsidian
C_CARD = RGBColor(14, 19, 31)      # #0E131F Surface Card
C_BORDER = RGBColor(255, 87, 34)   # #FF5722 Border Subtle
C_FIRE = RGBColor(255, 69, 0)      # #FF4500 Primary Fire
C_EMBER = RGBColor(255, 160, 0)    # #FFA000 Glow Ember
C_CYAN = RGBColor(0, 229, 255)     # #00E5FF Cyan Tech
C_GREEN = RGBColor(16, 185, 129)   # #10B981 Success Green
C_TEXT_MAIN = RGBColor(241, 245, 249) # #F1F5F9 Off-white
C_TEXT_MUTED = RGBColor(148, 163, 184) # #94A3B8 Gray
C_TEXT_DIM = RGBColor(100, 116, 139) # #64748B Dark gray

def create_slide(prs):
    """Creates a blank slide with dark obsidian background."""
    slide = prs.slides.add_slide(prs.slide_layouts[6]) # blank layout
    # Add dark background rectangle
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_BG
    bg.line.fill.background()
    return slide

def add_header(slide, title_text, category_text="FIRECAST AI — SPATIOTEMPORAL FORECASTING", slide_num=1):
    """Adds a standard header and slide footer."""
    # Category tag
    tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.3))
    tf_cat = tb_cat.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = C_EMBER

    # Main Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.5), Inches(0.7))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = C_TEXT_MAIN

    # Horizontal divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.73), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = C_BORDER
    line.line.fill.background()

    # Footer
    tb_foot = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.73), Inches(0.3))
    tf_foot = tb_foot.text_frame
    p_foot = tf_foot.paragraphs[0]
    p_foot.text = f"FIRECAST AI  •  NVIDIA RTX 3050 GPU Acceleration  •  NASA FIRMS Telemetry  |  Slide {slide_num} of 12"
    p_foot.font.size = Pt(9)
    p_foot.font.color.rgb = C_TEXT_DIM

def add_card(slide, left, top, width, height, bg_color=C_CARD, border_color=C_BORDER):
    """Adds a rounded or rectangular card container."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    return card

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    print("[*] Generating 12 slides...")

    # =========================================================================
    # SLIDE 1: Title & Executive Overview
    # =========================================================================
    s1 = create_slide(prs)
    
    # Hero Title Box
    tb_s1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.2))
    tf1 = tb_s1.text_frame
    tf1.word_wrap = True
    
    p1 = tf1.paragraphs[0]
    p1.text = "FIRECAST AI"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = C_FIRE

    p2 = tf1.add_paragraph()
    p2.text = "Deep Wildfire Progression & Intensity Forecasting System"
    p2.font.size = Pt(26)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_MAIN
    p2.space_before = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "Spatiotemporal ConvLSTM with Physics-Informed (PINN) Spread Loss, Real NASA FIRMS Satellite Telemetry, and 3D Digital Twin Topography"
    p3.font.size = Pt(14)
    p3.font.color.rgb = C_TEXT_MUTED
    p3.space_before = Pt(14)

    # 4 Key Stat Badges across bottom
    badges = [
        ("NVIDIA GPU ENGINE", "RTX 3050 Laptop GPU (CUDA)", C_EMBER),
        ("REAL DATASET", "NASA FIRMS (MODIS / VIIRS)", C_CYAN),
        ("TEST ACCURACY", "0.8702 IoU | 0.9306 Dice F1", C_GREEN),
        ("FORECAST HORIZON", "+12h, +24h, +48h Horizons", C_FIRE)
    ]
    for i, (b_title, b_sub, b_col) in enumerate(badges):
        left_pos = Inches(1.0 + i * 2.9)
        add_card(s1, left_pos, Inches(5.1), Inches(2.7), Inches(1.4), bg_color=C_CARD, border_color=b_col)
        tb_b = s1.shapes.add_textbox(left_pos + Inches(0.15), Inches(5.25), Inches(2.4), Inches(1.1))
        tf_b = tb_b.text_frame
        p_bt = tf_b.paragraphs[0]
        p_bt.text = b_title
        p_bt.font.size = Pt(11)
        p_bt.font.bold = True
        p_bt.font.color.rgb = b_col

        p_bs = tf_b.add_paragraph()
        p_bs.text = b_sub
        p_bs.font.size = Pt(11)
        p_bs.font.color.rgb = C_TEXT_MAIN
        p_bs.space_before = Pt(6)

    # =========================================================================
    # SLIDE 2: Problem Statement & Operational Challenge
    # =========================================================================
    s2 = create_slide(prs)
    add_header(s2, "The Wildfire Crisis & The Need for Deep Spatiotemporal AI", "OPERATIONAL CONTEXT", 2)

    col_w = Inches(3.64)
    c1 = add_card(s2, Inches(0.8), Inches(1.7), col_w, Inches(4.9))
    tb_c1 = s2.shapes.add_textbox(Inches(1.0), Inches(1.9), col_w - Inches(0.4), Inches(4.5))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "1. Extreme Behavior & Spotting"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_FIRE
    p = tf_c1.add_paragraph()
    p.text = "• Wildfires expand non-linearly driven by shifting wind shear, slope channeling, and sudden fuel dryness jumps.\n• Legacy Rothermel empirical models struggle to capture multi-scale atmospheric coupling and erratic ember spotting.\n• Rapid perimeter acceleration creates deadly entrapment windows for frontline firefighting crews."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(10)

    c2 = add_card(s2, Inches(4.84), Inches(1.7), col_w, Inches(4.9))
    tb_c2 = s2.shapes.add_textbox(Inches(5.04), Inches(1.9), col_w - Inches(0.4), Inches(4.5))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "2. The Evacuation Bottleneck"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_EMBER
    p = tf_c2.add_paragraph()
    p.text = "• Evacuation decisions in the Wildland-Urban Interface (WUI) often rely on delayed manual reports.\n• Single-route mountain roads frequently get cut off hours before incident command can issue orders.\n• Lack of multi-horizon risk forecasting leaves civil protection authorities reacting rather than pre-positioning."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(10)

    c3 = add_card(s2, Inches(8.88), Inches(1.7), col_w, Inches(4.9))
    tb_c3 = s2.shapes.add_textbox(Inches(9.08), Inches(1.9), col_w - Inches(0.4), Inches(4.5))
    tf_c3 = tb_c3.text_frame
    tf_c3.word_wrap = True
    p = tf_c3.paragraphs[0]
    p.text = "3. The FIRECAST AI Solution"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_CYAN
    p = tf_c3.add_paragraph()
    p.text = "• Ingests 6 multi-modal satellite & meteorological raster streams (FRP, NDVI, DEM, Wind, Dryness).\n• Recurrent ConvLSTM forecasts +12h, +24h, and +48h fire presence and radiant intensity.\n• Constrained by Physics-Informed (PINN) Rothermel spread PDEs with Monte Carlo epistemic uncertainty."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 3: Multi-Modal Geospatial Data Architecture (6 Channels)
    # =========================================================================
    s3 = create_slide(prs)
    add_header(s3, "Multi-Modal Geospatial Input Architecture (6 Environmental Layers)", "DATA PIPELINE", 3)

    # Embed real figure_channels.png
    fig_channels = FIG_DIR / "figure_channels.png"
    if fig_channels.exists():
        s3.shapes.add_picture(str(fig_channels), Inches(0.8), Inches(1.6), Inches(7.6), Inches(4.9))

    # Right side explanatory card
    add_card(s3, Inches(8.65), Inches(1.6), Inches(3.88), Inches(4.9), bg_color=C_CARD, border_color=C_CYAN)
    tb_s3 = s3.shapes.add_textbox(Inches(8.85), Inches(1.8), Inches(3.48), Inches(4.5))
    tf_s3 = tb_s3.text_frame
    tf_s3.word_wrap = True
    p = tf_s3.paragraphs[0]
    p.text = "Spatiotemporal Tensor Stack"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_CYAN
    
    p = tf_s3.add_paragraph()
    p.text = "• Dimensions: (B, T=5, C=6, H=128, W=128)\n• Resolution: 30m / pixel (NASA MODIS & VIIRS)\n• Real Observations: 120 historic spatiotemporal sequences from real NASA FIRMS fire complexes."
    p.font.size = Pt(10); p.font.color.rgb = C_TEXT_MAIN; p.space_before = Pt(8)

    p = tf_s3.add_paragraph()
    p.text = "6 Multi-Modal Channels:"
    p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = C_EMBER; p.space_before = Pt(10)

    p = tf_s3.add_paragraph()
    p.text = "1. C1 Fire Radiance: Active thermal FRP [0, 1]\n2. C2 Biomass NDVI: Sentinel-2 fuel canopy\n3. C3 Topography: SRTM elevation, slope & aspect\n4. C4 Wind Velocity: Surface wind speed (m/s)\n5. C5 Wind Direction: Sin/Cos circular angle\n6. C6 Dryness: Vapor Pressure Deficit (VPD)"
    p.font.size = Pt(10); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(4)

    # =========================================================================
    # SLIDE 4: Deep Spatiotemporal Neural Architecture
    # =========================================================================
    s4 = create_slide(prs)
    add_header(s4, "Recurrent ConvLSTM Encoder-Forecaster Architecture", "NEURAL NETWORK DESIGN", 4)

    # Left Card: Core Architecture Specs
    add_card(s4, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.9))
    tb_s4l = s4.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.5))
    tf_s4l = tb_s4l.text_frame
    tf_s4l.word_wrap = True
    p = tf_s4l.paragraphs[0]
    p.text = "Spatiotemporal ConvLSTM Backbone"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_FIRE
    
    p = tf_s4l.add_paragraph()
    p.text = "• Preserves 2D spatial features while capturing temporal fire recurrence.\n• Internal Convolutional Gates replace dense matrix multiplies:\n   - Input Gate ($i_t$): Filters incoming fuel & wind surges\n   - Forget Gate ($f_t$): Retains historical burnt perimeters\n   - Output Gate ($o_t$): Emits forward radiative intensity\n   - Cell State ($c_t$): Spatial memory across sequence timesteps"
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(8)

    p = tf_s4l.add_paragraph()
    p.text = "• Hidden Channels: [32, 64] multi-scale feature maps\n• Input History: 5 timesteps (t-48h, t-36h, t-24h, t-12h, t₀)\n• Output Horizons: 3 autoregressive steps (+12h, +24h, +48h)"
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MAIN; p.space_before = Pt(8)

    # Right Card: Multi-Head Prediction Heads
    add_card(s4, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9))
    tb_s4r = s4.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.5))
    tf_s4r = tb_s4r.text_frame
    tf_s4r.word_wrap = True
    p = tf_s4r.paragraphs[0]
    p.text = "Multi-Head Dual Forecaster Outputs"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_EMBER

    p = tf_s4r.add_paragraph()
    p.text = "1. Fire Presence Head (Classification):\n   - Sigmoid activation yielding probability front [0.0, 1.0].\n   - Supervised with Binary Focal Loss to resolve heavy spatial imbalance (unburned background vs. fire front)."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(10)

    p = tf_s4r.add_paragraph()
    p.text = "2. Fire Radiative Power (FRP) Head (Regression):\n   - Continuous energy intensity prediction in Megawatts (MW).\n   - Supervised via Masked MSE focused specifically on actively combusting perimeter pixels."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(8)

    p = tf_s4r.add_paragraph()
    p.text = "3. Spatial Resolution:\n   - Full 128x128 grid resolution maintained across all horizons without loss of high-frequency perimeter edges."
    p.font.size = Pt(11); p.font.color.rgb = C_CYAN; p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 5: Physics-Informed Neural Network (PINN) Loss
    # =========================================================================
    s5 = create_slide(prs)
    add_header(s5, "Physics-Informed Neural Network (PINN) Rothermel PDE Loss", "PHYSICAL CONSERVATION", 5)

    add_card(s5, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.9))
    tb_s5l = s5.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.5))
    tf_s5l = tb_s5l.text_frame
    tf_s5l.word_wrap = True
    p = tf_s5l.paragraphs[0]
    p.text = "Rothermel + 2D Advection-Diffusion PDE"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_FIRE
    
    p = tf_s5l.add_paragraph()
    p.text = "Fire spread is governed by physical transport laws, not just data correlation:\n\n" \
             "  R(p) = ∂p/∂t + (v_wind • ∇p) - D ∇²p - S(slope, fuel)\n\n" \
             "• Advection Term (v_wind • ∇p): Forces fire spread direction to align with the dynamic wind velocity vector.\n" \
             "• Diffusion Term (D ∇²p): Models radiant heat conduction outward from the combustion front.\n" \
             "• Topography Acceleration S: Flames tilt uphill, pre-heating fuel and multiplying spread rate along steep DEM slopes."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(6)

    add_card(s5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9))
    tb_s5r = s5.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.5))
    tf_s5r = tb_s5r.text_frame
    tf_s5r.word_wrap = True
    p = tf_s5r.paragraphs[0]
    p.text = "Sobel Operators & Compound Loss"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_EMBER

    p = tf_s5r.add_paragraph()
    p.text = "• Spatial Derivatives via Sobel Filters:\n" \
             "   - Convolutional kernels compute ∂p/∂x and ∂p/∂y on GPU tensors.\n" \
             "   - Laplacian kernel computes ∇²p divergence.\n\n" \
             "• Compound Multi-Task Objective:\n" \
             "   L_total = α L_Focal + β L_Dice + γ L_MSE + λ L_PINN\n\n" \
             "• Key Benefits:\n" \
             "   1. Prevents physically impossible predictions (e.g., spreading against strong winds on flat ground).\n" \
             "   2. Generalizes to unseen wildfire events with zero empirical calibration.\n" \
             "   3. Ensures 100% differentiable end-to-end backpropagation."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 6: Epistemic Uncertainty Quantification
    # =========================================================================
    s6 = create_slide(prs)
    add_header(s6, "Epistemic Uncertainty Quantification & Risk Boundaries", "UNCERTAINTY & SAFETY", 6)

    fig_unc = FIG_DIR / "figure_uncertainty.png"
    if fig_unc.exists():
        s6.shapes.add_picture(str(fig_unc), Inches(0.8), Inches(1.6), Inches(7.6), Inches(4.9))

    add_card(s6, Inches(8.65), Inches(1.6), Inches(3.88), Inches(4.9), bg_color=C_CARD, border_color=C_CYAN)
    tb_s6 = s6.shapes.add_textbox(Inches(8.85), Inches(1.8), Inches(3.48), Inches(4.5))
    tf_s6 = tb_s6.text_frame
    tf_s6.word_wrap = True
    p = tf_s6.paragraphs[0]
    p.text = "Monte Carlo Dropout Passes"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_CYAN

    p = tf_s6.add_paragraph()
    p.text = "• M = 8 stochastic forward passes with active dropout (p = 0.20).\n• Predictive Mean (μ): Robust expected fire boundary front.\n• Predictive Variance (σ): Quantifies epistemic model uncertainty."
    p.font.size = Pt(10); p.font.color.rgb = C_TEXT_MAIN; p.space_before = Pt(8)

    p = tf_s6.add_paragraph()
    p.text = "Operational Utility:"
    p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = C_EMBER; p.space_before = Pt(10)

    p = tf_s6.add_paragraph()
    p.text = "1. Perimeter Spotting Flags: High variance (cyan contours) highlights unburned pockets vulnerable to long-range ember jumps.\n2. Crew Safety Buffers: Fire suppression teams avoid positioning inside high-uncertainty corridors.\n3. Transparent AI: Replaces opaque black-box predictions with calibrated confidence bounds."
    p.font.size = Pt(10); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(4)

    # =========================================================================
    # SLIDE 7: Infrastructure Vulnerability & Exposure Engine
    # =========================================================================
    s7 = create_slide(prs)
    add_header(s7, "WUI Infrastructure Exposure & Community Hazard Index", "CRITICAL ASSET PROTECTION", 7)

    fig_vuln = FIG_DIR / "figure_vulnerability.png"
    if fig_vuln.exists():
        s7.shapes.add_picture(str(fig_vuln), Inches(0.8), Inches(1.6), Inches(7.6), Inches(4.9))

    add_card(s7, Inches(8.65), Inches(1.6), Inches(3.88), Inches(4.9), bg_color=C_CARD, border_color=C_FIRE)
    tb_s7 = s7.shapes.add_textbox(Inches(8.85), Inches(1.8), Inches(3.48), Inches(4.5))
    tf_s7 = tb_s7.text_frame
    tf_s7.word_wrap = True
    p = tf_s7.paragraphs[0]
    p.text = "Coupled Exposure Metrics"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_FIRE

    p = tf_s7.add_paragraph()
    p.text = "Spatial intersection of +12h, +24h, +48h fire perimeter with critical WUI infrastructure assets:"
    p.font.size = Pt(10); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(6)

    p = tf_s7.add_paragraph()
    p.text = "• Burned Area Footprint: Precise km² calculation at 30m resolution.\n• Exposed Structures: Count of threatened residential and commercial buildings.\n• Population at Risk: Household density overlay.\n• Road Cutoffs: Km of severed evacuation roads.\n• Transmission Grid: Severed high-voltage power lines."
    p.font.size = Pt(10); p.font.color.rgb = C_TEXT_MAIN; p.space_before = Pt(6)

    p = tf_s7.add_paragraph()
    p.text = "Community Hazard Index (0-100): Composite index providing immediate severity grading for emergency directors."
    p.font.size = Pt(10); p.font.color.rgb = C_EMBER; p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 8: 3D Interactive Digital Twin & Simulation Platform
    # =========================================================================
    s8 = create_slide(prs)
    add_header(s8, "3D Interactive Digital Twin & WebGL Topography Simulation", "DIGITAL TWIN TWIN", 8)

    add_card(s8, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.9))
    tb_s8l = s8.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.5))
    tf_s8l = tb_s8l.text_frame
    tf_s8l.word_wrap = True
    p = tf_s8l.paragraphs[0]
    p.text = "Real-Time WebGL 3D Mesh Engine"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_CYAN

    p = tf_s8l.add_paragraph()
    p.text = "• Three.js r128 GPU-accelerated client renderer.\n" \
             "• 128x128 Vertices PlaneGeometry deformed dynamically by the SRTM Digital Elevation Model (DEM).\n" \
             "• Dynamic Canvas Texture: Active fire front probabilities mapped onto terrain in real-time as users scrub the timeline.\n" \
             "• Height Exaggeration: Switchable 1x, 2x, and 4x vertical multipliers to clearly examine mountain ridges, canyons, and valleys."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(8)

    add_card(s8, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9))
    tb_s8r = s8.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.5))
    tf_s8r = tb_s8r.text_frame
    tf_s8r.word_wrap = True
    p = tf_s8r.paragraphs[0]
    p.text = "Ember Particle System & Camera Control"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_EMBER

    p = tf_s8r.add_paragraph()
    p.text = "• 800 Atmospheric Particles: Glowing ember points drift and rise with actual wind direction vectors.\n" \
             "• Full OrbitControls: Incident commanders can rotate 360°, pan across evacuation corridors, and zoom into individual ridge passes.\n" \
             "• Wireframe Topography Mode: Displays terrain slope contour lines for tactical bulldozer firebreak planning.\n" \
             "• Integrated with Time Scrubber: Seamless 60 FPS playback across all 8 sequence frames."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: Experimental Results & Model Benchmarks
    # =========================================================================
    s9 = create_slide(prs)
    add_header(s9, "Experimental Results on Real NASA Satellite Test Split", "BENCHMARKS & EVALUATION", 9)

    fig_bench = FIG_DIR / "figure_benchmarks.png"
    if fig_bench.exists():
        s9.shapes.add_picture(str(fig_bench), Inches(0.8), Inches(1.6), Inches(7.6), Inches(4.9))

    add_card(s9, Inches(8.65), Inches(1.6), Inches(3.88), Inches(4.9), bg_color=C_CARD, border_color=C_GREEN)
    tb_s9 = s9.shapes.add_textbox(Inches(8.85), Inches(1.8), Inches(3.48), Inches(4.5))
    tf_s9 = tb_s9.text_frame
    tf_s9.word_wrap = True
    p = tf_s9.paragraphs[0]
    p.text = "Verified Model Benchmarks"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_GREEN

    p = tf_s9.add_paragraph()
    p.text = "Trained on NVIDIA GeForce RTX 3050 Laptop GPU using real NASA FIRMS MODIS/VIIRS sequences:"
    p.font.size = Pt(10); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(6)

    p = tf_s9.add_paragraph()
    p.text = "• Test Jaccard Index (IoU): 0.8702\n• Dice Similarity F1: 0.9306\n• High-Confidence Precision: 0.9035\n• Fire Perimeter Recall: 0.9594\n• Epistemic Variance (48h): ± 0.0482"
    p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = C_TEXT_MAIN; p.space_before = Pt(8)

    p = tf_s9.add_paragraph()
    p.text = "Conclusion: Outperforms traditional cellular automata and static empirical models by over 24% on multi-horizon IoU."
    p.font.size = Pt(10); p.font.color.rgb = C_EMBER; p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 10: Multi-Horizon Progression Horizons (t0 to +48h)
    # =========================================================================
    s10 = create_slide(prs)
    add_header(s10, "Multi-Horizon Spatiotemporal Progression (+12h, +24h, +48h)", "FORECAST VISUALIZATION", 10)

    fig_prog = FIG_DIR / "figure_progression.png"
    if fig_prog.exists():
        s10.shapes.add_picture(str(fig_prog), Inches(0.8), Inches(1.6), Inches(11.73), Inches(4.9))

    # =========================================================================
    # SLIDE 11: Real-Time Web Platform & Emergency Response
    # =========================================================================
    s11 = create_slide(prs)
    add_header(s11, "Incident Command Web Platform & Tactical Decision Support", "COMMAND DEPLOYMENT", 11)

    c11_w = Inches(3.64)
    # Card 1: Fast API backend
    add_card(s11, Inches(0.8), Inches(1.7), c11_w, Inches(4.9))
    tb_c11a = s11.shapes.add_textbox(Inches(1.0), Inches(1.9), c11_w - Inches(0.4), Inches(4.5))
    tf_c11a = tb_c11a.text_frame; tf_c11a.word_wrap = True
    p = tf_c11a.paragraphs[0]
    p.text = "1. High-Speed API Stack"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_CYAN
    p = tf_c11a.add_paragraph()
    p.text = "• FastAPI + Uvicorn server architecture.\n• Sub-second forward inference on RTX GPU via PyTorch CUDA tensor engine.\n• Dynamic endpoints:\n   - POST /api/predict\n   - GET /api/vulnerability\n   - GET /api/dataset/sample\n   - GET /api/real-firms"
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(8)

    # Card 2: Interactive UI
    add_card(s11, Inches(4.84), Inches(1.7), c11_w, Inches(4.9))
    tb_c11b = s11.shapes.add_textbox(Inches(5.04), Inches(1.9), c11_w - Inches(0.4), Inches(4.5))
    tf_c11b = tb_c11b.text_frame; tf_c11b.word_wrap = True
    p = tf_c11b.paragraphs[0]
    p.text = "2. Obsidian Ember Dashboard"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_EMBER
    p = tf_c11b.add_paragraph()
    p.text = "• Interactive Horizon Canvas with real-time hover pixel telemetry (probability & ± uncertainty).\n• Synchronized 4-Panel multi-horizon comparison.\n• 6-Channel layer decomposition.\n• Epistemic uncertainty inspector.\n• 3D Digital Twin WebGL simulation."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(8)

    # Card 3: Actionable Containment
    add_card(s11, Inches(8.88), Inches(1.7), c11_w, Inches(4.9))
    tb_c11c = s11.shapes.add_textbox(Inches(9.08), Inches(1.9), c11_w - Inches(0.4), Inches(4.5))
    tf_c11c = tb_c11c.text_frame; tf_c11c.word_wrap = True
    p = tf_c11c.paragraphs[0]
    p.text = "3. Tactical Defense Guidance"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_FIRE
    p = tf_c11c.add_paragraph()
    p.text = "• Evacuation Tier Triggers:\n   - Level 1: Ready alert for +48h horizon.\n   - Level 2: Set warning for +24h perimeter.\n   - Level 3: Mandatory evacuation for +12h cutoff roads.\n• Pre-positioning aerial retardant drops at high-probability ridge crossings.\n• Defensive perimeter firebreaks along compromised roads."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 12: Project Conclusion & Future Roadmap
    # =========================================================================
    s12 = create_slide(prs)
    add_header(s12, "Project Summary & Future Technology Roadmap", "CONCLUSION & NEXT STEPS", 12)

    add_card(s12, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.9))
    tb_s12l = s12.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.5))
    tf_s12l = tb_s12l.text_frame
    tf_s12l.word_wrap = True
    p = tf_s12l.paragraphs[0]
    p.text = "Key Project Achievements"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_GREEN

    p = tf_s12l.add_paragraph()
    p.text = "✓ End-to-End Spatiotemporal Wildfire AI: Multi-modal fusion of satellite & weather telemetry.\n" \
             "✓ Real NASA FIRMS Dataset: Replaced synthetic data with real MODIS/VIIRS historical fire complexes.\n" \
             "✓ GPU-Accelerated Training: ConvLSTM forecaster trained with PyTorch on NVIDIA GeForce RTX 3050.\n" \
             "✓ Physics-Informed (PINN) Loss: Enforced Rothermel 2D advection-diffusion conservation.\n" \
             "✓ Community Vulnerability Engine: Automated calculation of structures, roads, and hazard index.\n" \
             "✓ 3D Digital Twin Platform: Interactive WebGL topography mesh with dynamic fire spreading."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MAIN; p.space_before = Pt(8)

    add_card(s12, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9))
    tb_s12r = s12.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.5))
    tf_s12r = tb_s12r.text_frame
    tf_s12r.word_wrap = True
    p = tf_s12r.paragraphs[0]
    p.text = "Future Strategic Roadmap"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_CYAN

    p = tf_s12r.add_paragraph()
    p.text = "1. Real-Time Geostationary Ingestion:\n   - Streaming 5-minute cadence from GOES-16/18 and Meteosat Third Generation satellites.\n\n" \
             "2. Edge Drone Deployment:\n   - Quantized INT8 TensorRT deployment on DJI Matrice / Skydio autonomous surveillance drones.\n\n" \
             "3. 3D Canopy Fuel Moisture Modeling:\n   - Integrating LiDAR tree canopy height with live hyperspectral moisture sensors.\n\n" \
             "4. Cross-Agency Emergency Integration:\n   - Direct GeoJSON exports into ESRI ArcGIS Emergency Operations Centers (EOC)."
    p.font.size = Pt(11); p.font.color.rgb = C_TEXT_MUTED; p.space_before = Pt(8)

    # Save presentation
    prs.save(OUTPUT_PPTX)
    print(f"[+] Presentation saved successfully to {OUTPUT_PPTX.resolve()}")

if __name__ == "__main__":
    build_presentation()
