# PROJECT MASTER PROMPT
## Deep Wildfire Progression & Intensity Forecasting Using Spatio-Temporal Deep Learning

### ROLE & DEVELOPMENT EXPECTATIONS

Act as a Senior AI Research Scientist, Geospatial Data Engineer, Deep Learning Architect, and Full-Stack Python Developer.

Your task is to design and develop a complete, research-oriented Deep Learning application titled:

**FIRECAST AI — Deep Wildfire Progression & Intensity Forecasting System**

This is a final-year academic Deep Learning project. Build it as a standalone, modular, well-documented software application using PyTorch, ConvLSTM, Rasterio, GeoPandas, NumPy, and a modern interactive frontend.

Do not create a simple demonstration or a collection of disconnected Python scripts. Build an integrated application with a complete data pipeline, model training workflow, evaluation engine, inference engine, and interactive visualization dashboard.

The system must be reproducible, testable, modular, and capable of running immediately in synthetic demonstration mode without requiring external satellite datasets or API credentials.

IMPORTANT SAFETY AND VALIDATION REQUIREMENTS:
- Position this application as an academic research and wildfire-risk-awareness system.
- Use synthetic data and historical retrospective datasets for model development and evaluation.
- Do not present experimental predictions as authoritative real-time emergency forecasts.
- Never recommend evacuation routes, firefighting tactics, or operational deployment decisions based on model output.
- Clearly communicate prediction uncertainty and direct emergency decisions to official fire and disaster-management authorities.
- Do not claim operational accuracy without independent validation.

---

# 1. PROJECT ARCHITECTURE

Create the following modular architecture:

FIRECAST_AI/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── checkpoints/
│
├── datasets/
│   ├── data_loader.py
│   ├── raster_processor.py
│   ├── temporal_builder.py
│   ├── synthetic_generator.py
│   └── normalization.py
│
├── models/
│   ├── convlstm_cell.py
│   ├── convlstm_model.py
│   ├── spatiotemporal_unet.py
│   └── model_factory.py
│
├── training/
│   ├── trainer.py
│   ├── losses.py
│   ├── metrics.py
│   └── evaluator.py
│
├── forecasting/
│   ├── inference_engine.py
│   ├── autoregressive.py
│   └── uncertainty.py
│
├── visualization/
│   ├── fire_map.py
│   ├── temporal_animation.py
│   ├── metrics_dashboard.py
│   └── geospatial_visualizer.py
│
├── utils/
│   ├── device.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── validation.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_inference.py
│
└── frontend/
    ├── components/
    └── pages/

Maintain separation between data engineering, model architecture, training, inference, and visualization.

Use object-oriented programming where it improves maintainability.

# 2. MULTI-MODAL GEOSPATIAL DATA PIPELINE

Develop a complete spatial-temporal preprocessing pipeline.

### Input data sources

Design optional adapters for:

1. NASA FIRMS — historical active-fire observations.
2. Sentinel-2 — vegetation indices and NDVI.
3. Digital Elevation Models — terrain information.
4. Historical meteorological datasets — wind, humidity, and temperature.

External data integrations must be optional. The application must work independently using synthetic data.

### Six-channel input representation

Maintain exactly six model input channels:

| Channel | Description |
|---|---|
| C1 | Historical fire perimeter and normalized fire intensity |
| C2 | NDVI vegetation index |
| C3 | Terrain risk representation derived from DEM slope and aspect |
| C4 | Wind speed |
| C5 | Wind direction, normalized consistently |
| C6 | Atmospheric dryness representation derived from relative humidity and temperature |

Document how multiple physical variables are transformed into composite channels.

Handle circular wind-direction values appropriately during preprocessing and document the information lost through channel aggregation.

### Tensor dimensions

Use the following strict tensor convention:

Input:

(Batch, Time, Channels, Height, Width)

Example:

(100, 5, 6, 128, 128)

Target:

(100, 3, 1, 128, 128)

Requirements:

- Raster alignment and reprojection.
- Consistent spatial resolution.
- Temporal synchronization.
- No-data value handling.
- Missing observation handling.
- Feature normalization.
- Train/validation/test splitting without temporal leakage.
- Geospatial metadata preservation.
- Conversion between Rasterio arrays and PyTorch tensors.

Implement a SyntheticWildfireDataset generator.

Generate:
- 100 synthetic samples.
- 5 historical observations per sample.
- 6 input channels.
- 128 × 128 spatial resolution.
- 3 consecutive future fire-map targets.

Synthetic samples must contain spatially correlated patterns and plausible temporal continuity, rather than independent random noise.

Use fixed random seeds for reproducibility.

Clearly distinguish synthetic benchmark data from physically validated wildfire simulations.

# 3. CONVLSTM DEEP LEARNING ARCHITECTURE

Implement a custom PyTorch ConvLSTM architecture.

### ConvLSTM Cell

Implement:
- Input-to-hidden convolution.
- Hidden-to-hidden convolution.
- Input, forget, output, and candidate gates.
- Hidden state initialization.
- Cell state initialization.
- Configurable kernel size.
- Configurable hidden dimensions.

### Multi-layer ConvLSTM

Build a configurable stacked architecture.

Suggested configuration:

- Input channels: 6.
- Historical time steps: 5.
- Hidden dimensions: [32, 64].
- Kernel size: 3.
- Output channels: 1.
- Prediction horizon: 3 frames.
- Spatial resolution: 128 × 128.

### Sequence-to-sequence forecasting

The architecture must:

1. Receive five historical multi-modal observations.
2. Extract spatial features through convolutional operations.
3. Learn temporal dependencies through recurrent ConvLSTM cells.
4. Generate future fire-probability maps.
5. Preserve spatial dimensions.
6. Support configurable forecasting horizons.

Apply sigmoid to produce per-pixel fire-presence probabilities.

Separate fire-presence prediction from intensity estimation. Implement an optional intensity prediction head if the dataset provides suitable intensity ground truth.

Also implement a SpatioTemporalUNet as an alternative architecture, with a common model interface.

Allow architecture selection through model_factory.py.

# 4. TRAINING ENGINE & CUSTOM LOSS

Implement a complete training pipeline.

### Combined loss function

Create a custom loss combining:

Total Loss = Focal Loss + Weighted MSE Loss

Focal Loss:
- Handles sparse fire pixels.
- Configurable alpha and gamma.
- Supports binary fire-presence targets.

MSE:
- Measures intensity prediction error.
- Must operate on a separately supervised intensity output.
- Do not calculate intensity MSE against binary perimeter labels.

Provide configurable loss weights.

### Training functionality

Implement:

- Training and validation loops.
- AdamW optimizer.
- Learning-rate scheduler.
- Configurable batch size.
- Epoch tracking.
- Gradient clipping.
- Early stopping.
- Checkpoint saving.
- Best-model restoration.
- Reproducible training.
- CPU/GPU device detection.

Save:
- Best model checkpoint.
- Final model checkpoint.
- Training configuration.
- Normalization parameters.
- Training history.

### Evaluation metrics

Track:

1. Intersection over Union (IoU).
2. Dice coefficient.
3. Precision.
4. Recall.
5. F1 score.
6. Binary cross-entropy.
7. Intensity MAE/MSE where applicable.

Evaluate fire presence using a configurable probability threshold.

Display training and validation curves.

Ensure all metrics are calculated correctly across spatial and temporal dimensions.

# 5. FUTURE PROGRESSION FORECASTING ENGINE

Implement a modular autoregressive inference engine.

The engine should accept a historical multi-modal tensor and produce future prediction frames.

### Forecasting workflow

1. Load the trained model checkpoint.
2. Validate incoming tensor dimensions.
3. Apply the same normalization used during training.
4. Pass historical observations through ConvLSTM.
5. Generate future fire-presence probability maps.
6. Preserve temporal ordering.
7. Return prediction probabilities and optional intensity estimates.
8. Provide uncertainty estimates where supported.

Support configurable prediction horizons corresponding to:

- 12 hours.
- 24 hours.
- 48 hours.

IMPORTANT: These horizons must be connected to actual observation intervals and model training targets. Do not simply relabel three consecutive frames as 12, 24, and 48 hours.

Implement temporal metadata and explicit frame-to-hour conversion.

Autoregressive prediction must correctly handle the mismatch between six-channel inputs and single-channel fire predictions.

Never feed a one-channel prediction directly into a six-channel model.

Use a dedicated feedback adapter that updates the fire-observation channel while retaining or appropriately updating the other environmental channels. Clearly identify assumptions about future meteorological and vegetation conditions.

When future environmental observations are unavailable, label them as assumed or held constant rather than pretending they are known.

# 6. INTERACTIVE GEOSPATIAL VISUALIZATION

Build an interactive dashboard for exploring historical observations, model predictions, and evaluation results.

Use Streamlit as the primary frontend.

### Dashboard sections

**A. Overview Dashboard**

Display:
- Project title and description.
- Model architecture.
- Dataset size.
- Training status.
- Latest evaluation metrics.
- Device information.

**B. Historical Fire Analysis**

Display:
- Historical fire observation maps.
- Vegetation index layer.
- Terrain layer.
- Meteorological feature summaries.
- Temporal observation sequence.

**C. Future Forecast Visualization**

Create synchronized panels:

Panel 1: Historical Fire Observations

Panel 2: Predicted 12-Hour Map

Panel 3: Predicted 24-Hour Map

Panel 4: Predicted 48-Hour Map

Use consistent spatial extents, color scales, and probability thresholds.

**D. Temporal Animation**

Create an animation that displays historical observations followed by model-generated future frames.

Include:
- Play/pause.
- Frame selection.
- Timestamp.
- Probability legend.
- Historical/predicted frame distinction.

**E. Model Performance**

Display:
- Training loss.
- Validation loss.
- IoU.
- Dice coefficient.
- Precision and recall.
- Prediction-versus-ground-truth comparisons.

Use Plotly for interactive charts.

Use Folium or an appropriate geospatial mapping library for map visualization.

Preserve geographic coordinates whenever real geospatial datasets are loaded.

For synthetic datasets, clearly label the coordinate system as illustrative.

# 7. UNCERTAINTY & MODEL RELIABILITY

Add an uncertainty-analysis module.

Implement a research-oriented Monte Carlo Dropout or ensemble-based uncertainty estimation method.

Display:
- Mean predicted probability.
- Prediction variability.
- High-uncertainty regions.
- Confidence limitations.

Do not interpret model confidence as a guarantee of forecast correctness.

Include an explicit warning that the model is not validated for operational wildfire response.

# 8. DATASET AND MODEL MANAGEMENT

Create an easy-to-use dataset management interface.

Support:

- Synthetic dataset generation.
- Loading historical raster datasets.
- Dataset validation.
- Raster metadata inspection.
- Preprocessing preview.
- Train/validation/test splitting.
- Model checkpoint selection.

Provide optional adapters for public data sources.

Do not require API credentials for the initial demonstration.

Document source attribution, licensing, temporal resolution, and spatial resolution for every supported dataset.

# 9. SOFTWARE ENGINEERING REQUIREMENTS

Use:

- Python 3.11 or compatible environment.
- PyTorch.
- Rasterio.
- GeoPandas.
- NumPy.
- Pandas.
- Scikit-learn.
- Matplotlib.
- Plotly.
- Streamlit.
- Shapely.
- Pytest.

Requirements:

- Type hints.
- Docstrings.
- Clear exception handling.
- Centralized configuration.
- Logging.
- Modular classes.
- Reusable functions.
- Reproducible experiments.
- No hardcoded machine-specific paths.
- No unnecessary dependencies.
- No duplicate implementations.

Create requirements.txt and a detailed README.md.

# 10. TESTING & VALIDATION

Create automated tests for:

1. Synthetic data generation.
2. Input tensor dimensions.
3. Target tensor dimensions.
4. ConvLSTM forward pass.
5. Multi-layer model output.
6. Loss computation.
7. Metric calculation.
8. Training checkpoint saving.
9. Checkpoint restoration.
10. Autoregressive inference.
11. Raster loading and preprocessing.
12. Geospatial metadata preservation.

Test the entire workflow using synthetic data before integrating external datasets.

The application must successfully execute:

Generate Dataset → Preprocess → Train → Evaluate → Save Checkpoint → Load Model → Forecast → Visualize.

# 11. USER INTERFACE & DESIGN

Create a professional scientific dashboard.

Design direction:

- Dark scientific interface.
- Deep charcoal background.
- Subtle orange and amber accent colors.
- Clean typography.
- Modern metric cards.
- Interactive charts.
- Large geospatial visualization panels.
- Clear legends.
- Consistent spacing.
- Responsive layout.

Use wildfire-inspired colors only for map overlays and visual semantics.

Avoid unnecessary animations and decorative components.

The dashboard should look like a professional geospatial research platform rather than a generic machine-learning demo.

# 12. EXECUTION WORKFLOW

The application must support the following commands:

Install dependencies:

pip install -r requirements.txt

Generate synthetic dataset:

python -m datasets.synthetic_generator

Train model:

python -m training.trainer

Evaluate model:

python -m training.evaluator

Launch dashboard:

streamlit run app.py

Run tests:

pytest tests/ -v

Provide a configuration option to run the entire workflow using synthetic data.

# 13. DEVELOPMENT STRATEGY

Do not attempt to generate the entire application in one unverified code-generation step.

Follow these development phases:

PHASE 1 — Project scaffolding and dependency configuration.

PHASE 2 — Synthetic dataset generator and geospatial preprocessing.

PHASE 3 — ConvLSTM cell and sequence-to-sequence model.

PHASE 4 — Custom loss, training loop, and evaluation metrics.

PHASE 5 — Autoregressive inference and uncertainty estimation.

PHASE 6 — Interactive geospatial dashboard.

PHASE 7 — Automated testing and integration.

PHASE 8 — Documentation and final execution validation.

At the end of each phase:

- Inspect generated files.
- Run relevant tests.
- Fix errors before proceeding.
- Verify tensor shapes.
- Check module imports.
- Avoid replacing working modules unnecessarily.

Do not leave incomplete functions, pseudocode, or unexplained TODO statements.

If a feature cannot be implemented reliably, document the limitation rather than pretending it works.

# 14. FINAL ACCEPTANCE CRITERIA

The completed application must demonstrate:

1. Successful generation of 100 synthetic multi-modal time-series samples.
2. Correct input shape: (100, 5, 6, 128, 128).
3. Correct future target shape: (100, 3, 1, 128, 128).
4. Successful ConvLSTM forward and backward passes.
5. Successful model training.
6. Working Focal Loss and intensity MSE supervision.
7. Correct IoU and Dice calculations.
8. Saved and reloadable model checkpoints.
9. Working multi-step autoregressive inference.
10. Future forecast visualization.
11. Interactive dashboard.
12. Successful automated tests.
13. Reproducible synthetic demonstration.
14. Clear distinction between experimental predictions and validated wildfire forecasts.

### FINAL INSTRUCTION

Build FIRECAST AI as a complete, integrated academic research application.

Prioritize scientific correctness, tensor dimensionality, spatial-temporal consistency, modular architecture, reproducibility, and an impressive interactive visualization experience.

Start with PHASE 1. Inspect the project environment, establish the directory structure, and implement each phase sequentially.

Do not stop after generating architecture diagrams or explaining the approach. Create the actual working files, execute the application, test the implementation, and fix errors before declaring the project complete.