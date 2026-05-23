FROM freqtradeorg/freqtrade:stable

USER root
# Install AI/ML dependencies (no pandas-ta, we use pure-pandas indicators)
RUN pip install --no-cache-dir \
    lightgbm \
    joblib \
    scikit-learn \
    numpy

# PyTorch for LSTM models (CPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

USER ftuser
