set -ex

### A5000
Follow install from the original document.


### A6000
# Create completely fresh environment
conda create -n prob python=3.10 -y
conda activate prob

# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt

# Build MSDA
cd models/ops
rm -rf build/
sh ./make.sh

# Test MSDA
python test.py

echo "=== Setup complete! ==="