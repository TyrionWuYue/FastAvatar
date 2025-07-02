# install torch 2.5.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu121

# install dependencies
pip install -r requirements.txt

# === If you fail to install some modules due to network connection, you can also try the following: ===
git clone https://github.com/facebookresearch/pytorch3d.git
pip install ./pytorch3d
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
git clone https://github.com/camenduru/simple-knn.git
pip install ./simple-knn
git clone https://github.com/NVlabs/nvdiffrast.git
pip install ./nvdiffrast

cd external/landmark_detection/FaceBoxesV2/utils/
sh make.sh
cd ../../../../