#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"all"} 
PROJECT_DIR=${REPO_DIR:-"/workspace/Magic-1-for-1-UI"}
MODEL_DIR=${MODEL_DIR:-"/workspace/models"}

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS"

if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    if [ ! -d "${MODEL_DIR}/HunyuanVideo" ]; then
        huggingface-cli download tencent/HunyuanVideo --local-dir "${MODEL_DIR}/HunyuanVideo"
    else
        echo "Skipping the model tencent/HunyuanVideo download because it already exists."
    fi

    if [ ! -d "${MODEL_DIR}/text_encoder" ]; then
        huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir "${MODEL_DIR}/text_encoder"
    else
        echo "Skipping the model xtuner/llava-llama-3-8b-v1_1-transformers download because it already exists."
    fi

    if [ ! -d "${MODEL_DIR}/text_encoder_2" ]; then
        huggingface-cli download openai/clip-vit-large-patch14 --local-dir "${MODEL_DIR}/text_encoder_2"
    else
        echo "Skipping the model openai/clip-vit-large-patch14 download because it already exists."
    fi

    # Create marker file
    touch "$INIT_MARKER"
    echo "Initialization complete."
else
    echo "Container already initialized. Skipping first-time setup."
fi

echo "Adding environment variables"
export PATH="$PROJECT_DIR:$PATH"

echo $PATH
echo $PYTHONPATH

cd $PROJECT_DIR

# Use conda python instead of system python
echo "Starting Gradio interface..."
uv run python interface.py &

# Use debugpy for debugging
# exec python -m debugpy --wait-for-client --listen 0.0.0.0:5678 gradio_interface.py

# echo "Starting Tensorboard interface..."
# $CONDA_DIR/bin/conda run -n pyenv tensorboard --logdir_spec=/workspace/outputs --bind_all --port 6006 &
wait