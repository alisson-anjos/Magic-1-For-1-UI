export USE_FLASH_ATTENTION3=1
export CUDA_VISIBLE_DEVICES=0
python test_ti2v.py --config configs/test/text_to_video/4_step_ti2v.yaml --quantization True