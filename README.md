# Glaucoma Classification using Vision Models and Contrastive Learning

This project focuses on using vision models and contrastive learning techniques to classify Glaucoma in 3D Macular and Optic OCT scans. The main objective is to develop a deep learning model that can accurately classify OCT (Optical Coherence Tomography) scans as either showing signs of Glaucoma or being healthy.

## Getting Started

To run the training script and start training your model, follow the instructions below:

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/anthony-c-cuturrufo/Glaucoma.git
   ```

2. Navigate to the project directory:

   ```
   cd Glaucoma
   ```

3. For dependency management, we use a Conda environment. Please email me to request the environment YAML file. Once received, create and activate the Conda environment:

   ```
   conda env create -f environment.yml
   conda activate glaucoma_env
   ```

4. Run the main training script:

   ```
   python train.py --model_name ResNext50 --cuda cuda:2 --batch_size 4 --dropout 0.2 --contrastive_mode None --augment True
   ```

   Or check out `./tune_macular_dist.sh`

## Script Arguments

- `--model_name`: Name of the vision model architecture to use for training (e.g., ResNext50, ViT, 3DCNN, and ResNext121).
- `--cuda`: CUDA device to use for training. Specify in the format "cuda:x", where x is the GPU index (e.g., cuda:0, cuda:1, etc.).
- `--batch_size`: Batch size for both training and validation.
- `--contrastive_loss`: Set to 1 to use contrastive loss, 0 to not use it.
- `--dropout`: Dropout rate to be applied in the model.
- `--contrastive_mode`: Contrastive learning mode (e.g., 'augmentation' or 'MacOp'). Set to 'None' if not using contrastive learning.
- `--augment`: Apply data augmentation if set to `True`.
- `--precompute`: Set to `True` to precompute data augmentation, `False` otherwise.
- `--test`: Set to `True` to test training pipeline with only first 10 scans, `False` for full training.
- `--dataset`: Dataset filename (e.g., 'local_database8_Macular_SubMRN_v4.csv').
- `--image_size`: Image size as a tuple (e.g., 128,200,200).
- `--epochs`: Number of epochs for training.
- `--lr`: Learning rate.
- `--weight_decay`: Weight decay.
- `--add_denoise`: Set to `True` to add denoised scans, applicable only when `contrastive_mode` is 'None'.
- `--prob`: Probability of transformation (e.g., 0.5).
- `--imbalance_factor`: Multiplicative factor to increase the number of glaucoma scans.
- `--use_focal_loss`: Use Focal Loss as opposed to Cross Entropy Loss if set to `True`.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

