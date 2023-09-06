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

3. Install the required dependencies. It's recommended to use a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Run the main training script:

   ```
   python train.py --model_name ResNext50 --cuda cuda:2 --batch_size 4 --dropout 0.2 --contrastive_mode None --augment True
   ```

## Script Arguments

- `--model_name`: Name of the vision model architecture to use for training (e.g., ResNext50, ViT, etc.).
- `--cuda`: CUDA device to use for training. Specify in the format "cuda:x", where x is the GPU index (e.g., cuda:0, cuda:1, etc.).
- `--batch_size`: Batch size for both training and validation.
- `--dropout`: Dropout rate to be applied in the model.
- `--contrastive_mode`: Contrastive learning mode. Set to "None" if not using contrastive learning.
- `--augment`: Apply data augmentation if set to `True`.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

