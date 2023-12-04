This the code for "Generative Design by Embedding Topology Optimization into Conditional Generative Adversarial Network"

Data Creation: the code for generating wheel
cGAN: conditional GAN to generate wheel
cGAN+TO: conditional GAN with topology optimization to generate functional designs

Training (works for both cGAN and cGAN+TO)

`python main.py --batch_size 64 --imsize 64 --dataset celeb --adv_loss hinge --version sagan_celeb`

The pre-trained weights can be found in the link: https://drive.google.com/drive/folders/12Y63kQr5QF26IJNzI_nrK4mDxRltEC2n?usp=drive_link
The weights for cGAN: epoch 553500
The weights for cGAN+TO: 555537
