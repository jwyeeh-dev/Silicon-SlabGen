# Inverse Design of Surface Geometries using Generative AI Models (CVAE)

<img width="1000" align="center" alt="System Overview" src="https://github.com/jwyeeh-dev/Silicon-SlabGen/assets/99489807/439d766e-0268-42a6-a2ad-b20dc2568a89">


## Motivation
- Considering the material space for designing new materials is laborintensive and inefficient to manually find materials with desired properties.
- Inverse design technology using AI generative models[1-5] that can create new structures based on target physical properties is able to overcome this problem.


## Objectives
- Constructing the high-quality slab structure database (DB).
- Developing the Si slab generator with the targeted ionization energy using AI generative models.


## Datasets
- Generating Si Slab DB with surface properties obtained by the extended Hubbard U+V approach[6].
- 23,137 Silicon Slab geometry with JSON file format.
- To enhance the performance of generative models, we are augmented to 80,000 Databases by Supercell operation and Translation operation method to insert at the generative model.

<div align="center">
  <img width="750" alt="스크린샷 2023-10-13 오후 1 55 29" src="https://github.com/jwyeeh-dev/Silicon-SlabGen/assets/99489807/b42b2aca-de58-4b06-a93b-caa99cc0ba61">
</div>


## Network Structures

### 1. Conditional Variational AutoEncoder (CVAE)
<div align="center">
  <img width="1000" alt="스크린샷 2023-10-13 오후 1 30 28" src="https://github.com/jwyeeh-dev/Silicon-SlabGen/assets/99489807/657e5bae-7e84-4510-975a-03efeac5e747">
</div>
<br>

### 2. Wesserstain Generative Adverserial Network (WGAN)
<div align="center">
  <img width="1000" align="center" alt="스크린샷 2023-10-13 오후 1 31 59" src="https://github.com/jwyeeh-dev/Silicon-SlabGen/assets/99489807/d34da36e-46fa-4202-9c2d-d09dab0b907d">
</div>
<br>

## Usage
### command for CVAE 
```
$ cd generate_new_structures
$ python structure_generator.py --out-dir result_end --gpu cuda --sampling-size 10000 --sampling slerp --batch-size 8
```

### Options
```
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training VAE for materials generator')
    # for data
    parser.add_argument('--materials-generator-path', default='./materials_generator/result/best_model.pth',
                        type=str, help='path to materials generator model (relative path)')
    parser.add_argument('--cell-ae-path', default='./cell/result/best_model.pth',
                        type=str, help='path to cell autoencoder path (relative path)')
    parser.add_argument('--basis-ae-path', default='./basis/result/best_model.pth',
                        type=str, help='path to basis autoencoder path (relative path)')
    parser.add_argument('--out-dir', '-o', default='result_end ',
                        type=str, help='path for output directory')

    # for model
    parser.add_argument('--cell-z-size', default=20, type=int,
                        help='size for latent variable (200) in cell image auto encoder')
    parser.add_argument('--basis-z-size', default=200, type=int,
                        help='size for latent variable (500) in basis image auto encoder')
    parser.add_argument('--z-size', default=500, type=int,
                        help='size for latent variable (200) in materials generator')

    parser.add_argument('--gpu', '-g', action='store_true', help='using gpu during training')
    parser.add_argument('--sampling-size', default=10000, type=int, help='using gpu during training')
    parser.add_argument('--sampling', choices=['random', 'slerp'], default='random',
                        help='choose sampling method, (default: random)')
    parser.add_argument('--batch-size', '-b', default=8, type=int,
                        help='mini-batch size (8)')
    parser.add_argument('--seed', default=1234, type=int, help='seed value (default: 1234)')

    return parser.parse_args()
```


## References
- *Sanchez-Lengeling and Aspuru-Guzik, Science 361, 360 (2018).*
- *Noh, Kim, Stein, Sanchez-Lengeling, Gregoire, Aspuru- Guzik, Jung, Matter 1, 1370 (2019).*
- *Kim, Lee, and Kim, Sci. Adv. 6, 9324 (2020).*
- *Court, Yildirim, Jain, and Cole, J. Chem. Inf. Model. 60, 4518 (2020).*
- *Kim, Noh, Gu, Aspuru-Guzik, and Jung, ACS Cent. Sci. 6, 1412 (2020).*
- *Lee and Son, Phys. Rev. Res 2, 043410 (2020).*


## Citation
```
@misc{HWANG_2022_ESCW,
  author = {Jae-Yeong Hwang, Weon-Gyu Lee, Sang-Hoon Lee, Young-Woo Son, and Jung-Hoon Lee},
  title = {Inverse Design of Surface Geometries using Generative AI Models},
  howpublished = {Presented at \textit{The KIAS Electronic Structure Calculation Workshop 2022}},
  year = {2022},
  note = {Date of Presentation: July 7, 2022. Accessed: Date}
}
```
