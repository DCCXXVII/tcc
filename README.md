## Dataset

**Kuzushiji-MNIST** is a drop-in replacement for the MNIST dataset (28x28 grayscale, 70,000 images), provided in the original MNIST format as well as a NumPy format. Since MNIST restricts us to 10 classes, we chose one character to represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST.

**Kuzushiji-49**, as the name suggests, has 49 classes (28x28 grayscale, 270,912 images), is a much larger, but imbalanced dataset containing 48 Hiragana characters and one Hiragana iteration mark.

**Kuzushiji-Kanji** is an imbalanced dataset with a total of 3,832 Kanji characters (64x64 grayscale, 140,424 images), ranging from 1,766 examples to only a single example per class.

<p align="center">
  <img src="images/kmnist_examples.png">
  The 10 classes of Kuzushiji-MNIST, with the first column showing each character's modern hiragana counterpart.
</p>

## License

"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" 
(created by NIJL and others), doi:10.20676/00000341
