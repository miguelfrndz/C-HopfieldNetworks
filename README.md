# Hopfield Networks in C
A simple implementation of traditional Hopfield Networks (w/ binary states) in C. A two-class binarized subset of MNIST is used as a toy example.

## How to Use This Code?

1) Run the `generateData.py` script to load and preprocess the example data.
2) Run `make` or `make debug` to compile the code (*Note: Compiling the code in DEBUG mode will print the network state and energy at every training iteration.*).
3) Run the binaries in `./HopfieldNetworks`

## Ressources
- ["Neural networks and physical systems with emergent collective computational abilities" (Seminal Paper)](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/)

- [Wikipedia Article on Hopfield Networks](https://en.wikipedia.org/wiki/Hopfield_network)

- ["Hopfield Networks is All You Need" (Paper)](https://arxiv.org/abs/2008.02217)

- [Good Blog Reference on Hopfield Networks](https://ml-jku.github.io/hopfield-layers/)

