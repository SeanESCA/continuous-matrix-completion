# Continuous matrix completion

In 2025, I undertook an undergraduate research project that aimed to:

- Benchmark the performance of stochastic gradient descent (SGD) and [Adam](https://arxiv.org/abs/1412.6980) against [alternating steepest descent (ASD)](https://www.sciencedirect.com/science/article/pii/S1063520315001062?via%3Dihub) in the context of matrix completion.
- Show that bilinear interpolation can be used to improve the performance of completion algorithms.
- Test how mini-batching affects the performance of ASD.

This repo contains the code used to conduct the experiments for my project. More specifically:

- `Experiments.ipynb` runs the experiments and visualises results. The plots produced for each experiment are stored in the `img` folder.
- `utils.py` contains the functions used to run the completion algorithms.
- `continuousmatrixcompletion.pdf` is the 30-page report submitted for grading.
- `data.npz` contains the flattened spectromicroscopy data used in the experiments. You can find more details about this data set in the report.
- `experiment_data.json` contains the experiment results of over-estimating the rank of the synthetic matrix, and finding the optimal rank of the spectromicroscopy data. You can find more details about these experiments in the report and `Experiment.ipynb`.

## Contributing

Please feel free to open an issue if you find an error in the code, or want to propose improvements to the presentation of the material.
