# HeartNet

For a high-level overview of this project, check out this [blog post](https://oapostrophe.github.io/heartnet/).  For a video presentation and more detailed writeup on our methodology, check out the report on our [project website](https://oapostrophe.github.io/HeartNet/).

The trained model can be demoed by downloading `app.py` and `demo_model.pkl`, installing [streamlit](https://anaconda.org/conda-forge/streamlit), then running:
```shell
streamlit run app.py
```
You can then visit the provided url in your browser; for convenience, sample generated MI and Normal EKG images are provided in the `/test files` directory.

To use any of the other files, you'll have to download the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) dataset.

The important files are the following:
- `app.py` StreamLit-based web interface using a trained model
- `dataset generation/generate_imgset1.py` our initial attempt generating a dataset directly with MatPlotLib; these images are a bit rough-looking
- `dataset generation/generate_imgset2.py` our refined version that generates nicer-looking images
- `dataset generation/generate_imgset3.py` adds random shadows overlaying generated images
- `dataset generation/generate_rnn_imgset.py` generates individual images for each of 12 leads, for input into an RNN (rnn code currently fails to learn).
- `cnn_learner.py` trains and saves a cnn on generated images.
- `automold.py` library with image augmentation code for adding shadows

Feel free to [email me](swow2015@mymail.pomona.edu) with any questions!
