# HeartNet

A joint project by [oapostrophe](https://github.com/oapostrophe), [gkenderova](https://github.com/gkenderova), [soksamnanglim](https://github.com/soksamnanglim), [syaa2018](https://github.com/syaa2018)

For a high-level overview of this project, check out this [blog post](https://oapostrophe.github.io/heartnet/) and [90-second demo](https://www.youtube.com/watch?v=EqAU-FRu6C4).  For a full presentation and more detailed writeup on our methodology, check out the report on our [project website](https://oapostrophe.github.io/HeartNet/).

The trained model can be demoed by downloading `app.py` and `demo_model.pkl`, installing [streamlit](https://anaconda.org/conda-forge/streamlit) and [fastai](https://pypi.org/project/fastai/), then running:
```shell
streamlit run app.py
```
You can then visit the provided url in your browser; for convenience, sample generated MI and Normal EKG images are provided in the `/test files` directory.

To use any of the other files, you'll have to download the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) dataset.

The important files are the following:
- `app.py` StreamLit-based web interface using a trained model
- `dataset generation/generate_imgset1.py` our first iteration generating a dataset directly with MatPlotLib; these images look rough.
- `dataset generation/generate_imgset2.py` our second iteration that generates nicer-looking images
- `dataset generation/generate_imgset3.py` adds random simulated shadows overlaying generated images
- `dataset generation/generate_rnn_imgset.py` generates individual images for each of 12 leads, for input into an RNN (rnn code currently fails to learn).
- `dataset generation/automold.py` library with image augmentation code for adding shadows
- `training/cnn_learner.py` trains and saves a cnn on generated images.

Feel free to [email me](swow2015@mymail.pomona.edu) with any questions!
