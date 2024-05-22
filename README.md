# Neural-Homomorphic-Vocoder
Unofficial PyTorch implementation of [Neural Homomorphic Vocoder](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html) by Zhijun Liu, Kuan Chen, Kai Yu.

This paper propose the neural homomorphic vocoder (NHV), a source-filter model based neural vocoder framework.

**Abstract :**
NHV synthesizes speech by filtering impulse trains and noise with linear time-varying (LTV) filters. A neural network controls the LTV filters by estimating complex cepstrums of time-varying impulse responses given acoustic features. The proposed framework can be trained with a combination of multi-resolution STFT loss and adversarial loss functions. Due to the use of DSP-based synthesis methods, NHV is highly efficient, fully controllable and interpretable. A vocoder was built under the framework to synthesize speech given log-Mel spectrograms and fundamental frequencies. While the model cost only 15 kFLOPs per sample, the synthesis quality remained comparable to baseline neural vocoders in both copy-synthesis and text-to-speech.

Audio samples and further information are provided in the [online supplement](https://zjlww.github.io/nhv-web/).
