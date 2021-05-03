Tristan Shah

13 April 2021

# Short abstract

We train a type of Generative Adversarial Network on artificial datasets and evaluate its ability to generate new data. The datasets created for this work are RGB and grayscale images of irregular shaped blobs. We find that the system is able to generate images which are visually similar to the datasets however the model is unstable and has several issues.

# Medium abstract (\<200 words)

Unsupervised deep learning methods are a highly scalable way of deriving meaning from large datasets. One such unsupervised method is known as Generative Adversarial Networks (GAN) which learn how to generate data similar to its training data. In this paper we train and evaluate a type of GAN called DCGAN which is built specifically to learn from images. This system is trained using an artificial dataset of images which is generated in RGB and grayscale. The images in this dataset contain a variable number of irregular shaped blobs ranging from 1 to 9. After training the system separately on both variants of the dataset we find that it is able to generate images which are visually similar to their respective training data. However, we also find that there are issues with the system which make it uninterpretable. Firstly, the training process was highly unstable which resulted in noisy loss curves. Additionally, the models were only able to generate data which captured a portion of the training data distribution in a phenomenon known as mode-collapse. Lastly, we found that the noise input of the model was not as interpretable as shown in other papers using this system.

# Introduction

In many previous works supervised learning methods have been very successful in classification and regression. For these cases, especially using deep neural networks, large amounts of labeled data must be prepared which often requires human participation. However, this is not always sustainable since as model size and compute power increase exponentially more labeled data is required. In 2008 a system known as reCAPTCHA­^1^ was introduced which allowed for crowdsourcing of image labeling with a high degree of accuracy. Even this solution is not ideal due to its reliance on hand labeled data. A scalable solution towards extracting information from large unlabeled datasets is to train deep unsupervised learning systems on this data. There are several systems which are suited to this purpose including autoencoders and the subject of this paper: GANs.

The unsupervised learning system known as Generative Adversarial Networks or GANs was invented by Ian Goodfellow in 2014^2^. The training process of a GAN is an adversarial game between two neural networks. One network, the discriminator, aims to determine if a given sample is real or fake. Real data being sampled from the dataset and fake data being produced by the second network, the generator. The generator’s objective is to produce samples which fool the discriminator into classifying the generated sample as real. In order to produce the generated sample, the generator takes in a random noise vector sampled from a probability distribution. The loss function defined by Goodfellow et al.^2^ is defined as:

$$\min_{G}{\max_{D}V}(D,\ G) = E_{x\sim p_{\text{data}}(x)}\lbrack\log{D(x)\rbrack + E_{z\sim p_{z}(z)}\lbrack\log{(1 - D\left( G(z) \right))\rbrack.}}$$

Equation : Minimax loss function defined in the original GAN paper.

In order to satisfy its objective, the generator must find a set of parameters which minimize this loss. Since the generator is only involved in the expectation on the right, that is the only term it needs to minimize. Reducing a log function requires the input value to approach zero, therefore the output of the discriminator given an input from the generator would need to be 1. Or in other words, the generator needs to produce samples which are classified by the discriminator as true. It is worth mentioning that the expectation for the generator is over the probability distribution $p_{z}(z)$ from which the noise vector is sampled. As stated before, the discriminator's objective is adversarial to that of the generator. It seeks to maximize this loss, and is involved in both expectations. For the first expectation it must maximize the log of its output given a real sample which is sampled from the data distribution $p_{\text{data}}(x)$. Therefore, its output must approach 1 and must classify real samples as real. The discriminator is also involved in the second expectation over the noise distribution. Maximization of this log would require the output of the generator to be close to 0 which would mean that it is classifying samples from the generator as fake.

Since its inception, this system has spawned a wide variety of variations^3,4^ and become a popular topic in Artificial Intelligence. One particular variant of the GAN known as Deep Convolutional GAN (DCGAN)^3^ is used specifically to generate images. This is the system that we implemented and trained for the dataset of blob images. The goal of this research is to evaluate the trained DCGAN system in terms of its ability to match the data distribution, specifically the number of blobs. Additionally, we will evaluate the latent space of noise vectors supplied to the generator. In the DCGAN paper^3^ the authors were able to manipulate generated faces by finding the average noise vector responsible for certain attributes and adding and subtracting them to produce intuitive results. For example, they took the average vector responsible for producing a man with glasses, subtracted a vector for a man, and added the vector for a woman. The authors sampled the latent space around this modified vector and found that it produced images of a woman with glasses when fed to the generator. This same technique will be used for the blob dataset by adding and subtracting the vectors for different blob numbers and evaluating the number of blobs in the generated image.

# Literature Review

In this section we will review several important papers which describe variations of the original GAN. The original or 'vanilla' GAN was susceptible to some problems due to its design. Since then, many papers have built upon the work to make up for its shortcomings.

Most relevant to this research project is the DCGAN. Radford et al.^3^ introduced a GAN variant that is specifically designed to generate images. The DCGAN makes use of convolutional layers in the discriminator and convolutional transpose layers in the generator. This system showed improvements over the images produced in the original GAN paper. The authors introduce several measures which help stabilize the performance of the system including: removing any fully connected layers and replacing with convolutional ones, remove pooling layers and replace with strided convolutions, use batch norm layers, use ReLU activation in the generator, and use Leaky ReLU in the discriminator.

Another paper of interest is the one responsible for the introduction of the WGAN from Arjovsky et al.^4^ The WGAN was able to solve many of the problems involved with training GANs and greatly stabilized training. The major changes that the WGAN makes is its use of a critic instead of a discriminator. One problem with using a discriminator is that it does not provide strong gradients during training due to its use of a sigmoid function. A critic instead uses no activation function on its output which enables it to always provide a gradient. Additionally, the loss function of the WGAN is quite different.

$$\underset{{||f||}_{L \leq 1}}{m\text{ax}}{E_{x\sim p_{r}}\left\lbrack f(x) \right\rbrack - E_{x\sim p_{\theta}}\lbrack f(x)\rbrack}$$

Equation : Loss function for critic in Wasserstein GAN

In this loss, the critic's objective is to maximize this function, where the critic is denoted as $f(x)$. The left-hand expectation is over a distribution where $x$ is sampled from real data. In order to maximize this term, the critic must output high values for real data. The right-hand term is the expectation over fake data sampled from the generator, the discriminator must maximize this term by outputting large negative values. As with the original GAN implementation, the generator's goal is adversarial to the discriminator. The discriminator is optimized to create a distribution of data $p_{\theta}$ which produces a large positive number from the critic. As a result of this new loss function, the authors showed that the WGAN overcomes the problem of the generator only producing examples from a specific part of the data distribution (mode-collapse).

# Model

The images supplied to the DCGAN for this project were 64 x 64 in width and height but varied in number of channels for the grayscale and RGB case. An even number of images was created for each number of blobs ranging from 1 to 9. A total of 1.35 million images were produced using multiprocessing to speed up the process. The model architecture used for this project was constructed with the guidelines given in the DCGAN paper. Only convolutional layers were used in the discriminator and convolutional transpose in the generator, batch norm was used in both networks, and a Leaky ReLU activation function with a slope of 0.2 was used in the discriminator. One copy of the model was trained on the grayscale images and the other on the RGB. Both trained for two epochs over the dataset. In order to evaluate the models, 100 generated images were manually labeled each for grayscale and RGB.

# Results 

Loss of the discriminator and generator are shown in Figure 1. These curves display significant training instability with large spikes at regular intervals throughout the process. Several attempts were made to smooth out the training process including increasing batch size for more accurate gradient updates and decrease the learning rate to prevent large updates. Neither method was successful and training remained unstable. An additional concern from these loss curves is that the discriminator loss is much smaller than that of the generator. This is an issue because it is known that when the discriminator is perfect, the gradient supplied to the generator is very small and prevents learning.

<img src="media\image1.png" style="width:3.18in;height:1.59in" /><img src="media\image2.png" style="width:3.18in;height:1.59in" />

Figure 1: Loss of the discriminator and generator neural networks while training on the RGB images over two epochs.

During training, the ability of the discriminator to correctly identify real or fake samples was tracked in the form of accuracy shown in Figure 2. These plots show that the discriminator trains to an extremely high accuracy early on and maintains high performance throughout training. Both accuracies drop in spikes however they are not sustained. The discriminator seems to be slightly less accurate on real samples than on fake ones. Extremely high accuracy further supports the concern that the discriminator will be too certain on its predictions and therefore push its sigmoid function towards extremes where the gradient is small.

<img src="media\image3.png" style="width:3.18in;height:1.59in" /><img src="media\image4.png" style="width:3.18in;height:1.59in" />

Figure 2: Accuracy of the discriminator on real and fake data while training on the RGB images over two epochs.

Despite the unusual and unstable loss functions, the DCGAN system seems to be learning to produce meaningful results. The real and generated images from the model trained on grayscale images is shown in Figure 3. We can see that the generated images do produce irregular shaped blobs in discrete numbers. However, there are several problems with the generated images. Some of the blobs are half merged together and this is not the case in the original images which have a minimum distance between one another. There is also a checkerboard pattern present in the images in the space between the blobs. Checkerboarding is a known phenomenon when using convolutional transpose layers to up sample images (Kinoshita and Kiya)^5^. Lastly there is the problem of the images containing numbers of blobs that were not present in the original dataset. The bottom left image in the generated images contains 10 blobs whereas the dataset only contains blobs numbering from 1 to 9.

<img src="media\image5.png" style="width:3.2in;height:2.4in" /><img src="media\image6.png" style="width:3.2in;height:2.4in" />

Figure 3: Real (left) and fake (right) images for the DCGAN trained on single channel grayscale images.

Furthermore, we can see that the generated images from the RGB model are of relatively high quality shown in Figure 4. They appear clearer than even the single channel grayscale images. There is no checkerboarding in the space between blobs. The colors of the blobs also match the colors in the dataset and seem sufficiently randomized. Additionally, the blobs appear to be well formed and not overlapping. One issue is readily apparent, there seems to be a great deal of mode collapse due to the repeated patterns of blobs despite different noise vectors. Again, it is surprising that the images are as good as they are due to the generator loss being high.

<img src="media\image7.png" style="width:3.2in;height:2.4in" /><img src="media\image8.png" style="width:3.2in;height:2.4in" />

Figure 4: Real (left) and fake (right) images for the DCGAN trained on three channel RGB images.

The distribution of number of blobs are shown in Figure 5. Samples of 100 images are plotted as histograms with the number of blobs on the x axis and the count of images on the y axis. For the original dataset the distribution is mostly uniform as is expected since the actual distribution is known to be uniform. For the grayscale images there appears to be a large spike at 5 blobs and it falls off on either side. This is evidence that the model is not learning to match the data distribution since it is favoring a specific portion of the distribution. We can see that the generated images produce blobs numbering up to 12 which is well outside the number of blobs in the training data. The RGB generated images also do not display the full range as the original data, instead most of the generated samples are between 3 and 7 blobs.

<img src="media\image9.png" style="width:1.92in;height:1.44in" /><img src="media\image10.png" style="width:1.92in;height:1.44in" /><img src="media\image11.png" style="width:1.92in;height:1.44in" />

Figure 5: Distributions of number of blobs for the original dataset (left), the grayscale DCGAN (middle), and RGB DCGAN (right).

Figure 6 shows the results of experimenting with the interpretability of the noise space. The average vector responsible for producing images containing 3 blobs was subtracted from the one for 5 blobs. The space around the resultant vector was sampled and fed through the generator. We can see from the generated images they do not contain 2 blobs as one would intuitively expect. Instead, the images produced are similar variants of the same image containing 6 to 7 blobs. Additional experiments of this method were conducted and no interesting pattern was found.

<img src="media\image12.png" style="width:1.92in;height:1.44in" /><img src="media\image13.png" style="width:1.92in;height:1.44in" /><img src="media\image14.png" style="width:1.92in;height:1.44in" />

Figure 6: 16 vectors averaged for 5 blobs (left) and 3 blobs (middle). The average vectors are subtracted and the resultant vector is used as the mean of a normal distribution with standard deviation 0.25 which is sampled 4 times (right). Sampled vectors are fed to the generator to produce images.

# Conclusions

We implemented a DCGAN system and trained it on an artificial dataset of images containing blobs. The system was able to successfully generate images which were visually similar to the dataset for RGB and grayscale. While the generator did produce results meaningful results, there were problems in the training stability which caused spiky loss. The generator for both types of images seems to favor certain regions of the data distribution. This can be seen by comparing the histograms of the data and the generated images. From inspecting the generated images, common configuration of blobs can be identified which indicates mode collapse, especially in the case of the RGB model. Unfortunately, in our experiments, the noise space is not as interpretable as has been shown in the DCGAN paper. We were unable to recreate results from adding and subtracting noise vectors. There are many directions that future work could take, however implementing the WGAN^4^ system would seem to be the most promising given the problems we faced. It was shown in the WGAN paper that it addresses the training stability issue by providing stronger gradients from the critic. Also, because it directly optimizes matching the distributions of the generated data to the data it completely solves the problem of mode collapse.

# Bibliography

\[1\] ReCaptcha: <https://www.google.com/recaptcha/about/>

\[2\] GAN: <https://arxiv.org/abs/1406.2661>

\[3\] DCGAN: <https://arxiv.org/abs/1511.06434>

\[4\] WGAN: <https://arxiv.org/pdf/1701.07875.pdf>

\[5\] Checkerboarding: <https://arxiv.org/pdf/2002.02117.pdf>
