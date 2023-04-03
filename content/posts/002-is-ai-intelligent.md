---
title: "002 Is Artificial Intelligence Intelligent?"
date: 2023-04-02T17:30:38-04:00
draft: false
---
The idea that large language models could be capable of cognition is not obvious. Neural language modeling has been around since [Jeff Elman’s 1990 structure-in-time work](https://papers.baulab.info/Elman-1990.pdf), but 33 years passed between that initial idea and [first contact with ChatGPT](https://papers.baulab.info/also/Bubek-2023.pdf).  What took so long?  One reason is that very few people anticipated the efficacy of the language modeling task.  In this blog I write about why few saw it coming, why some remain skeptical even in the face of amazing GPT-4 behavior, why it might be succeeding anyway, and what we should study next.

# Spark-Jones’s worry about language modeling

This blog was inspired after rereading [Karen Spark-Jones’ 2004 note](https://papers.baulab.info/also/Spark-Jones-2004.pdf) that asks whether the generative model for language modeling is rational or not. In it, she points out that language models (such as GPT) are based on a highly implausible statistical model of the mechanisms of language, like the one depicted here:

![Simple autoregressive graphical model](images/graph_compute.png)

The reason this picture seems so unlikely to lead to a rational model of intelligence is that *nobody actually believes that words are actually the cause for other words!* This graphical model is just a shorthand way to express the assertion that the probability distribution of the next word y depends on nothing else other than the observation of the previous words x. When skeptical critics note that LLMs are mere [stochastic parrots](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922), or when they warn of the [false promise of ChatGPT](https://www.nytimes.com/2023/03/08/opinion/noam-chomsky-chatgpt-ai.html), the implausibility of the language modeling framework seems to be the root of it.  Words are not the cause of other words. But that is the basic assumption that language models make.

For example: a generation [Turing-test-challenge](https://en.wikipedia.org/wiki/Loebner_Prize) programs such as [Jabberwacky](https://en.wikipedia.org/wiki/Jabberwacky) and [Cleverbot](https://en.wikipedia.org/wiki/Cleverbot) and [Eugene Goostman](https://en.wikipedia.org/wiki/Eugene_Goostman) are structured like this graphical model, imitating human conversation by choosing textual responses based on a direct calculation of statistics and pattern-matching on previous words. Nobody, including the creators of these systems, seriously believed that the design of these pattern-matching programs really contained profound cognitive capabilities. They are parlor tricks, automata that provide the semblance of intelligence while just following simplistic procedures.

Yet, fundamentally, ChatGPT works in the same way. And somehow, it does seem profound.  What is the difference?  Are they really the same, separated only by scale and slightly better programming?  Or if they are different, can we trace something essential that makes the difference?

To appreciate what Spark-Jones saw to be missing in the traditional language modeling view, contrast it to the following graph that provides a more rational model for the cognitive process underlying language.

![Simple graphical model incorporating meaning](images/graph_meaning1.png)

Here the “m” denotes the underlying meaning m within your mind that causes you to utter the words. This model is more plausible, because it is not words that cause other words. Rather, it is our thoughts and intentions and desires that cause words to be spoken.

# GANs have the right shape but cannot do language

Before you object that such an imaginative abstraction is unrelated to practical consideration in artificial neural networks, keep in mind that it is common practice to create neural architectures with an explicit state vector that plays the role of m.  For example, contrast Pixel-CNN networks, which model an image by predicting each pixel as a consequence of the previously-seen pixels above and to the left, with generative adversarial networks (GANs), that explicitly model a small hidden state z that is the representation that predicts all the pixels, where z has no upstream dependencies.

Both architectures are able to synthesize realistic-looking images of the world. However, it seems very unlikely that the Pixel-CNN architecture would contain any sensible representation of the world, because *nobody believes that pixels cause other pixels.*

![PixelCNN vs GAN models](images/graph_pixelcnn_vs_gan.png)

On the other hand, GAN architecture seems more rational and more promising, because it posits a set of variables z that are the cause of all the pixels together.  In a GAN, we are hoping for z to represent “state of the world” and “state of the camera,” and for this state to somehow lead to a reasonble set of calculations to produce the image of a realistic scene.

Merely setting up a computation graph in the same shape as a plausible graphical model does not guarantee that we will get an understandable representation. However, in GANs, remarkably, it does start to happen.  If you are unfamiliar with GANs, I recommend reading [Karras’s StyleGAN papers](https://arxiv.org/abs/1812.04948) and then the [StyleSpace paper from Wu](https://arxiv.org/abs/2011.12799). Wu found that there is a small set of bottleneck “stylespace” neurons within StyleGAN that correspond to real-world concepts such as whether a person is wearing glasses or whether they are smiling.  The results are empirical, but they have been observed in various models with several architectural variations and trained on many different data sets. In my own research, [I previously found similar neurons in other GANs](https://arxiv.org/abs/1811.10597), but Wu’s subsequent finding on StyleGAN is the clearest example of single-neuron disentanglement seen to date. For example, when we reproduce Wu's results, we find individual neurons that control complex but very sensible things like whether the lights are turned on or off in a room.

![GAN single-neuron control](images/gan_neuron_control.gif)

Unfortunately, GANs and similar architectures that have such a rational graphical model have not (so far) been successful at modeling anything as complex as natural language.  That might be due to the fact that the cognitive processes within a human mind that lead to language are too intricate for those architectures to imitate. For example, humans draw upon an enormous amount of knowledge when thinking about a sentence to utter.

Consider: the words “soccer” or “tennis” are often reasonable alternatives to “basketball,” but when we talk about Shaq playing basketball, those other sports cannot be substituted in that particular sentence. That is not because theey would be grammatically incorrect, but because the actual thinking process behind the sentence is not just about which words would fit in a grammar. Rather, the sentence ”Shaquille O’Neal plays basketball” is a reflection of the memory-retrieval process within our mind reflecting our knowledge about Shaq. We would only say “soccer” if we were remembering the sport of a soccer-playing athlete, like Megan Rapinoe. A decomposition of the individual ideas within our mind might be diagrammed like this.

![Simple graphical model decomposing knowledge](images/graph_meaning2.png)

For complex problems like modeling the knowledge that is revealed through human language, GANs cannot seem to learn how to do it.  At least not yet.

# Transformers have the wrong shape but they can do language

In contrast, large-scale transformers have succeeded at modeling sentences like ”Shaquille O’Neal plays basketball.” if you substitute different athletes at the start of the sentence, a large language models will also switch its prediction of the output word so that the sport at the end matches the athlete, revealing some knowledge about real-world facts that goes beyond knowledge of mere grammar. In this era when we have all had a chance to play with ChatGPT, This capability may not seem surprising but when this ability was observed in large language models just a few years ago, it was really unexpected and interesting, because their graphical model seems wrong for modeling this kind of capability. How do they do it?

The diagram below shows one hypothesis.

![Simple graphical model decomposing knowledge](images/graph_meaning2.png)

Note that this hypothetical model is just an expansion of the original x-to-y language modeling framework, with more intermediate steps.  However, I have intentionally drawn it to resemble the red-arrow graph which is a more rational view of the cognitive process.  Instead of variables "m," that stand for abstract components of meaning within a mental process, we have written hidden variables "h," which stand for physically realized hidden states within an artificial neural network.  The "h" variables are numbers, or combinations of numbers, or functions of numbers, within the neural network.  Some of the arrows flow in the wrong direction, but the independence relations have the same structure as the rational model.  Perhaps, when we model language in our irrational way, with words causing other words, it might be learning a structure like this internally.

The question posed by this hypothesis is: are there physical variables inside an artificial nerual network that directly correspond to the components of meaning in a rational model of cognition?  The top-level graphical model seems implausible, but as shown here, perhaps the lower-level graphical model captures more structure.  Might the models contain a reasonable model of cognition, with hidden variables and computations that correspond to the components of rational thought?  

The excitement of recent research in mechanistic interpretability is that there are hints that the answer could be "yes."

[to be continued]
