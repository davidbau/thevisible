---
title: "Is Artificial Intelligence Intelligent?"
date: 2023-04-02T17:30:38-04:00
draft: false
---
The idea that large language models could be capable of cognition is not obvious. Neural language modeling has been around since [Jeff Elman’s 1990 structure-in-time work](https://papers.baulab.info/Elman-1990.pdf), but 33 years passed between that initial idea and [first contact with ChatGPT](https://papers.baulab.info/also/Bubek-2023.pdf).

What took so long?  In this blog I write about why few saw it coming, why some remain skeptical even in the face of amazing GPT-4 behavior, why it might be succeeding anyway, and what we should study next.

# Spark-Jones’s worry about language modeling

This blog was inspired by [Karen Spark-Jones’ 2004 note](https://papers.baulab.info/also/Spark-Jones-2004.pdf) that asks whether the generative model for language modeling is rational or not. In it, she points out that language models (such as GPT) are based on a highly implausible statistical model of the mechanisms of language, like the one depicted here:

![Simple autoregressive graphical model](/images/graph_wordpred.png)

The reason this picture seems so unlikely to lead to a rational model of intelligence is that *nobody actually believes that words are actually the cause for other words!* This graphical model is just a shorthand way to express the assertion that the probability distribution of the next word y depends on nothing else other than the observation of the previous words x. When skeptical critics note that LLMs are mere [stochastic parrots](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922), or when they warn of the [false promise of ChatGPT](https://www.nytimes.com/2023/03/08/opinion/noam-chomsky-chatgpt-ai.html), the implausibility of the language modeling framework seems to be the root of it.  Words are not the cause of other words. But that is the basic assumption that language models make.

For example: a generation [Turing-test-challenge](https://en.wikipedia.org/wiki/Loebner_Prize) programs such as [Jabberwacky](https://en.wikipedia.org/wiki/Jabberwacky) and [Cleverbot](https://en.wikipedia.org/wiki/Cleverbot) and [Eugene Goostman](https://en.wikipedia.org/wiki/Eugene_Goostman) are structured like this graphical model, imitating human conversation by choosing textual responses based on a direct calculation of statistics and pattern-matching on previous words. Nobody, including the creators of those systems, seriously believe that the design of such pattern-matching engines really contains profound cognitive capabilities. They are parlor tricks, automata that provide the semblance of intelligence while in reality just following simplistic procedures.

Yet, somehow, while ChatGPT works the same way, it does seem profound.  What is the difference?  Is ChatGPT really the same as Jabberwaacky, separated only by scale, a few years of Moore's law, and slightly better programming?  Or if they are really different, what is the funadmental difference?  Where is the line?

To appreciate what Spark-Jones saw to be missing in the traditional language modeling view, contrast it to the following graph that provides a more rational model for the cognitive process underlying language.

![Simple graphical model incorporating meaning](/images/graph_meaning1.png)

Spark-Jones drew graphs like this to indicate what we are really after. Here the “m” denotes the underlying meaning m within your mind that causes you to utter the words. This model is more plausible, because it is not words that cause other words. Rather, it is our thoughts and intentions and desires that cause words to be spoken.

# GANs have the right shape but cannot do language

Before you object that such an imaginative abstraction is unrelated to practical consideration in artificial neural networks, keep in mind that it is common practice to create neural architectures with an explicit state vector that plays the role of m.  For example, contrast Pixel-CNN networks, which model an image by predicting each pixel as a consequence of the previously-seen pixels above and to the left, with generative adversarial networks (GANs), that explicitly model a small hidden state z that is the representation that predicts all the pixels, where z has no upstream dependencies.

Both architectures are able to synthesize realistic-looking images of the world. However, it seems very unlikely that the Pixel-CNN architecture would contain any sensible representation of the world, because *nobody believes that pixels cause other pixels.*

![PixelCNN vs GAN models](/images/graph_pixelcnn_vs_gan.png)

On the other hand, the GAN architecture seems more rational and more promising, because it posits a set of variables z that are the cause of all the pixels together.  In a GAN, we are hoping for z to represent “state of the world” and “state of the camera,” and for this state to lead to a reasonble set of calculations to produce the image of a realistic scene.

Remarkably, in GANs this setup shows evidence of working.  If you are unfamiliar with GANs, I recommend reading [Karras’s StyleGAN papers](https://arxiv.org/abs/1812.04948) and then the [StyleSpace paper from Wu](https://arxiv.org/abs/2011.12799). Wu found that there is a small set of bottleneck “stylespace” neurons within StyleGAN that correspond to real-world concepts such as whether a person is wearing glasses or whether they are smiling.  The results are empirical, but they have been observed in various models with several architectural variations and trained on many different data sets.

In my own research, [I previously found similar neurons in other GANs](https://arxiv.org/abs/1811.10597), but Wu’s subsequent finding on StyleGAN is the clearest example of single-neuron disentanglement seen to date. For example, when we reproduce Wu's results, we find individual neurons that control complex but very sensible things like whether the lights are turned on or off in a room.

![GAN single-neuron control](/images/gan_neuron_control.gif)

Unfortunately, GANs and similar architectures that have such a rational graphical model have not (so far) been successful at modeling anything as complex as natural language.  That might be due to the fact that the cognitive processes within a human mind that lead to language are too intricate for those architectures to imitate. For example, humans draw upon an enormous amount of knowledge when thinking about a sentence to utter.

# Transformers have the wrong shape but they can do language

Consider the sentence "Shaquille O'Neal plays basketball." Although the words “soccer” or “tennis” are often reasonable alternatives to the word “basketball,” when we talk about Shaq playing basketball, those other sports cannot be substituted in that particular sentence.

That is because the sentence is not just about making a grammatically correct statement.  The sentence reflects an actual thinking process: it reflects our knowledge about Shaq. We would only say “soccer” if we were remembering the sport of a soccer-playing athlete, like Megan Rapinoe, or if, for some reason, we held (or wanted to pretend to hold) the mistaken belief that Shaq played soccer. A decomposition of the individual ideas within our mind might be diagrammed like the figure on the left.

![Simple graphical model decomposing knowledge](/images/graph_side_by_side.png)

Unfortunately, autoregressive models have the "wrong" top-level structure to directly implement the model on the left: in an autoregressive transformer, preceding words are inputs rather than outputs.

On the other hand, research from my lab ([ROME](https://rome.baulab.info), [MEMIT](https://memit.baulab.info)) suggests that transforers can actually implement reasonable models of cognition, with explicit hidden states carrying the information about understandable components of the world.  The graphs end up looking like the picture on the right, with the same structure as a rational model with some arrows reversed.

For example, the plot below shows the effect of swapping individual hidden states between two runs of a GPT model when it predicts the word "basketball."  It is unsurprising that swapping states late in the model, at (b) will cause the model to flip its predictions to "soccer," but the suprising finding is that a small set of states deep within the model, at (a), also cause the model to flip its predictions.  In the ROME paper we find evidence that these early states correspond to the point at which the model retrieves its knowledge about which sport the athlete plays.  For example, if we intercept and modify the parameters of the model the early site (a), we can edit the model's belief and make it think that Shaq plays soccer instead.

And then in the MEMIT paper, we find that our understanding of associations is good enough that adjusting transformer memories explicitly in this way is able to provide several orders-of-magnitude better control over transformer memories than traditional fine-tuning methods.

The emergence of reasonable "world models" inside large transformers, despite arrows going the wrong way, has been observed in several other works that are woth reading about.  Be sure to read the [Othello](https://thegradient.pub/othello/) paper by Kenneth Li, et al., as well as [Neel Nanda's followup](https://www.neelnanda.io/mechanistic-interpretability/othello), the [induction heads](https://arxiv.org/abs/2209.11895) paper by Catherine Olssen, et al., the [Alchemy](https://arxiv.org/abs/2106.00737) paper by Belinda Li, et al., and the [indirect object identification](https://arxiv.org/abs/2211.00593) work by Kevin Wang, et al.  All these works reveal a little secret: transformer models do not just imitate surface statistics.  But they often construct internal computational mechanisms that mimic causal mechanisms in the real-world.

# Beyond the Turing Test

A decade ago, [Gary Marcus argued that we need to move beyond the Turing test](https://www.npr.org/2014/06/14/322008378/moving-beyond-the-turing-test-to-judge-artificial-intelligence) as our metric for the emergence of machine intelligence.  He observed that the test is too easy, that humans are too easily fooled by the mere appearance of intelligent behavior, that a true intelligent agent would contain rational thoughts, that they would understand actual relationships and motivations and causes in the world rather than just word statistics.  At the time, [Marcus gathered together a series of more difficult tests](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2650/2527) of behavior, such as open-ended questions about movies and stories, or questions about theory of mind, motivation, or desires, or problems that seem to demand an understanding of the world such as visual question answering.

Yet now a decade later when [large language models are begnning to solve all these more-difficult tasks](https://papers.baulab.info/also/Bubek-2023.pdf), Marcus continues to [meet the results with skepticism](https://www.nytimes.com/2023/01/06/opinion/ezra-klein-podcast-gary-marcus.html), pointing at flaws in logic and flaws in knowledge as evidence that the models are not really thinking. Is perfect logic and complete knowledge a prerequistite for "thought?"  Certainly most of us human beings are not capable of total factual recall and flawless logical reasoning.  Today, Marcus's solution for the dilemma seems to fall short.

This seeming need for an endless escalation of more-difficult external tasks for probing cognition suggests that Turing's basic assumption was missing an essential point.  [In 1950, Turing proposed that we do not care what the implementation is inside a machine intelligence](https://papers.baulab.info/also/Turing-1950.pdf), that a convincing artifice should be counted as just as good as the real thing.  But today, it is becoming increasingly clear fooling an external judge is not enough.  We *do* care what is inside the box.

The mechanistic-interpretability research program offers a qualitatively different alternative to the Turing test.  Instead of suggesting ever-more-difficult external tests of behavior, it aims to develop experimental methods that tear back the curtain on black-box-model and look inside.  We ask, "what does a model learn" by asking what forms of computation it is able to develop, and whether those computations capture useful, meaningful, causal structure about the world.

We are in early days, and this new research program is very immature.  We do not yet have abstractions that describe what it is that we are doing in a satisfying way.  But as the world grapples with the emergence of suprising intelligent behavior in large models, the detailed study of mechanisms within those models will become increasingly important, because it offers a new path for understanding what our machines are really learning.
