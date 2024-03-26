# Roadmap

### Motivation

- **Accelerate SAE Research**: Support fast experimentation to understand SAEs and improve SAE training so we can train SAEs on larger and more diverse models.
- **Make Research like Play**: Support research into language model internals via SAEs. Good tooling can make research tremendously exciting and enjoyable. Balancing modifiability and reliability with ease of understanding / access is the name of the game here.
- **Build an awesome community**: Mechanistic Interpretability already has an awesome community but as that community grows, it makes sense that there will be niches. I'd love to build a great community around Sparse Autoencoders.

### Goals

#### **SAE Training**
SAE Training features will fit into a number of categories including:

- **Making it easy to train SAEs**: Training SAEs is hard for a number of reasons and so making it easy for people to train SAEs with relatively little expertise seems like the main way this codebase will create value. 
- **Training SAEs on more models**: Supporting training of SAEs on more models, architectures, different activations within those models.
- **Being better at training SAEs**: Enabling methodological changes which may improve SAE performance as measured by reconstruction loss, Cross Entropy Loss when using reconstructed activation, L1 loss, L0 and interpretability of features as well as improving speed of training or reducing the compute resources required to train SAEs. 
- **Being better at measuring SAE Performance**: How do we know when SAEs are doing what we want them to? Improving training metrics should allow better decisions about which methods to use and which hyperparameters choices we make.
- **Training SAE variants**: People are already training “Transcoders” which map from one activation to another (such as before / after an MLP layer). These can be easily supported with a few changes. Other variants will come in time and 

#### **Analysis with SAEs**

Using SAEs to understand neural network internals is an exciting, but complicated task.

- **Feature-wise Interpretability**: This looks something like "for each feature, have as much knowledge about it as possible". Part of this will feature dashboard improvements, or supporting better integrations with Neuronpedia.
- **Mechanistic Interpretability**: This comprises the more traditional kinds of Mechanistic Interpretability which TransformerLens supports and should be supported by this codebase. Making it easy to patch, ablate or otherwise intervene on features so as to find circuits will likely speed up lots of researchers.

### Other Stuff

I think there are lots of other types of analysis that could be done in the future with SAE features. I've already explored many different types of statistical tests which can reveal interesting properties of features. There are also things like saliency mapping and attribution techniques which it would be nice to support.

- Accessibility and Code Quality: The codebase won’t be used if it doesn’t work and it also won’t get used if it’s too hard to understand, modify or read. 
Making the code accessible: This involves tasks like turning the code base into a python package.
- Knowing how the code is supposed to work: Is the code well-documented? This will require docstrings, tutorials and links to related work and publications. Getting aligned on what the code does is critical to sharing a resource like this. 
- Knowing the code works as intended: All code should be tested. Unit tests and acceptance tests are both important.
- Knowing the code is actually performant: This will ensure code works as intended. However deep learning introduces lots of complexity which makes actually running benchmarks essential to having confidence in the code. 
