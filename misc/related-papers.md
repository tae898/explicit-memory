# Related papers

## Things to note

The related papers are mostly about continual learning and reinforcement learning with memory. Again, the biggest difference between ours and the existing works is as follows:

1. We explictly differentiate semantic and episodic memory.
   - This means that our agent have two memory systems, not one.
   - Our agent also has to learn how to encode a group of episodic memories into one semantic memory.
1. Most of the existing works save observations in memory in a tabular format, without altering them.
   - This approach doesn't scale, especially when the observations are big (e.g., image)
   - Therefore in our work, we "encode" them, just like humans do. When the observations are encoded into memories, they are saved as symbolic knowledge graphs, which saves both space and encourages XAI. At the moment our memory storage is also just a tabular format. Converting this into a symbolic knowledge graph is a TODO.
1. Some existing works have their memory as just numeric hidden representations (e.g., an RNN hidden state).
   - I don't like this approach at all. It's not interpretable. I want more symbolic.

## List of papers

They are ordered somewhat randomly.

1. [Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840)
   - This is a continual learning paper that experimented on typical continual learning datasets (e.g., Incremental CIFAR100).
   - Their optimization procedure includes gradient constraints, which allows the model not to "forget" what it has learned from the previous observations.
1. [MEMO: A Deep Network for Flexible Combination of Episodic Memories](https://arxiv.org/abs/2001.10913)
1. [Episodic Memory Reader: Learning What to Remember for Question Answering from Streaming Data](https://arxiv.org/abs/1903.06164)
   - So far this paper is the most similar to ours, since I got lot of ideas from them (e.g., using QA for rewards, replacing irrelevant memories with relevant ones, etc.)
1. [Generalizable Episodic Memory for Deep Reinforcement Learning](https://arxiv.org/abs/2103.06469)
1. [Improving Multi-hop Question Answering over Knowledge Graphs using
   Knowledge Base Embeddings](https://aclanthology.org/2020.acl-main.412/)
   - The main takeaway from this paper is that symbols in KGs can be converted into embeddings, just like a LM or MLM, so that deep learning can kick in.
1. [Integrating Episodic and Semantic Information in Memory for Natural Scenes](https://escholarship.org/uc/item/22c512rb)
   - This is a congnitive science paper, where they do distinguish semantic and episodic memories, unlike computer science based RL papers.
   - The main takeaway is *"Short  study  times  lead  to  recall  guided  by  episodic  memory,  whereas  recall  after  longer  study  times  is  more  influenced  by  semantic  information"*, which is actually what our agents are or should be learning.
1. [Encoder Based Lifelong Learning](https://arxiv.org/abs/1704.01920)
1. [Less-forgetting Learning in Deep Neural Networks](https://arxiv.org/abs/1607.00122)
1. [Time-Aware Language Models as Temporal Knowledge Bases](https://arxiv.org/abs/2106.15110)
1. [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
1. [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)
1. [Facts as Experts: Adaptable and Interpretable Neural Memory over Symbolic Knowledge](https://arxiv.org/abs/2007.00849)
1. [Working Memory Networks: Augmenting Memory Networks with a Relational Reasoning Module](https://aclanthology.org/P18-1092/)
1. [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417)
   - This is an old paper (2016).
   - The memory is not a storage here, but just a vector.
   - This memory vector is computed using the question itself. This means that you store something in your brain because you already know what will be asked.
1. [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285)
1. [Reinforcement learning and episodic memory in humans and animals: an integrative framework](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5953519/)
1. [The Missing Link Between Memory and Reinforcement Learning](https://www.frontiersin.org/articles/10.3389/fpsyg.2020.560080/full)
1. [Superposed Episodic and Semantic Memory via Sparse Distributed Representation](https://arxiv.org/abs/1710.07829)
1. [Integration of Semantic and Episodic Memories](https://www.researchgate.net/publication/319070283_Integration_of_Semantic_and_Episodic_Memories)
1. [ART neural network-based integration of episodic memory and semantic memory for task planning for robots](https://link.springer.com/article/10.1007/s10514-019-09868-x)
1. [Episodic Reinforcement Learning with Associative Memory ](https://openreview.net/pdf?id=HkxjqxBYDB)
1. [Episodic Memory in Lifelong Language Learning](https://arxiv.org/abs/1906.01076)
