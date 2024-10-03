# (tentative) How Much Have We Achieved? Revisiting and Benchmarking on LLMs for Graph Tasks.

This is our ongoing project to explore the effectiveness of Large Language Models (LLMs) for graph tasks using two popular approaches:

* **LLM as Encoder**: Utilize LLMs' text representation capabilities to generate node embeddings. Compare these with embeddings from smaller language models and shallow methods.
* **LLM as Predictor**: Harness LLMs' reasoning abilities to directly predict node labels. Evaluate this in both supervised and zero-shot scenarios.

In terms of implementation and engineering, unlike existing LLM+Graph benchmarks (e.g., [GLBench](https://github.com/NineAbyss/GLBench)) that merely collect code repositories without unified integration, we will develop **a comprehensive sandbox** where other researchers can easily invoke LM/LLM/GNN/LLM+GNN pipelines and observe results seamlessly.




## Work in Progress 

### **RQ1 How Effective Are LLMs as Enhancers?**

- [x] GNN as Backbones - investigate the effectiveness of LLM's generated embeddings, compare it with small-scale LMs and shallow embedding

- [x] TAPE Implementation and Experiments


- [ ] ZeroG Implementation and Experiments



## Dataset & Setup

- **Environment Setup**: Check out `requirements.txt` for dependencies.
- **Datasets**: Download the processed datasets from [this link](https://drive.google.com/drive/folders/1WfBIPA3dMd8qQZ6QlQRg9MIFGMwnPdFj) and place them in the `datasets` folder.


