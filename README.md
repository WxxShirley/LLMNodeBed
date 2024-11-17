# (tentative) How Much Have We Achieved? Revisiting and Benchmarking on LLMs for Graph Tasks.



## Work in Progress 

- [x] *Nov 17* Finish the implementation of representative LLM+GNN methods:
       
 *  **Supervised**
      * LLM as Embedder: GNN, ENGINE
      * LLM as Reasoner: TAPE
      * LLM as Predictor: Instruction Tuning, GraphGPT, LLaGA
 * **Zero-shot**
      * LLM's direct inference: GPT-4o, GPT-4
      * LLM's Cross-Domain Ability: ZeroG

- [ ] *Nov 24* Refactor the overall implementation
  * Re-organize the folders based on LLM's roles
  * Re-use commonly used modules, including dataloader (add `prefix` argument), constants, prompts, descriptions (and `classes`), checkpoints, lm&llm names and configs, etc
  * Enlarge the experimental datasets

- [ ] *Nov 25* Re-Start all experiments on refactored codes and new devices 

## Dataset & Setup

- **Environment Setup**: Check out `requirements.txt` for dependencies.
- **Datasets**: Download the processed datasets from [this link](https://drive.google.com/drive/folders/1WfBIPA3dMd8qQZ6QlQRg9MIFGMwnPdFj) and place them in the `datasets` folder.


