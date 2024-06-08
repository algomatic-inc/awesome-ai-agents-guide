# Awesome AI Agents Guide

[![Algomatic](https://img.shields.io/badge/About-Algomatic-blue)](https://algomatic.jp/)
[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/algomatic-inc/awesome-ai-agents-guide/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The awesome-ai-agent-guide repository is an initial effort to put together a comprehensive list of AI/LLM Agents focused on research and products.

Please note, this repository is a voluntary project and does not list all existing AI agents. This repository is a <u>work in progress</u>, and items are being added gradually. Contributions are welcome, so **we look forward to your proactive PRs!**

**Disclaimer**<br>
・ If there are any errors in interpretation or quotations, please let us know.<br>
・ Please be sure to refer to the licenses and terms of use when using.

🌟 If this was helpful, we’d love it if you followed us!: [@AlgomaticJP](https://x.com/AlgomaticJp)

### Overview

- LLM Agent
    - [Agent](#agent)
    - [Planning](#planning-and-reasoning)
    - [Action](#action)
    - [Reason](#reason)
    - [Memory](#memory)
    - [Reflection](#reflection)
    - [Multi-modal](#multi-modal)
    - [Multi Agent](#multi-agent)
- Application
    - [Web Navigation](#web-navigation)
    - [Code Generation and Software Engineer](#code-generation-and-software-engineer)

---

## AI Agent

We are not sure about the exact definition, but one commonly known definition is the following.

> An autonomous agent is a system situated within and a part of an environment that senses that environment and acts on it, over time, in pursuit of its own agenda and so as to effect what it senses in the future.<br>
> --- Franklin and Graesser (1997)

**Survey**

- 2024.02 - Huang et al., Position Paper: Agent AI Towards a Holistic Intelligence [[arXiv](https://arxiv.org/abs/2403.00833)]
- 2023.09 - Xi et al., The Rise and Potential of Large Language Model Based Agents: A Survey [[arXiv](https://arxiv.org/abs/2309.07864)]
- 2023.09 - Zhao et al., An In-depth Survey of Large Language Model-based Artificial Intelligence Agents [[arXiv](https://arxiv.org/abs/2309.14365)]
- 2023.08 - Wang et al., A Survey on Large Language Model Based Autonomous Agents [[arXiv](https://arxiv.org/abs/2308.11432)]
- 2023.06 - Taniguchi et al., World models and predictive coding for cognitive and developmental robotics: frontiers and challenges (Advanced Robotics) [[tandfonline](https://www.tandfonline.com/doi/full/10.1080/01691864.2023.2225232?src=most-read-last-year)]

**Workshop or Tutorial**

- 2024.07 - ICML 2024 Tutorial Understanding the Role of Large Language Models in Planning [[Home](https://icml.cc/virtual/2024/tutorial/35226)]
- 2024.06 - CVPR 2024 Tutorial on Generalist Agent AI [[Home](https://multimodalagentai.github.io/)]
- 2024.05 - ICLR 2024 Workshop on LLM Agents [[Home](https://llmagents.github.io/)]
- 2023.08 - IJCAI 2023 Symposium on Large Language Models (LLM 2023) [[Home](https://bigmodel.ai/llm-ijcai23)]

**Misc**

- 2025.XX - Micheal Lanham, GPT Agents in Action [[manning](https://www.manning.com/books/gpt-agents-in-action)]
- 2024.XX - LangChain, Go autonomous  with LangChain Agents [[langchain](https://www.langchain.com/agents)]
- 2024.06 - DeepLearning.AI, AI Agents in LangGraph [[deeplearning.ai](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)]
- 2024.05 - DeepLeanring.AI, AI Agentic Design Patterns with AutoGen [[deeplearning.ai](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/)]
- 2024.05 - DeepLearning.AI, Multi AI Agent Systems with crewAI [[deeplearning.ai](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)]
- 2024.05 - DeepLearning.AI, Functions, Tools and Agents with LangChain [[deeplearning.ai](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/)]
- 2024.05 - DeepLearning.AI, Building Agentic RAG with LlamaIndex [[deeplearning.ai](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)]
- 2024.05 - Chi Wang, Agents in AutoGen [[autogen](https://microsoft.github.io/autogen/blog/2024/05/24/Agent/)]
- 2024.05 - Bavor and Taylor, The Guide to AI Agents [[SIERRA](https://sierra.ai/blog/ai-agents-guide)]
- 2024.05 - Yohei Nakajima, Future of Autonomous Agents [[X's broadcast](https://x.com/i/broadcasts/1lPKqbVgNrZGb)]
- 2024.05 - Alex Klein, The agentic era of UX [[Medium](https://uxdesign.cc/the-agentic-era-of-ux-4b58634e410b)]
- 2024.03 - Andrew Ng, Agentic Design Patterns Part 1 [[deeplearning.ai](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8Kh954rkXmE4vgpKvro3Klpjhn7IuT-Y_eXIYtgVIq9PTzwa5zFWX7FZZqv1tuDEEsTDuY)]
- 2024.03 - Harrison Chase, What's next for AI agents [[Sequoia Capital, Youtube](https://youtu.be/pBBe1pk8hf4?feature=shared)]
- 2024.02 - Vincent Koc, Generative AI Design Patterns: A Comprehensive Guide [[Medium](https://towardsdatascience.com/generative-ai-design-patterns-a-comprehensive-guide-41425a40d7d0)]
- 2023.12 - OpenAI, Practices for Governing Agentic AI Systems [[OpenAI](https://openai.com/index/practices-for-governing-agentic-ai-systems/)]
- 2023.12 - Victor Dibia, Multi-Agent LLM Applications | A Review of Current Research, Tools, and Challenges [[newsletter](https://newsletter.victordibia.com/p/multi-agent-llm-applications-a-review)]
- 2023.11 - Tanay Varshney, Introduction to LLM Agents [[NVIDIA Blog](https://developer.nvidia.com/blog/introduction-to-llm-agents/)]
- 2023.06 - Lilian Weng, LLM Powered Autonomous Agents [[Lil'Log](https://lilianweng.github.io/posts/2023-06-23-agent/)]
- Prompt Engineering Guide, LLM Agents [[promptingguide.ai](https://www.promptingguide.ai/research/llm-agents)]

**Libraries**

- 2024 - Wu et al., AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (ICLR) [[GitHub](https://github.com/microsoft/autogen)][[openreview](https://openreview.net/forum?id=tEAF9LBdgu)]
- 2024.02 - Liu et al., AgentLite: A Lightweight Library for Building and Advancing Task-Oriented LLM Agent System [[GitHub](https://github.com/SalesforceAIResearch/AgentLite)][[arXiv](https://arxiv.org/abs/2402.15538)]
- 2023.10 - Khattab et al., DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines [[arXiv](https://github.com/stanfordnlp/dspy)]
- https://github.com/langchain-ai/langchain
- https://github.com/langchain-ai/langgraph
- https://github.com/joaomdmoura/crewAI
- https://github.com/langgenius/dify
- https://platform.openai.com/docs/assistants/overview
- https://github.com/AgentOps-AI/agentops

- Vertex AI Agent Builder, [[Google](https://cloud.google.com/blog/products/ai-machine-learning/build-generative-ai-experiences-with-vertex-ai-agent-builder?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8TZzur2df1qdnGx09b-Fg94DTsc3-xXao4StKvKNU2HR51el3n8yOm0CPSw6GiAoLQNKua)]
- Agents for Amazon Bedrock, [[Amazon](https://aws.amazon.com/bedrock/agents/?nc1=h_ls)]


### Agent

- 2023 - Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models (ICLR) [[openreview](https://openreview.net/forum?id=WE_vluYUL-X)]

- 2024.05 - Liu et al., Agent Design Pattern Catalogue: A Collection of Architectural Patterns for Foundation Model based Agents [[arXiv](https://arxiv.org/abs/2405.10467)]
- 2024.02 - Zhang et al., Offline Training of Language Model Agents with Functions as Learnable Weights [[arXiv](https://arxiv.org/abs/2402.11359)]
- 2023.12 - Ge et al., LLM as OS, Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem [[arXiv](https://arxiv.org/abs/2312.03815)]
- 2023.10 - Zeng et al., AgentTuning: Enabling Generalized Agent Abilities for LLMs [[arXiv](https://arxiv.org/abs/2310.12823)]
- 2023.09 - Sumers et al., Cognitive Architectures for Language Agents [[arXiv](https://arxiv.org/abs/2309.02427)]
- 2023.05 - Xie et al., OlaGPT: Empowering LLMs With Human-like Problem-Solving Abilities [[arXiv](https://arxiv.org/abs/2305.16334)]
- 2023.03 - Shen et al., HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face [[arXiv](https://arxiv.org/abs/2303.17580)]


### Profile

**Survey**

- 2024.04 - Chen et al., From Persona to Personalization: A Survey on Role-Playing Language Agents [[arXiv](https://arxiv.org/abs/2404.18231)]
- 2024.04 - Mathur et al., Advancing Social Intelligence in AI Agents: Technical Challenges and Open Questions [[arXiv](https://arxiv.org/abs/2404.11023)]

**Papers**

- 2024.04 - Yang et al., Social Skill Training with Large Language Models [[arXiv](https://arxiv.org/abs/2404.04204)]
- 2024.02 - Xie et al., Can Large Language Model Agents Simulate Human Trust Behaviors? [[arXiv](https://arxiv.org/abs/2402.04559)]
- 2023.12 - Yan et al., LARP: Language-Agent Role Play for Open-World Games [[arXiv](https://arxiv.org/abs/2312.17653)]

### Planning and Reasoning

**Survey**

- 2023 - Valmeekam et al., On the Planning Abilities of Large Language Models: A Critical Investigation (NeurIPS) [[arXiv](https://arxiv.org/abs/2305.15771)]

- 2024.04 - Zhang et al., LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models [[arXiv](https://arxiv.org/abs/2404.01230)]
- 2024.02 - Huang et al., Understanding the planning of LLM agents: A survey [[arXiv](https://arxiv.org/abs/2402.02716)]
- 2023.12 - Sun et al., A Survey of Reasoning with Foundation Models [[arXiv](https://arxiv.org/abs/2312.11562)]
- 2023.03 - Yang et al., Foundation Models for Decision Making: Problems, Methods, and Opportunities [[arXiv](https://arxiv.org/abs/2303.04129)]

**Papers**

- 2024 - Chen et al., When is Tree Search Useful for LLM Planning? It Depends on the Discriminator (ACL) [[arXiv](https://arxiv.org/abs/2402.10890)]
- 2024 - Qiao et al., AutoAct: Automatic Agent Learning from Scratch for QA via Self-Planning (ACL) [[arXiv](https://arxiv.org/abs/2401.05268)]

- 2024 - Kim et al., An LLM Compiler for Parallel Function Calling (ICML) [[arXiv](https://arxiv.org/abs/2312.04511)]

- 2024 - Prasad et al., ADaPT: As-Needed Decomposition and Planning with Language Models (NAACL) [[arXiv](https://arxiv.org/abs/2311.05772)]
- 2024 - Zhou et al., Enhancing the General Agent Capabilities of Low-Parameter LLMs through Tuning and Multi-Branch Reasoning (NAACL) [[arXiv](https://arxiv.org/abs/2403.19962)]
- 2024 - Wang et al., RecMind: Large Language Model Powered Agent For Recommendation (NAACL) [[arXiv](https://arxiv.org/abs/2308.14296)]
- 2024 - Roy et al., FLAP: Flow-Adhering Planning with Constrained Decoding in LLMs (NAACL) [[arXiv](https://arxiv.org/abs/2403.05766)]
- 2024 - Lee et al., PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers [[openreview](https://openreview.net/forum?id=4sajV6NEnWE)]

- 2024 - Ning et al., Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation (ICLR) [[openreview](https://openreview.net/forum?id=mqVgBbNCm9)]
- 2023 - Hao et al., Reasoning with Language Model is Planning with World Model (EMNLP) [[aclanthology](https://aclanthology.org/2023.emnlp-main.507/)]
- 2023 - Press et al., Measuring and Narrowing the Compositionality Gap in Language Models (EMNLP) [[aclanthology](https://aclanthology.org/2023.findings-emnlp.378/)]
- 2023 - Gupta et al., Visual Programming: Compositional Visual Reasoning Without Training (CVPR) [[CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.html)]
- 2023 - Khot et al., Decomposed Prompting: A Modular Approach for Solving Complex Tasks (ICLR) [[openreview](https://openreview.net/forum?id=_nGgzQjzaRy)]
- 2023 - Zhou et al., Least-to-Most Prompting Enables Complex Reasoning in Large Language Models (ICLR) [[openreview](https://openreview.net/forum?id=WZH7099tgfM)]
- 2023 - Valmeekam et al., PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change (NeurIPS) [[arXiv](https://arxiv.org/abs/2206.10498)]
- 2023 - Yao et al., Tree of Thoughts: Deliberate Problem Solving with Large Language Models [[arXiv](https://arxiv.org/abs/2305.10601)]
- 2023 - Wang et al., Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models (ACL) [[aclanthology](https://aclanthology.org/2023.acl-long.147/)]
- 2023 - Subramanian et al., Modular Visual Question Answering via Code Generation (ACL) [[aclanthology](https://aclanthology.org/2023.acl-short.65/)]
- 2023 - Chen et al., Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks (TMLR) [[openreview](https://openreview.net/forum?id=YfZ4ZPt8zd)] 

- 2022 - Dua et al., Successive Prompting for Decomposing Complex Questions (EMNLP) [[aclanthology](https://aclanthology.org/2022.emnlp-main.81/)]

- 2024.05 - Xu et al., Faithful Logical Reasoning via Symbolic Chain-of-Thought [[arXiv](https://arxiv.org/abs/2405.18357)]
- 2024.05 - Stechly et al., Chain of Thoughtlessness? An Analysis of CoT in Planning [[arXiv](https://arxiv.org/abs/2405.04776)]
- 2024.05 - Verma et al., On the Brittle Foundations of ReAct Prompting for Agentic Large Language Models [[arXiv](https://arxiv.org/abs/2405.13966)]
- 2024.04 - Jin et al., Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs [[arXiv](https://arxiv.org/abs/2404.07103)]
- 2024.04 - Juneja et al., 𝙻𝙼𝟸: A Simple Society of Language Models Solves Complex Reasoning [[arXiv](https://www.arxiv.org/abs/2404.02255)]
- 2024.03 - Zhu et al., KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents [[arXiv](https://arxiv.org/abs/2403.03101)]
- 2024.02 - Stechly et al., On the Self-Verification Limitations of Large Language Models on Reasoning and Planning Tasks [[arXiv](https://arxiv.org/abs/2402.08115)]
- 2024.02 - Kambhampati et al., LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks [[arXiv](https://arxiv.org/abs/2402.01817)]
- 2023.10 - Wang et al., PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization [[arXiv](https://arxiv.org/abs/2310.16427)]
- 2023.08 - Besta et al., Graph of Thoughts: Solving Elaborate Problems with Large Language Models [[arXiv](https://arxiv.org/abs/2308.09687)]
- 2023.08 - Dagan et al., Dynamic Planning with a LLM [[arXiv](https://arxiv.org/abs/2308.06391)]
- 2023.05 - Surís et al., ViperGPT: Visual Inference via Python Execution for Reasoning [[arXiv](https://arxiv.org/abs/2303.08128)]
- 2023.05 - Xu et al., ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models [[arXiv](https://arxiv.org/abs/2305.18323)]
- 2023.05 - Brahman et al., PlaSma: Making Small Language Models Better Procedural Knowledge Models for (Counterfactual) Planning [[arXiv](https://arxiv.org/abs/2305.19472)]
- 2023.04 - Lu et al., Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models [[arXiv](https://arxiv.org/abs/2304.09842)]
- 2022.11 - Gao et al., PAL: Program-aided Language Models [[arXiv](https://arxiv.org/abs/2211.10435)]

### Action

**Survey**
- 2024.03 - Wang et al., What Are Tools Anyway? A Survey from the Language Model Perspective [[arXiv](https://arxiv.org/abs/2403.15452)]
- 2023.04 - Qin et al., Tool Learning with Foundation Models [[arXiv](https://arxiv.org/abs/2304.08354)]

**Papers**

- 2024 - Basu et al., API-BLEND: A Comprehensive Corpora for Training and Benchmarking API LLMs (ACL) [[arXiv](https://arxiv.org/abs/2402.15491)]
- 2024 - Qiao et al., Making Language Models Better Tool Learners with Execution Feedback (NAACL) [[arXiv](https://arxiv.org/abs/2305.13068)]
- 2024 - Zhang et al., Reverse Chain: A Generic-Rule for LLMs to Master Multi-API Planning (NAACL) [[arXiv](https://arxiv.org/abs/2310.04474)]
- 2024 - Huang et al., Planning and Editing What You Retrieve for Enhanced Tool Learning (NAACL) [[arXiv](https://arxiv.org/abs/2404.00450)]
- 2024 - Qian et al., Toolink: Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model (NAACL) [[arXiv](https://arxiv.org/abs/2310.05155)]
- 2024 - Zheng et al., ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval (LREC-COLING) [[arXiv](https://arxiv.org/abs/2403.06551)]
- 2024 - Xu et al., On the Tool Manipulation Capability of Open-sourced Large Language Models (ICLR) [[openreview](https://openreview.net/forum?id=iShM3YolRY)]
- 2024 - Gou et al., ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving (ICLR) [[openreview](https://openreview.net/forum?id=Ep0TtjVoap)]
- 2024 - Li et al., Tool-Augmented Reward Modeling (ICLR) [[openreview](https://openreview.net/forum?id=d94x0gWTUX)]
- 2023 - Li et al., API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs (EMNLP) [[aclanthology](https://aclanthology.org/2023.emnlp-main.187/)]
- 2023 - Jacovi et al., A Comprehensive Evaluation of Tool-Assisted Generation Strategies (EMNLP) [[aclanthology](https://aclanthology.org/2023.findings-emnlp.926/)]
- 2023 - Chen et al., ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models (EMNLP) [[aclanthology](https://aclanthology.org/2023.findings-emnlp.985/)]
- 2023 - Schick et al., Toolformer: Language Models Can Teach Themselves to Use Tools (NeurIPS) [[openreview](https://openreview.net/forum?id=Yacmpz84TH)]
- 2023 - Hao et al., ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings (NeurIPS) [[openreview](https://openreview.net/forum?id=BHXsb69bSx)]
- 2023 - Srinivasan et al., NexusRaven: a commercially-permissive Language Model for function calling (NeurIPS) [[openreview](https://openreview.net/forum?id=Md6RUrGz67)]
- 2023 - Yang et al., GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction (NeurIPS) [[openreview](https://openreview.net/forum?id=cwjh8lqmOL)]
- 2022 - Parisi et al., TALM: Tool Augmented Language Models [[arXiv](https://arxiv.org/abs/2205.12255)]
- 2024.03 - Wang et al., LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error [[arXiv](https://arxiv.org/abs/2403.04746)]
- 2024.02 - Das et al., MATHSENSEI: A Tool-Augmented Large Language Model for Mathematical Reasoning [[arXiv](https://arxiv.org/abs/2402.17231)]
- 2024.02 - Du et al., AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls [[arXiv](https://arxiv.org/abs/2402.04253)]
- 2024.02 - Mekala et al., TOOLVERIFIER: Generalization to New Tools via Self-Verification [[arXiv](https://arxiv.org/abs/2402.14158)]
- 2024.01 - Shen et al., Small LLMs Are Weak Tool Learners: A Multi-LLM Agent [[arXiv](https://arxiv.org/abs/2401.07324)]
- 2024.01 - Gao et al., Efficient Tool Use with Chain-of-Abstraction Reasoning [[arXiv](https://arxiv.org/abs/2401.17464)]
- 2024.01 - Yuan et al., EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction [[arXiv](https://arxiv.org/abs/2401.06201)]
- 2024.01 - Wang et al., TroVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks [[arXiv](https://arxiv.org/abs/2401.12869)]
- 2023.12 - NexusRaven-V2: Surpassing GPT-4 for Zero-shot Function Calling [[Nexusflow](https://nexusflow.ai/blogs/ravenv2)]
- 2023.08 - Hsieh et al., Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models [[arXiv](https://arxiv.org/abs/2308.00675)]
- 2023.07 - Qin et al., ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs [[arXiv](https://arxiv.org/abs/2307.16789)]
- 2023.06 - Song et al., RestGPT: Connecting Large Language Models with Real-World RESTful APIs [[arXiv](https://arxiv.org/abs/2306.06624)]
- 2023.06 - Tang et al., ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases [[arXiv](https://arxiv.org/abs/2306.05301)]
- 2023.05 - Cai et al., Large Language Models as Tool Makers [[arXiv](https://arxiv.org/abs/2305.17126)]
- 2023.05 - Patil et al., Gorilla: Large Language Model Connected with Massive APIs [[arXiv](https://arxiv.org/abs/2305.15334)]
- 2023.03 - Shen et al., HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face [[arXiv](https://arxiv.org/abs/2303.17580)]
- 2023.03 - Paranjape et al., ART: Automatic multi-step reasoning and tool-use for large language models [[arXiv](https://arxiv.org/abs/2303.09014)]

### Memory

**Survey**

- 2024.04 - Zhang et al., A Survey on the Memory Mechanism of Large Language Model based Agents [[arXiv](https://arxiv.org/abs/2404.13501)][[GitHub](https://github.com/nuster1128/LLM_Agent_Memory_Survey)]

**Papers**

- 2024 - Na et al., Efficient Episodic Memory Utilization of Cooperative Multi-Agent Reinforcement Learning (ICLR) [[openreview](https://openreview.net/forum?id=LjivA1SLZ6)]
- 2024.06 - Yang et al., Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models [[arXiv](https://arxiv.org/abs/2406.04271)]
- 2023.10 - Zhang et al., Retrieve Anything To Augment Large Language Models [[arXiv](https://arxiv.org/abs/2310.07554)]
- 2023.05 - Modarressi et al., RET-LLM: Towards a General Read-Write Memory for Large Language Models [[arXiv](https://arxiv.org/abs/2305.14322)]

### Reflection

**Survey**

- 

**Papers**

- 2024 - Yu et al., Teaching Language Models to Self-Improve through Interactive Demonstrations (NAACL) [[arXiv](https://arxiv.org/abs/2310.13522)]
- 2024 - Xu et al., LLMRefine: Pinpointing and Refining Large Language Models via Fine-Grained Actionable Feedback (NAACL) [[arXiv](https://arxiv.org/abs/2311.09336)]
- 2023 - Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (NeurIPS) [[openreview](https://openreview.net/forum?id=vAElhFcKW6)]
- 2023 - Madaan et al., Self-Refine: Iterative Refinement with Self-Feedback (NeurIPS) [[openreview](https://openreview.net/forum?id=S37hOerQLB)]

- 2024.04 - Naik et al., Generating Situated Reflection Triggers about Alternative Solution Paths: A Case Study of Generative AI for Computer-Supported Collaborative Learning [[arXiv](https://arxiv.org/abs/2404.18262)]
- 2024.04 - Tian et al., Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing [[arXiv](https://arxiv.org/abs/2404.12253)]
- 2024.03 - Song et al., Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents [[arXiv](https://arxiv.org/abs/2403.02502)]


### Multi-modal

**Survey**

- 2024.01 - Agent AI: Surveying the Horizons of Multimodal Interaction [[arXiv](https://arxiv.org/abs/2401.03568)]

**Papers**

- 2023 - Hu et al., AVIS: Autonomous Visual Information Seeking with Large Language Model Agent (NeurIPS) [[openreview](https://openreview.net/forum?id=7EMphtUgCI&referrer=%5Bthe%20profile%20of%20Cordelia%20Schmid%5D(%2Fprofile%3Fid%3D~Cordelia_Schmid1))]

- 2024.04 - Shaham et al., A Multimodal Automated Interpretability Agent [[arXiv](https://arxiv.org/abs/2404.14394)]
- 2024.03 - Mar et al., Modeling Multimodal Social Interactions: New Challenges and Baselines with Densely Aligned Representations [[arXiv](https://arxiv.org/abs/2403.02090)]
- 2022.09 - Liang et al., Code as Policies: Language Model Programs for Embodied Control [[arXiv](https://arxiv.org/abs/2209.07753)]

### Environments

**Survey**

**Papers**

- 2024.06 - Xi et al., AgentGym: Evolving Large Language Model-based Agents across Diverse Environments [[arXiv](https://arxiv.org/abs/2406.04151)]
- 2024.04 - Xie et al., OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments [[arXiv](https://arxiv.org/abs/2404.07972)]
- 2024.03 - SIMA Team, Scaling Instructable Agents Across Many Simulated Worlds [[arXiv](https://arxiv.org/abs/2404.10179v2)]
- 2023.05 - Wang et al., Voyager: An Open-Ended Embodied Agent with Large Language Models [[arXiv](https://arxiv.org/abs/2305.16291)]
- 2023.04 - Park et al., Generative Agents: Interactive Simulacra of Human Behavior [[arXiv](https://arxiv.org/abs/2304.03442)]


### Multi Agent

**Survey**

- 2024.02 - Han et al., LLM Multi-Agent Systems: Challenges and Open Problems [[arXiv](https://arxiv.org/abs/2402.03578)]
- 2024.01 - Guo et al., Large Language Model based Multi-Agents: A Survey of Progress and Challenges [[arXiv](https://arxiv.org/abs/2402.01680)]
- 2024.01 - Cheng et al., Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects [[arXiv](https://arxiv.org/abs/2401.03428)]

**Papers**

- 2024 - Zhang et al., ProAgent: Building Proactive Cooperative Agents with Large Language Models (AAAI) [[arXiv](https://arxiv.org/abs/2308.11339)]

- 2024 - Chen et al., CoMM: Collaborative Multi-Agent, Multi-Reasoning-Path Prompting for Complex Problem Solving (NAACL) [[arXiv](https://arxiv.org/abs/2404.17729)]
- 2024 - Gong et al., MindAgent: Emergent Gaming Interaction (NAACL) [[arXiv](https://arxiv.org/abs/2309.09971)]

- 2024 - Zhang et al., Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View (ICLR) [[openreview](https://openreview.net/forum?id=ueqTjOcuLc)]
- 2024 - Du et al., Improving Factuality and Reasoning in Language Models through Multiagent Debate (ICLR) [[openreview](https://openreview.net/forum?id=QAwaaLJNCk)]
- 2024 - Chen et al., AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors (ICLR) [[openreview](https://openreview.net/forum?id=EHg5GDnyq1)]
- 2024 - Chen et al., AutoAgents: A Framework for Automatic Agent Generation (ICLR) [[openreview](https://openreview.net/forum?id=PhJUd3mbhP)]
- 2024 - Wang et al., Adapting LLM Agents Through Communication (ICLR) [[openreview](https://openreview.net/forum?id=wOelVq8fwL)]

- 2023 - Xiong et al., Examining Inter-Consistency of Large Language Models Collaboration: An In-depth Analysis via Debate (EMNLP) [[aclanthology](https://aclanthology.org/2023.findings-emnlp.508/)]

- 2024.05 - Sarkar et al., Normative Modules: A Generative Agent Architecture for Learning Norms that Supports Multi-Agent Cooperation [[arXiv](https://arxiv.org/abs/2405.19328)]
- 2024.05 - Sun et al., Facilitating Multi-Role and Multi-Behavior Collaboration of Large Language Models for Online Job Seeking and Recruiting [[arXiv](https://arxiv.org/abs/2405.18113)]
- 2024.04 - Yue et al., MathVC: An LLM-Simulated Multi-Character Virtual Classroom for Mathematics Education [[arXiv](https://arxiv.org/abs/2404.06711)]
- 2024.02 - Wang et al., Multi-Agent Collaboration Framework for Recommender Systems [[arXiv](https://arxiv.org/abs/2402.15235)]
- 2024.02 - Li et al., Can LLMs Speak For Diverse People? Tuning LLMs via Debate to Generate Controllable Controversial Statements [[arXiv](https://arxiv.org/abs/2402.10614)]
- 2024.02 - Fang et al., A Multi-Agent Conversational Recommender System [[arXiv](https://arxiv.org/abs/2402.01135)]
- 2023.10 - Agashe et al., LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Modes [[arXiv](https://arxiv.org/abs/2310.03903)]
- 2023.07 - Nascimento et al., Self-Adaptive Large Language Model (LLM)-Based Multiagent Systems [[arXiv](https://arxiv.org/abs/2307.06187)]
- 2023.07 - Wang et al., Unleashing the Emergent Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration [[arXiv](https://arxiv.org/abs/2307.05300)]
- 2023.06 - Cui et al., Chatlaw: A Multi-Agent Collaborative Legal Assistant with Knowledge Graph Enhanced Mixture-of-Experts Large Language Model [[arXiv](https://arxiv.org/abs/2306.16092)]
- 2023.05 - Liang et al., Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate [[arXiv](https://arxiv.org/abs/2305.19118)]

## Application

### Information Seeking

- 2024.04 - Chen et al., ChatShop: Interactive Information Seeking with Language Agents [[arXiv](https://arxiv.org/abs/2404.09911)]

### Web Navigation

- 2024 - Tao et al., WebWISE: Unlocking Web Interface Control for LLMs via Sequential Exploration (NAACL) [[arXiv](https://arxiv.org/abs/2310.16042)]
- 2024 - Wang et al., Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception (ICLR) [[openreview](https://openreview.net/forum?id=jE6pDYCnVF)]
- 2024 - Gur et al., A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis (ICLR) [[openreview](https://openreview.net/forum?id=9JQtrumvg8)]
- 2024 - Furuta et al., Multimodal Web Navigation with Instruction-Finetuned Foundation Models (ICLR) [[openreview](https://openreview.net/forum?id=efFmBWioSc)]
- 2024 - Zhang et al., You Only Look at Screens: Multimodal Chain-of-Action Agents (ICLR) [[openreview](https://openreview.net/forum?id=iSAgvYhZzg)]
- 2024 - Zhou et al., WebArena: A Realistic Web Environment for Building Autonomous Agents (ICLR) [[openreview](https://openreview.net/forum?id=oKn9c6ytLx)]

- 2024 - AutoDroid: LLM-powered Task Automation in Android (MobiCom) [[arXiv](https://arxiv.org/abs/2308.15272)]

- 2023 - Ma et al., LASER: LLM Agent with State-Space Exploration for Web Navigation (NeurIPS) [[openreview](https://openreview.net/forum?id=sYFFyAILy7)]
- 2023 - Deng et al., Mind2Web: Towards a Generalist Agent for the Web (NeurIPS) [[arXiv](https://arxiv.org/abs/2306.06070)]

- 2022 - Yao et al., WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents (NeurIPS) [[neurips.cc](https://papers.neurips.cc/paper_files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html)]

- 2020 - Li et al., Mapping Natural Language Instructions to Mobile UI Action Sequences (ACL) [[aclanthology](https://aclanthology.org/2020.acl-main.729/)]
- 2018 - Liu et al., Reinforcement Learning on Web Interfaces Using Workflow-Guided Exploration (ICLR) [[openreview](https://openreview.net/forum?id=ryTp3f-0-)]
- 2017 - Shi et al., World of Bits: An Open-Domain Platform for Web-Based Agents (ICML) [[PMLR](https://proceedings.mlr.press/v70/shi17a)]


- 2024.06 - Wang et al., Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration [[arXiv](https://arxiv.org/abs/2406.01014)]
- 2024.05 - Tan et al., Towards General Computer Control: A Multimodal Agent for Red Dead Redemption II as a Case Study [[arXiv](https://arxiv.org/abs/2403.03186)]
- 2024.05 - Rawles et al., AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents [[arXiv](https://arxiv.org/abs/2405.14573v1)]
- 2024.04 - Zhang et al., MMInA: Benchmarking Multihop Multimodal Internet Agents [[arXiv](https://arxiv.org/abs/2404.09992)]
- 2024.04 - Lai et al., AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Web Navigating Agent [[arXiv](https://arxiv.org/abs/2404.03648)]
- 2024.04 - Huang et al., AutoCrawler: A Progressive Understanding Web Agent for Web Crawler Generation [[arXiv](https://arxiv.org/abs/2404.12753)]
- 2024.04 - Pan et al., Autonomous Evaluation and Refinement of Digital Agents [[arXiv](https://arxiv.org/abs/2404.06474v2)]
- 2024.03 - Drouin et al., WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks? [[arXiv](https://arxiv.org/abs/2403.07718)]
- 2024.02 - Zhang et al., UFO: A UI-Focused Agent for Windows OS Interaction [[arXiv](https://arxiv.org/abs/2402.07939)]
- 2024.02 - Lù et al., WebLINX: Real-World Website Navigation with Multi-Turn Dialogue [[arXiv](https://arxiv.org/abs/2402.05930)]
- 2024.02 - Baechler et al., ScreenAI: A Vision-Language Model for UI and Infographics Understanding [[arXiv](https://arxiv.org/abs/2402.04615)]
- 2024.01 - Zheng et al., GPT-4V(ision) is a Generalist Web Agent, if Grounded [[arXiv](https://arxiv.org/abs/2401.01614)]
- 2023.12 - Zhang et al., AppAgent: Multimodal Agents as Smartphone Users [[arXiv](https://arxiv.org/abs/2312.13771)]
- 2023.11 - Yan et al., GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation [[arXiv](https://arxiv.org/abs/2311.07562)]
- 2023.11 - Furuta et al., Exposing Limitations of Language Model Agents in Sequential-Task Compositions on the Web [[arXiv](https://arxiv.org/abs/2311.18751)]
- 2023.10 - Ma et al., How to Teach Programming in the AI Era? Using LLMs as a Teachable Agent for Debugging [[arXiv](https://arxiv.org/abs/2310.05292)]
- 2023.07 - Rawles et al., Android in the Wild: A Large-Scale Dataset for Android Device Control [[arXiv](https://arxiv.org/abs/2307.10088)]
- 2022.02 - Humphreys et al., A Data-Driven Approach for Learning to Control Computers [[arXiv](https://arxiv.org/abs/2202.08137)]
- 2021.07 - Nakano et al., WebGPT: Browser-assisted question-answering with human feedback [[arXiv](https://arxiv.org/abs/2112.09332?ref=ja.stateofaiguides.com)]
- 2021.05 - Toyama et al., AndroidEnv: A Reinforcement Learning Platform for Android [[arXiv](https://arxiv.org/abs/2105.13231)]
- 2021.05 - Shvo et al., AppBuddy: Learning to Accomplish Tasks in Mobile Apps via Reinforcement Learning [[arXiv](https://arxiv.org/abs/2106.00133)]


### Code Generation and Software Engineer

**Survey**

- 2023.11 - Zheng et al., A Survey of Large Language Models for Code: Evolution, Benchmarking, and Future Trends [[arXiv](https://arxiv.org/abs/2311.10372)]
- 2023.10 - Fan et al., Large Language Models for Software Engineering: Survey and Open Problems [[arXiv](https://arxiv.org/abs/2310.03533)]
- 2023.07 - Wang et al., Software Testing with Large Language Models: Survey, Landscape, and Vision (IEEE) [[arXiv](https://arxiv.org/abs/2307.07221)]

**Paper**

- 2024 - Qian et al., ChatDev: Communicative Agents for Software Development (ACL) [[arXiv](https://arxiv.org/abs/2307.07924)]

- 2024 - Wang et al., Executable Code Actions Elicit Better LLM Agents (ICML) [[arXiv](https://arxiv.org/abs/2402.01030)]
- 2024 - Nan et al., On Evaluating the Integration of Reasoning and Action in LLM Agents with Database Question Answering (NAACL) [[arXiv](https://arxiv.org/abs/2311.09721)]
- 2024 - Olausson et al., Is Self-Repair a Silver Bullet for Code Generation? (ICLR) [[openreview](https://openreview.net/forum?id=y0GJXRungR)]
- 2024 - Hong et al., MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework (ICLR) [[openreview](https://openreview.net/forum?id=VtmBAGCN7o)]
- 2024 - Jimenez et al., SWE-bench: Can Language Models Resolve Real-world Github Issues? (ICLR) [[openreview](https://openreview.net/forum?id=VTF8yNQM66)]

- 2023 - Li et al., CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society (NeurIPS) [[arXiv](https://arxiv.org/abs/2303.17760)]

- 2023 - Dibia, LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models (ACL) [[aclanthology](https://aclanthology.org/2023.acl-demo.11/)]
- 2023 - Zhang et al., Self-Edit: Fault-Aware Code Editor for Code Generation (ACL) [[openreview](https://aclanthology.org/2023.acl-long.45/)]


- 2024.05 - Yang et al., SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering [[arXiv](https://arxiv.org/abs/2405.15793)]
- 2024.05 - Tang et al., Code Repair with LLMs gives an Exploration-Exploitation Tradeoff [[arXiv](https://arxiv.org/abs/2405.17503)]
- 2024.04 - Zhang et al., AutoCodeRover: Autonomous Program Improvement [[arXiv](https://arxiv.org/abs/2404.05427)]
- 2024.03 - Jain et al., LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code [[arXiv](https://arxiv.org/abs/2403.07974)]
- 2024.03 - Tufano et al., AutoDev: Automated AI-Driven Development [[arXiv](https://arxiv.org/abs/2403.08299)]
- 2024.03 - Tao et al., MAGIS: LLM-Based Multi-Agent Framework for GitHub Issue Resolution [[arXiv](https://arxiv.org/abs/2403.17927)]
- 2024.02 - Hong et al., Data Interpreter: An LLM Agent For Data Science [[arXiv](https://arxiv.org/abs/2402.18679)]
- 2024.02 - Zheng et al., OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement [[arXiv](https://arxiv.org/abs/2402.14658)]
- 2023.12 - Huang et al., AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation [[arXiv](https://arxiv.org/abs/2312.13010)]
- 2023.11 - Qiao et al., TaskWeaver: A Code-First Agent Framework [[arXiv](https://arxiv.org/abs/2311.17541)]
- 2023.10 - Huang et al., MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation [[arXiv](https://arxiv.org/abs/2310.03302)]
- 2023.06 - Jiang et al., SelfEvolve: A Code Evolution Framework via Large Language Models [[arXiv](https://arxiv.org/abs/2306.02907)]
- 2023.04 - Ma et al., Demonstration of InsightPilot: An LLM-Empowered Automated Data Exploration System [[arXiv](https://arxiv.org/abs/2304.00477)]
- 2022.11 - Lai et al., DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation [[arXiv](https://arxiv.org/abs/2211.11501)]

## Recruitment Information

<img src="assets/algomatic.png" width="50%"/>

Algomatic creates generative AI-native businesses across various fields.<br>
We are looking for colleagues with diverse skills.

<a href="https://jobs.algomatic.jp/">Learn More</a>

