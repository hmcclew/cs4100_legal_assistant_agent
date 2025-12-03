

# **Legal Information Retrieval & Reasoning Agent**

*CS4100 Final Project Report*

## **1. Abstract**

Legal information is notoriously difficult for non-experts to understand: statutes, regulations, and case law rely on dense, technical language that creates barriers to accessibility. This project builds an AI agent that retrieves, interprets, and summarizes open-license legal information using verified sources. The system combines TF-IDF retrieval, chain-of-thought prompting, and a small open-source language model (Qwen2.5-0.5B-Instruct) to answer user questions with transparent reasoning and document-grounded evidence. Results show strong performance on direct, fact-based questions and reveal limitations with vague or highly contextual legal queries.

## **2. Overview**

### **a. Problem & Motivation**

Legal language is complex, specialized, and often inaccessible to the general public. Even lawyers spend significant time locating relevant statutes or case explanations. Everyday people face high-stakes decisions (e.g. housing, employment, taxes, immigration, contracts) without an easy way to ask a simple question and receive a clear, transparent, and cited answer.

The broader motivation is that legal understanding strengthens fairness. When more people can understand their rights and obligations, they can make better decisions, spot misinformation, and participate more confidently in civic and economic life.

### **b. Why This Problem Matters**

Legal awareness affects:

* Tenants deciding whether an eviction notice is lawful
* Consumers evaluating a contract
* Immigrants navigating status requirements
* Employees wondering about workplace protections

By lowering the barrier to understanding the law, even in small ways, AI systems can support equity, access to justice, and public education. An accessible legal QA system could serve libraries, clinics, classrooms, and community organizations.

### **c. Approach**

This project builds a legal QA agent with four core components:

1. **Knowledge Base:**
   A small curated corpus of open-license legal summaries (e.g., Cornell LII, simplified case-law explanations, amendment-rights resources).

2. **Search:**
   A TF-IDF retrieval module (Milestone 1) ranks documents by semantic relevance.

3. **Prompting Techniques:**
   A ReAct-style prompt format mixes chain-of-thought reasoning with tool use (Milestone 2).

4. **Language Model:**
   Qwen/Qwen2.5-0.5B-Instruct (Milestone 3) handles natural-language reasoning, generating both intermediate thoughts and final answers.

5. **Agent Workflow:**
   A loop of **Thought → Action → Observation** enables iterative search and refinement, then a final summarized answer grounded in retrieved evidence (Milestone 4).

### **d. Rationale & Relation to Prior Work**

Legal QA systems generally rely on two families of approaches:

* **IR-first pipelines** (search → summary), which are interpretable and efficient
* **LLM-first pipelines** (direct generation), which are flexible but prone to hallucination

This project combines both through a structured ReAct prompting format. TF-IDF provides transparency: the agent retrieves exactly the documents it uses. Chain-of-thought prompting allows the model to show its reasoning rather than hiding it.

Unlike other large legal LLM systems (Lexis AI, Westlaw AI), this agent is fully open-source and reproducible.

### **e. Key Components & Limitations**

**Key Components**

* TF-IDF search
* ReAct prompting
* HuggingFace model integration
* Agent loop with action parsing
* Verified-answer summarization

**Limitations**

* Small dataset limits coverage of legal topics
* No understanding of jurisdiction-specific nuances
* Vague queries yield vague answers (as seen in the Miranda example)
* Short model context window required increasing max tokens to avoid truncation
* No grounding in statutes beyond what appears in the corpus

## **3. Approach (Technical Breakdown)**

### **a. Methodology**

The agent receives a user query and enters a structured loop:

1. Generate a **Thought** describing the next reasoning step
2. Emit an **Action**, which is parsed for tool calls (e.g., `search[...]`)
3. Receive an **Observation** containing tool output
4. Continue until issuing a `finish[...]` action

### **b. Algorithms & Models Used**

* **TF-IDF (scikit-learn)** for ranking corpus documents by cosine similarity
* **Qwen2.5-0.5B-Instruct** for text generation
* **ReAct prompting** to interleave reasoning with retrieval
* **Custom action parsing** for tool invocation and argument extraction

### **c. Assumptions and Design Choices**

* Pure retrieval-based grounding ensures no hallucinated citations
* TF-IDF is preferred over embedding search to maintain interpretability
* Small model chosen for reproducibility rather than performance
* Corpus limited to open-license materials to avoid copyright issues

### **d. Limitations**

* No domain-specific fine-tuning
* TF-IDF struggles with vague or mis-specified queries
* The model repeats itself when retrieval yields irrelevant documents
* Not designed to give legal advice—only factual summaries

## **4. Experiments**

### **a. Dataset**

Small custom corpus of U.S.-focused legal summaries (approx. 15–20 documents):

* Constitutional amendment rights
* Police procedures (Miranda, search/seizure)
* Contract formation
* Tenant rights
* Tax basics
* Immigration status categories

**Statistics:**

* Average document length: 150–400 words
* Vocabulary size: ~2,800 unique tokens
* All documents licensed for educational reuse

### **b. Implementation**

* **Transformer:** Qwen/Qwen2.5-0.5B-Instruct
* **Context window:** extended with higher `max_new_tokens` to handle dense legal text
* **Environment:** Jupyter Notebook, Python 3.10, CPU execution
* **Key parameters:**

  * Repetition penalty: increased to avoid token repetition
  * Max tokens: increased to avoid truncation

### **c. Model Architecture**

Qwen2.5-0.5B-Instruct is a decoder-only transformer with ~500M parameters. This is good for single turn instruction following and short chain-of-thought reasoning.

## **5. Results**

### **a. Main Results**

**Example 1: Direct, fact-based question**
*Question:* *What rights does the First Amendment protect?*
**Final Answer:** freedom of speech, religion, press, assembly, and petition.

The agent retrieved the correct document repeatedly and provided a complete answer.

**Trajectory snippet:**

```
Thought: To find out what rights the First Amendment protects, I need to search...
Action: search[query="First Amendment rights"]
Observation: {...}
Thought: The First Amendment protects...
Action: finish[answer="..."]
```

This demonstrates strong reliability for structured factual queries.

---

**Example 2: Vague procedural/legal question**
*Question:* *When are police required to give Miranda warnings?*
The agent repeatedly attempted to “refine the user question,” retrieving irrelevant documents like *The Starry Night* and the *Pythagorean Theorem*, reflecting a failure in search specificity.
It eventually guessed:
**Final Answer:** “When arresting a suspect.”
This is incomplete and not strictly correct (Miranda applies before custodial interrogation).

This reveals a limitation:
→ **retrieval-only grounding struggles when the query is underspecified or high-context**.

---

**Example 3: Harder doctrinal question**
*Question:* *What are the basic elements required to form a valid contract?*
**Model Answer:**

> A clear offer, consideration, a legal obligation, mutual assent, and a material breach clause.

This answer is partially correct (material breach is not an element of formation).
This reveals the model’s tendency to overgeneralize beyond the corpus.

### **b. Supplementary Experiments**

Increasing `max_new_tokens` was necessary: legal text often contains long clauses, and early truncation cut off essential definitions. Adding more tokens improved completeness and coherence.

---

## **6. Discussion**

### **Strengths**

* Transparent reasoning and traceable evidence
* Strong performance on direct Q&A (e.g., amendment rights)
* ReAct structure prevents hallucinated citations
* Entire system is reproducible and customizable

### **Weaknesses**

* TF-IDF cannot handle vague or ambiguous user questions
* Small model cannot perform deeper legal reasoning
* Corpus size limits breadth of legal coverage
* Some answers appear confident but include inaccuracies
* Repetitive loops occur when search returns irrelevant documents

### **Comparison to Existing Approaches and Problem Diagnosis**

Industrial legal research platforms use large-scale RAG systems and proprietary LLMs with jurisdictional tagging and statute parsing.
Instead, This project focuses on interpretability and simplicity rather than industrial performance.

### **Future Directions**

* Replace TF-IDF with embedding-based retrieval
* Use a larger open-source model (e.g., Qwen1.5-7B or Llama 3)
* Expand corpus coverage and add metadata like jurisdiction & date
* Introduce retrieval-augmented generation (RAG) with chunking
* Add query rewriting or classification to handle vague inputs
* Evaluate accuracy using standardized legal QA datasets

---

## **7. Conclusion**

This project successfully implemented an interpretable, retrieval-grounded legal question-answering agent using TF-IDF search, ReAct prompting, and an open-source language model. The system performs well on factual, well-defined questions and demonstrates transparent reasoning through its step-by-step trajectories. Limitations with vaguer or more complex legal queries reveal opportunities for future improvement, particularly with better retrieval, larger models, and more comprehensive datasets. Overall, the project demonstrates how lightweight AI systems can improve public access to legal information in a transparent and reproducible way.

---

## **8. References**

* https://www.law.cornell.edu/
* https://huggingface.co/Qwen/
* https://medium.com/@raquelhvaz/efficient-llm-fine-tuning-with-lora-e5edb88b64a1
* https://huggingface.co/docs/transformers/main/main_classes/text_generation
* Class notes and starter code provided by Northeastern CS4100
