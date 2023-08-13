# Finetuning GPT-3 vs Semantic Search



## **finetuning**

a kind of transfer learning.

transfer learning is about teaching the model a new task, not new information or knowledge.

finetuning == new task



- slow, difficult, expensive
- Pron to confabulation

## **Semantic Search**

Also called "neural search" or "vector search"

- Use a "semantic embedding" to search rather than keywords
  - semantic embedding: a string of numbers that represents the meaning of the text
  - it allows next-gen databases to scale very large and very fast
  - search not just with keywords or indexes, it also allows them to search based on the semantic meaning right the actual content context and the topics discussed in the actual records in your new database. They can scale very large very fast and very cheap



## **similarity**

use semantic embeddings or vector embeddings



## **diff**

Finetuning:

- slow, difficult, expensive
- Pron to confabulation 
- Teaches new task, not new information
- Requires constant retraining
- Not scalable
- Does not work for QA*

Semantic Search

- Fast, easy, cheap
- Recalls exact information
- Adding neww information is a cinch
- Adding new vectors is easy
- Infinitely scalable
- Solve half of QA



**What is finetuning good for then?**

- Teaching the model new patterns, specific formats, etc
  - ChatGPT is a pattern - short user query, long machine answer
  - Email is a pattern
  - JSON, HTML, XML, PERL, C++
  - These are all specific kinds of patterns

TLDR: if you need a highly specific and reliable pattern, then finetuning is what you need.

How does a human do QA ?

1. Formulate a question
2. Forage for information
3. Compile a corpus of data
4. Extract salient bits
5. Produce an answer

 

## Simple Steps that can do with machines

1. Index your corpus with semantic embeddings【使用语义嵌入对语料库进行索引】
   1. This makes the whole thing searchable
2. Use LLM to come up with relevant search terms, queries,etc.
   1. Translate that to a semantic embedding, use it to search your corpus
   2. Pull most relevant search results
3. Use LLM to pose your question against the relevant search results
   1. Compile all the answers together like notes