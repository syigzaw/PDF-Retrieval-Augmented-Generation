# PDF-Retrieval-Augmented-Generation

In this repository, Retrieval Augmented Generation is used to extract knowledge from a PDF document (info.pdf) in order to answer questions from another PDF document (query.pdf). The extracted questions and answers are written in output.txt. There are three example outputs, in folders 1, 2, and 3.

The info.pdf document was split into pages and embedded using OpenAI's Embeddings API. The same was done for the query.pdf document. A similarity search was done over these embeddings in order to retrieve the relevant information from the info.pdf document which would be able to answer the questions in the query.pdf, and both the query and the relevant info were sent to ChatGPT to answer.

The exercise.ipynb file is a notebook containing the step-by-step process of all of this, and the exercise.py file is the script that can be run with an info.pdf file and query.pdf file in order to produce the output.txt file.
