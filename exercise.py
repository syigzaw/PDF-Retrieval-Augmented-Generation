# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search
import fitz
import argparse
from pathlib import Path

def extract_text_and_fields(pdf_path):
    doc = fitz.open(pdf_path)

    # Extract text
    text_list = []
    for page_num in range(doc.page_count):
        text_list_within_page = []
        page = doc[page_num]
        for text_block in page.get_text_blocks():
            text_list_within_page.append((text_block[1], text_block[4]))
        
        for widget in page.widgets():
            text_list_within_page.append((widget.rect[1], f'[{widget.field_name}: {widget.field_value}]'))

        text_list.append('\n'.join(list(map(lambda x: x[1], sorted(text_list_within_page)))))
    doc.close()

    return text_list

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 3
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = """
        Questions and answers have been extracted from Document1 in order to provide you \
        with the information required to answer the questions in Document2. The answers in \
        Document1 are within square brackets like this [first_part: second_part]. The \
        first_part is a label representing the field name, and the second_part is the \
        true answer. The extraction process was imperfect, so along with these answers in \
        square brackets, there are also repeats of these answers outside the square brackets. \
        Ignore these. If there are any texts resembling answers within Document2, ignore these \
        as well and provide your own answers. Those left in answers were not able to be \
        extracted. The output should be in the format of "(Document2 question) \n (answer from \
        Document1 information) \n\n\n" for each question within Document2. Answer all questions \
        labelled as "Optional".
    """
    question = f'\n\nDocument2 questions:\n"""\n{query}\n"""'
    message = introduction
    for string in strings:
        next_article = f'\n\nDocument1 information:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question

def ask(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the provided texts."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('first_pdf', type=str)
    parser.add_argument('second_pdf', type=str)
    args = parser.parse_args()

    first_pdf_page_list = extract_text_and_fields(args.first_pdf)
    embeddings = []
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=first_pdf_page_list)
    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input
    embeddings = [e.embedding for e in response.data]
    df = pd.DataFrame({"text": first_pdf_page_list, "embedding": embeddings})
    
    second_pdf_page_list = extract_text_and_fields(args.second_pdf)

    output = ''
    for page in second_pdf_page_list:
        output += ask(page, df, GPT_MODEL) + '\n\n'
    
    with open(f'{Path(args.first_pdf).parent}/output.txt', 'w', encoding='utf-8') as file:
        file.write(output)

EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4-turbo-preview"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

if __name__ == '__main__':
    main()