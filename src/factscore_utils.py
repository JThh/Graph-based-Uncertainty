"""All code below is taken directly from https://github.com/shmsw25/FActScore, with minor modifications.

If you use this code, please cite:
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}
"""

import json
import os
import pickle as pkl
import sqlite3
import string
import time
from collections import defaultdict
from typing import List, Optional, Tuple
import wikipedia

import numpy as np
from tqdm import tqdm

from src.models import OpenAIModel
from src.utils import remove_non_alphanumeric

FACTSCORE_CACHE_PATH = '../FActScore/.cache'
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect('../FActScore/.cache/enwiki-20230401.db', check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        if len(cursor.fetchall()) == 0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print(f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text) == str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip()) > 0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset + MAX_LENGTH])
                            offset += MAX_LENGTH

                psgs = [tokenizer.decode(tokens) for tokens in passages if
                        np.sum([t not in [0, 2] for t in tokens]) > 0]
                text = SPECIAL_SEPARATOR.join(psgs)
                
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert results is not None and len(
            results) == 1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        assert len(results) > 0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results


class Retrieval(object):
    def __init__(self, db, cache_path, embed_cache_path,
                 retrieval_type="gtr-t5-large", batch_size=None):
        self.db = db
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type == "bm25" or retrieval_type.startswith("gtr-")

        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}

    def save_cache(self):
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)

            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)

        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)

            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_gtr_passages(self, topic, retrieval_query, passages, k):
        if self.encoder is None:
            self.load_encoder()
        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
            passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device)
            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1
        query_vectors = self.encoder.encode([retrieval_query],
                                            batch_size=self.batch_size,
                                            device=self.encoder.device)[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_passages(self, topic, question, k):
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query

        if cache_key not in self.cache:
            passages = self.db.get_text_from_title(topic)
            if self.retrieval_type == "bm25":
                self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
            else:
                self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)

            self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
            assert len(self.cache[cache_key]) in [k, len(passages)]
            self.add_n += 1

        return self.cache[cache_key]
    
def get_wiki_passage(person_name): 
    # Set the language of Wikipedia you want to search in, for example, 'en' for English
    wikipedia.set_lang('en')
    try:
        # Get the page for the person
        page = wikipedia.page(person_name)

        # Print the title of the page
        print(f"Title: {page.title}\n")
        
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error, multiple articles found: {e.options}")
    except wikipedia.exceptions.PageError:
        print("Page not found.")


class FactScorer(object):
    def __init__(self,
                 data_dir=FACTSCORE_CACHE_PATH,
                 cache_dir=FACTSCORE_CACHE_PATH,
                 args=None,
                 batch_size=256):
        self.db = {}
        self.retrieval = {}
        self.batch_size = batch_size  # batch size for retrieval
        self.openai_key = os.environ.get("OPENAI_API_KEY", None)
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", None)

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.eval_model = OpenAIModel("gpt-4o-mini", args)

    def save_cache(self):
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)

    def construct_prompts_with_retrieval(
        self,
        topics: List[str],
        atomic_facts: List[List[Tuple[str, str]]],
        # model_name: str = 'claude-2.0',
        knowledge_source: Optional[str] = None,
        verbose: bool = True,
        dataset='factscore'
    ):
        # NOTE: Claude is not used at all!!!
        # assert model_name == 'claude-2.0', (
        #     'TODO: support other fact checkers. '
        #     'Currently our prompt construction uses Claude format (human / assistant).')

        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        assert len(topics) == len(atomic_facts), "`topics` and `atomic_facts` should have the same length"

        if verbose:
            topics = tqdm(topics)

        prompts = []
        n_prompts_per_topic = []
        data_to_return = defaultdict(list)

        for topic, facts in zip(topics, atomic_facts):
            data_to_return['entity'].append(topic)
            data_to_return['generated_answer'].append(facts)
            n_prompts = 0
            answer_interpretation_prompts = []
            # for atom, atom_type in facts:
            for atom in facts:
                # assert atom_type in {"Objective", 'Direct', 'Numerical Uncertainty', 'Linguistic Uncertainty'}
                atom = atom.strip()
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                if 'factscore' in dataset:
                    definition = "Human: Answer the question about {} based on the given context.\n\n".format(topic)
                elif dataset == 'nq':
                    definition = "Human: You will be provided some context and an input claim. Based on the given context, evaluate the input claim is subjective or objective. If objective, True or False. \n\n"
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(
                        psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "\n\n{}\n\nInput: {}.\nIs the input Subjective or Objective? If objective, True or False? Format your answer as one word as 'ANSWER: <Subjectve/True/False>'\nOutput:".format(definition.strip(), atom.strip())

                # Flattened list of prompts
                prompts.append(prompt)

                answer_interpretation_prompts.append(prompt)

                # But track how many there are per topic/entity
                n_prompts += 1

            n_prompts_per_topic.append(n_prompts)
            data_to_return['answer_interpretation_prompt'].append(answer_interpretation_prompts)

        return prompts, n_prompts_per_topic, data_to_return
    
    def fact_check_with_gpt(self,  topics: List[str], atomic_facts: List[List[Tuple[str, str]]], dataset):
        mark_dict = {'true': 'Y', 'false': 'N', 'subjective': 'S', 'objective': ''}
        marks, raw_results = [], []
        prompts, _, data_to_return = self.construct_prompts_with_retrieval(topics=topics, atomic_facts= atomic_facts, dataset=dataset, verbose=False)
        for prompt in prompts:
            response_content = self.eval_model.generate_given_prompt([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}])['generation']
            mark = mark_dict[response_content.split('ANSWER:')[1].strip().lower()]
            marks.append(mark)
            raw_results.append({'return': response_content, 'prompt': prompt})
        return marks, raw_results

class General_Wiki_Eval():
    def __init__(self, error_file, args) -> None:
        wikipedia.set_lang('en')
        self.eval_model = OpenAIModel("gpt-4o", args)
        self.error_file = error_file

    def general_wiki_page_prompt_construct(self, entity_name, atomic_facts):
        found = 'found'
        prompt = f'**Context**:\nTitle: {entity_name}\n\n"""\n'
        try:
            # Get the page for the person
            page = wikipedia.page(entity_name)
            prompt += page.content 
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error, multiple articles found: {e.options}")
        except wikipedia.exceptions.PageError:
            print("Page not found.")
            found = None
        prompt += '\n"""\n'
        prompt +=  '\n\n**Input Claims**: \n'
        for num, atomic_fact in enumerate(atomic_facts):
            prompt += f'Claim {num + 1}: {atomic_fact}\n'
            
        prompt += '\nFor each claim, begin by determining whether it represents a subjective opinion or an objective statement. Then, assess its accuracy as true or false based on any evidence available in the context. Document your analysis in JSON format, adhering to the following structure for each entry:{"content": "<Copy the claim here>", "reason": "<Detail your analysis and rationale for this claim>", "objectivity": "<Subjective/Objective>", "truthfulness": "<True/False>"}\n\nPlease approach each claim individually, employing step-by-step reasoning grounded in the context provided.'            
        
        return prompt, found
    
    def general_wiki_eval(self, topic, atomic_facts: List[str], batch_size=10):
        mark_dict = {'true': 'Y', 'false': 'N', 'subjective': 'S', 'objective': 'O', 'undeterminable': 'Unk', 'unknown': 'Unk'}
        
        # Make atomic_facts batched with 10 atomic facts per batch
        batched_atomic_facts = [atomic_facts[i:i + batch_size] for i in range(0, len(atomic_facts), batch_size)]
        marks, raw_results = [], []
        for atomic_fact_batch in tqdm(batched_atomic_facts):
            prompt, found = self.general_wiki_page_prompt_construct(topic, atomic_fact_batch)
            response_content = self.eval_model.generate_given_prompt([{"role": "system", "content": "You are a helpful assistant. Your task is to evaluate a series of claims based on provided context. For each claim, generate a JSON file containing a list of dictionaries, with each dictionary representing the analysis and evaluation of a claim."}, {"role": "user", "content": prompt}])['generation']
            try:
            # response_content = response_content.replace("```jsonl\n", "").replace("```json\n", "").replace("```", "")
                start, end = response_content.find('['), response_content.rfind(']')
                parsed_json = json.loads(response_content[start:end+1])
            except json.JSONDecodeError:
                marks.extend([{ 'truthfulness': '', 'objectivity': '',  'reason': '', 'content': '', 'gpt-score': ''}] * len(atomic_fact_batch))
                raw_results.append({'return': response_content, 'prompt': prompt})
                continue

            if type(parsed_json) == dict:
                
                if len(list(parsed_json.keys())) == 1:
                    parsed_json = parsed_json[list(parsed_json.keys())[0]]
                elif 'content' in parsed_json:
                    parsed_json = [parsed_json]
                else:
                    raise ValueError('Invalid JSON format')
 
            for i in range(len(parsed_json)):
                parsed_json[i]['gpt-score'] = ''
                try:
                    if parsed_json[i]['truthfulness'].strip().lower() in mark_dict:
                        parsed_json[i]['gpt-score'] = mark_dict[parsed_json[i]['truthfulness'].strip().lower()]
                    else:
                        parsed_json[i]['gpt-score'] = parsed_json[i]['truthfulness'].strip()
                        
                    if parsed_json[i]['objectivity'].strip().lower() in mark_dict:
                        parsed_json[i]['objectivity'] = mark_dict[parsed_json[i]['objectivity'].strip().lower()]
                        if parsed_json[i]['objectivity'].strip().lower() == 's':
                            parsed_json[i]['gpt-score'] = 'S'
                    else:
                        parsed_json[i]['objectivity'] = parsed_json[i]['objectivity'].strip()
                    
                except:
                    pass
                
            marks.extend(parsed_json)
            if found is None:
                prompt = 'No URL found'
            raw_results.append({'return': response_content, 'prompt': prompt})
        
        if len(marks) != len(atomic_facts):
            marks = post_process_marks(marks, atomic_facts)
        
        assert len(marks) == len(atomic_facts)
     
        return marks, raw_results   

def post_process_marks(marks, atomic_facts):
    revised_marks = []
    for atomic_fact in atomic_facts:
        for parsed_json in marks:
            if remove_non_alphanumeric(atomic_fact) == remove_non_alphanumeric(parsed_json['content']):
                revised_marks.append(parsed_json)
                break
        else:
            revised_marks.append({'truthfulness': '', 'objectivity': '',  'reason': '', 'content': atomic_fact, 'gpt-score': ''})

    return revised_marks

def rematching_marks(marks, atomic_facts):
    revised_marks = []
    for atomic_fact in atomic_facts:
        max_overlap, max_overlap_len = None, 2
        for parsed_json in marks:
            overlap = calculate_overlap(atomic_fact, parsed_json['content'])
            if overlap > max_overlap_len:
                max_overlap_len = overlap
                max_overlap = parsed_json
        
        if max_overlap is None:
            revised_marks.append({'truthfulness': '', 'objectivity': '',  'reason': '', 'content': '', 'gpt-score': ''})
        else:  
            revised_marks.append(max_overlap)
    return revised_marks

# """
# factscore_utils.py

# This file merges your advanced FactScorer (with abstaining detection, relevance checks, 
# atomic fact refinement, final scoring, etc.) from the first script, along with 
# supporting classes from your second script (DocDB, Retrieval, etc.).
# """

# import os
# import json
# import pickle as pkl
# import sqlite3
# import string
# import logging
# import time
# import numpy as np
# import wikipedia
# from collections import defaultdict
# from typing import List, Optional, Tuple
# from tqdm import tqdm

# # If you want the smaller version of GPT (like gpt-3.5-turbo) or a different LLM:
# # from src.models import OpenAIModel
# from src.models import OpenAIModel  # Adjust import to your environment

# FACTSCORE_CACHE_PATH = '../FActScore/.cache'
# SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
# MAX_LENGTH = 256

# ###############################################################################
# # Relevance classification placeholders
# ###############################################################################
# SYMBOL = 'Foo'
# NOT_SYMBOL = 'Not Foo'

# _PROMPT_PLACEHOLDER = '[PROMPT]'
# _RESPONSE_PLACEHOLDER = '[RESPONSE]'
# _STATEMENT_PLACEHOLDER = '[ATOMIC FACT]'

# _RELEVANCE_FORMAT = f"""\
# In a given RESPONSE, two subjects are considered "{SYMBOL}" if the RESPONSE \
# contains information that explains how the two subjects are related.


# Instructions:
# 1. The following STATEMENT has been extracted from the broader context of the \
# given RESPONSE to the given QUESTION.
# 2. First, state the broad subject of the STATEMENT and the broad subject of \
# the QUESTION.
# 3. Next, determine whether the subject of the STATEMENT and the subject of the \
# QUESTION should be considered {SYMBOL}, based on the given definition of \
# "{SYMBOL}."
# 4. Before showing your answer, think step-by-step and show your specific \
# reasoning.
# 5. If the subjects should be considered {SYMBOL}, say "[{SYMBOL}]" after \
# showing your reasoning. Otherwise show "[{NOT_SYMBOL}]" after showing your \
# reasoning.
# 6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
# Some examples have been provided for you to learn how to do this task.

# QUESTION:
# {_PROMPT_PLACEHOLDER}

# RESPONSE:
# {_RESPONSE_PLACEHOLDER}

# STATEMENT:
# {_STATEMENT_PLACEHOLDER}
# """

# _REVISE_FORMAT = f"""\
# Vague references include but are not limited to:
# - Pronouns (e.g., "his", "they", "her")
# - Unknown entities (e.g., "this event", "the research", "the invention")
# - Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


# Instructions:
# 1. The following STATEMENT has been extracted from the broader context of the \
# given RESPONSE.
# 2. Modify the STATEMENT by replacing vague references with the proper entities \
# from the RESPONSE that they are referring to.
# 3. You MUST NOT change any of the factual claims made by the original STATEMENT.
# 4. You MUST NOT add any additional factual claims to the original STATEMENT.
# 5. Before giving your revised statement, think step-by-step and show your \
# reasoning. As part of your reasoning, be sure to identify the subjects in the \
# STATEMENT and determine whether they are vague references. If they are vague \
# references, identify the proper entity that they are referring to and be sure \
# to revise this subject in the revised statement.
# 6. After showing your reasoning, provide the revised statement and wrap it in \
# a markdown code block.
# 7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
# Some examples have been provided for you to learn how to do this task.

# STATEMENT:
# {_STATEMENT_PLACEHOLDER}

# RESPONSE:
# {_RESPONSE_PLACEHOLDER}
# """

# def strip_string(s):
#     return s.strip()

# def extract_first_square_brackets(text):
#     import re
#     match = re.search(r'\[([^\[\]]+)\]', text)
#     result = match.group(1) if match else None
#     logging.debug(f"Extracted from square brackets: {result}")
#     return result

# def extract_first_code_block(text, ignore_language=False):
#     if not isinstance(text, str):
#         logging.error(f"Expected string in extract_first_code_block, got {type(text)}")
#         return None
#     import re
#     # If ignoring language, we allow: ``` + possible language + \n ... ```
#     pattern = r'```(?:[\w\+\-\.]+\n)?([\s\S]+?)```' if ignore_language else r'```[\w\+\-\.]+\n([\s\S]+?)```'
#     match = re.search(pattern, text)
#     result = match.group(1).strip() if match else None
#     logging.debug(f"Extracted code block: {result}")
#     return result

# def is_response_abstained(response, abstain_detection_type=None):
#     """
#     Basic function that checks if response is effectively 'abstaining', e.g.
#     returning "I cannot answer" or "Insufficient data" based on `abstain_detection_type`.
#     """
#     if abstain_detection_type is None or abstain_detection_type.lower() == 'none':
#         return False
#     # In practice, use some advanced detection (like perplexity).
#     # Here, we do a naive check
#     low_resp = response.lower()
#     if 'i cannot' in low_resp or 'i can\'t answer' in low_resp or 'not enough info' in low_resp:
#         return True
#     return False

# ###############################################################################
# # DB / Retrieval from second script
# ###############################################################################
# class DocDB(object):
#     """Sqlite backed document storage."""

#     def __init__(self, db_path=None, data_path=None):
#         self.db_path = db_path
#         self.connection = sqlite3.connect(db_path, check_same_thread=False)

#         cursor = self.connection.cursor()
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#         if len(cursor.fetchall()) == 0:
#             if data_path is None:
#                 raise ValueError(f"Empty DB. Provide data_path to build from scratch or supply a valid .db file.")
#             self.build_db(db_path, data_path)

#     def build_db(self, db_path, data_path):
#         # Implementation omitted for brevity
#         pass

#     def get_text_from_title(self, title):
#         cursor = self.connection.cursor()
#         cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
#         results = cursor.fetchall()
#         cursor.close()
#         if not results:
#             return []
#         # Split by SPECIAL_SEPARATOR
#         out = [{"title": title, "text": chunk} for chunk in results[0][0].split(SPECIAL_SEPARATOR)]
#         return out

#     def close(self):
#         self.connection.close()

# class Retrieval(object):
#     """Retrieval that uses either BM25 or a Transformer-based embedding to find relevant passages."""
#     def __init__(self, db, cache_path, embed_cache_path, retrieval_type="gtr-t5-large", batch_size=None):
#         self.db = db
#         self.cache_path = cache_path
#         self.embed_cache_path = embed_cache_path
#         self.retrieval_type = retrieval_type
#         self.batch_size = batch_size

#         self.encoder = None
#         self.load_cache()

#     def load_cache(self):
#         if os.path.exists(self.cache_path):
#             with open(self.cache_path, "r") as f:
#                 self.cache = json.load(f)
#         else:
#             self.cache = {}
#         if os.path.exists(self.embed_cache_path):
#             with open(self.embed_cache_path, "rb") as f:
#                 self.embed_cache = pkl.load(f)
#         else:
#             self.embed_cache = {}

#     def save_cache(self):
#         # Implementation for saving self.cache & self.embed_cache
#         pass

#     def get_passages(self, topic, question, k=5):
#         """
#         Retrieve up to k relevant passages for 'question' given a 'topic' as a DB title.
#         """
#         retrieval_query = f"{topic} {question.strip()}"
#         cache_key = topic + "#" + retrieval_query

#         if cache_key not in self.cache:
#             passages = self.db.get_text_from_title(topic)
#             # This snippet uses a dummy approach. Implement your logic or use BM25/embedding.
#             self.cache[cache_key] = passages[:k]  # naive
#         return self.cache[cache_key]

# ###############################################################################
# # Main advanced FactScorer from your first script
# ###############################################################################
# class FactScorer:
#     """
#     The advanced FactScorer with:
#       - Abstaining detection
#       - Relevance classification
#       - Fact refinement
#       - Summation of final scores
#     """
#     def __init__(self,
#                  model_name="retrieval+ChatGPT",
#                  data_dir=FACTSCORE_CACHE_PATH,
#                  model_dir=None,
#                  cache_dir=FACTSCORE_CACHE_PATH,
#                  openai_key="api.key",
#                  cost_estimate="consider_cache",
#                  abstain_detection_type=None,
#                  batch_size=256):
#         logging.debug(f"Initializing advanced FactScorer with model_name={model_name}...")

#         self.model_name = model_name
#         self.db = {}
#         self.retrieval = {}
#         self.batch_size = batch_size
#         self.openai_key = openai_key
#         self.abstain_detection_type = abstain_detection_type

#         self.data_dir = data_dir
#         self.cache_dir = cache_dir
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)

#         self.cost_estimate = cost_estimate
#         self.af_generator = None  # If using atomic fact generator from your system
#         self.lm = None
#         self.relevance_lm = None

#         # Example: we use an OpenAI model for relevance and refinement
#         from src.models import OpenAIModel
#         logging.debug("Setting up relevance & refinement LLM as ChatGPT model")
#         self.relevance_lm = OpenAIModel("ChatGPT", cache_file=os.path.join(cache_dir, "RelevanceChatGPT.pkl"), key_path=openai_key)

#     def save_cache(self):
#         logging.debug("Saving caches for FactScorer...")
#         if self.relevance_lm:
#             self.relevance_lm.save_cache()
#         for k, v in self.retrieval.items():
#             v.save_cache()

#     def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
#         logging.debug(f"Registering knowledge source: {name}")
#         if name in self.retrieval:
#             return
#         if db_path is None:
#             db_path = os.path.join(self.data_dir, f"{name}.db")

#         if data_path is None:
#             data_path = os.path.join(self.data_dir, f"{name}.jsonl")

#         cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
#         embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

#         self.db[name] = DocDB(db_path=db_path, data_path=data_path)
#         self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)

#     def get_score(self, topics, generations, knowledge_source=None, verbose=False):
#         """
#         Evaluate the factual correctness of each (topic, generation) pair.
#         Steps (in simplified form):
#           1) Possibly detect if generation is an abstained response.
#           2) Extract atomic facts from generation (if needed).
#           3) Revise each fact (remove pronouns, etc.)
#           4) Check relevance
#           5) Check supportedness
#           6) Compute final score
#         """
#         if knowledge_source is None:
#             knowledge_source = "enwiki-20230401"
#         if knowledge_source not in self.retrieval:
#             self.register_knowledge_source(knowledge_source)

#         # For simplicity, let's do step #2 in a naive way:
#         atomic_facts = []
#         for gen in generations:
#             # Check abstaining
#             if is_response_abstained(gen, self.abstain_detection_type):
#                 atomic_facts.append(None)
#                 continue
#             # Example: split by period as "facts"
#             facts = [f"{sent.strip()}." for sent in gen.split('.') if sent.strip()]
#             atomic_facts.append(facts)

#         scores = []
#         decisions = []
#         for idx, (topic, generation, facts) in enumerate(zip(topics, generations, atomic_facts)):
#             if facts is None:
#                 # Means it was abstained
#                 decisions.append(None)
#                 scores.append(0.0)
#                 continue

#             # Evaluate each atomic fact
#             d_list = []
#             for fact in facts:
#                 revised_fact = self.revise_fact(generation, fact)
#                 is_rel = self.check_relevance(topic, generation, revised_fact)
#                 if not is_rel:
#                     continue
#                 # Assume is_supported is always True (for demo) or do retrieval-based check:
#                 is_supported = True
#                 d_list.append({"atom": revised_fact, "is_supported": is_supported})

#             if len(d_list) == 0:
#                 final_score = 0.0
#             else:
#                 # e.g., ratio of is_supported == True
#                 final_score = sum(1 for dd in d_list if dd["is_supported"]) / len(d_list)

#             decisions.append(d_list)
#             scores.append(final_score)

#         out = {
#             "score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
#             "decisions": decisions
#         }
#         logging.info(f"Computed FactScore: {out['score']}")
#         return out
    
#     def construct_prompts_with_retrieval(
#         self,
#         topics: List[str],
#         atomic_facts: List[List[Tuple[str, str]]],
#         model_name: str = 'claude-2.0',
#         knowledge_source: Optional[str] = None,
#         verbose: bool = True,
#         dataset: str = 'factscore'
#     ):
#         """
#         Provided snippet to construct one prompt per (topic, single atomic fact).
#         Pulls up to 5 passages from self.retrieval[knowledge_source].
#         Then, for each (topic, fact), we produce a single prompt that 
#         ends with: 
#             "Is the input Subjective or Objective? If objective, True or False? 
#              Format your answer as one word as 'ANSWER: <Subjective/True/False>'"
#         """
#         # Currently only tested for 'claude-2.0' 
#         assert model_name == 'claude-2.0', (
#             'TODO: support other fact checkers. '
#             'Currently our prompt construction uses Claude format (human / assistant).'
#         )

#         if knowledge_source is None:
#             # Use the default knowledge source if none is specified
#             knowledge_source = "enwiki-20230401"

#         if knowledge_source not in self.retrieval:
#             self.register_knowledge_source(knowledge_source)

#         # We assume len(topics) == len(atomic_facts)
#         assert len(topics) == len(atomic_facts), (
#             "`topics` and `atomic_facts` should have the same length"
#         )

#         if verbose:
#             topics = tqdm(topics, desc="Constructing prompts")

#         prompts = []
#         n_prompts_per_topic = []
#         data_to_return = defaultdict(list)

#         for topic, facts in zip(topics, atomic_facts):
#             data_to_return['entity'].append(topic)
#             data_to_return['generated_answer'].append(facts)
#             n_prompts = 0
#             answer_interpretation_prompts = []

#             for atom in facts:
#                 # If your tuple is (atom_text, <some_type>), do:
#                 #   text = atom[0]
#                 # Or if your list is just a string, do:
#                 atom_text = atom.strip() if isinstance(atom, str) else atom[0].strip()

#                 # Retrieve passages (k=5)
#                 passages = self.retrieval[knowledge_source].get_passages(topic, atom_text, k=5)

#                 # Build a definition for 'factscore' dataset
#                 if dataset == 'factscore':
#                     definition = f"Human: Answer the question about {topic} based on the given context.\n\n"
#                 elif dataset == 'nq':
#                     definition = (
#                         "Human: You will be provided some context and an input claim. "
#                         "Based on the given context, evaluate the input claim is subjective or objective. "
#                         "If objective, True or False.\n\n"
#                     )
#                 else:
#                     definition = "Human: (No dataset specified) \n\n"

#                 context = ""
#                 # In reverse order for demonstration
#                 for psg_idx, psg in enumerate(reversed(passages)):
#                     # Replace any special tokens
#                     passage_text = psg["text"].replace("<s>", "").replace("</s>", "")
#                     context += f"Title: {psg['title']}\nText: {passage_text}\n\n"

#                 definition += context.strip()
#                 if not definition[-1] in string.punctuation:
#                     definition += "."

#                 # Construct final prompt
#                 prompt = (
#                     f"\n\n{definition}\n\n"
#                     f"Input: {atom_text}.\n"
#                     "Is the input Subjective or Objective? If objective, True or False?\n"
#                     "Format your answer as one word as 'ANSWER: <Subjective/True/False>'\n"
#                     "Output:"
#                 )

#                 prompts.append(prompt)
#                 answer_interpretation_prompts.append(prompt)
#                 n_prompts += 1

#             n_prompts_per_topic.append(n_prompts)
#             data_to_return['answer_interpretation_prompt'].append(answer_interpretation_prompts)

#         return prompts, n_prompts_per_topic, data_to_return

#     def fact_check_with_gpt(
#         self,
#         topics: List[str],
#         atomic_facts: List[List[str]],
#         dataset: str = 'factscore'
#     ):
#         """
#         For each (topic, list_of_facts), we call `construct_prompts_with_retrieval` 
#         to build a prompt. Then we feed them into `eval_model.generate_given_prompt(...)`.
#         We parse the single word after 'ANSWER:' to be either 'subjective', 'true', 'false'.

#         Return:
#           marks: list of classification labels (Y=True, N=False, S=Subjective, ''=Objective)
#           raw_results: list of dicts including the raw model output.
#         """
#         mark_dict = {
#             'true': 'Y',
#             'false': 'N',
#             'subjective': 'S',
#             'objective': ''
#         }

#         # Construct all prompts
#         prompts, _, data_to_return = self.construct_prompts_with_retrieval(
#             topics=topics,
#             atomic_facts=atomic_facts,
#             model_name='claude-2.0',
#             dataset=dataset,
#             verbose=False
#         )

#         marks = []
#         raw_results = []

#         # For each prompt, call your ChatGPT/Claude model
#         for prompt in prompts:
#             # Example: see if your eval_model is a ChatCompletion or Claude
#             response_content = self.eval_model.generate_given_prompt([
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user",   "content": prompt}
#             ])['generation']

#             # Attempt to parse the classification after "ANSWER:"
#             lower_resp = response_content.lower()
#             if 'answer:' in lower_resp:
#                 # e.g., "ANSWER: True"
#                 label_portion = lower_resp.split('answer:')[1].strip()
#                 # take the first word only
#                 answer_word = label_portion.split()[0].strip().lower()
#                 mark = mark_dict.get(answer_word, answer_word)
#             else:
#                 # fallback if not found
#                 mark = ''

#             marks.append(mark)
#             raw_results.append({'return': response_content, 'prompt': prompt})

#         return marks, raw_results

#     def revise_fact(self, response, atomic_fact):
#         """
#         Uses the refinement prompt to remove vague references from `atomic_fact`.
#         """
#         full_prompt = _REVISE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
#         full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
#         full_prompt = strip_string(full_prompt)

#         try:
#             model_response = self.relevance_lm.generate(full_prompt)
#             if isinstance(model_response, tuple):
#                 model_response = model_response[0]
#             revised_fact = extract_first_code_block(model_response, ignore_language=True)
#             if revised_fact:
#                 return revised_fact.strip()
#             else:
#                 return atomic_fact  # fallback
#         except Exception as e:
#             logging.error(f"Error refining fact: {e}")
#             return atomic_fact

#     def check_relevance(self, prompt, response, atomic_fact):
#         """
#         Use the relevance prompt to determine if the atomic_fact is relevant 
#         to the question (prompt) in the context of response.
#         """
#         full_prompt = _RELEVANCE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
#         full_prompt = full_prompt.replace(_PROMPT_PLACEHOLDER, prompt)
#         full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
#         full_prompt = strip_string(full_prompt)

#         try:
#             model_response = self.relevance_lm.generate(full_prompt)[0]
#             answer = extract_first_square_brackets(model_response)
#             if answer:
#                 return answer.strip().lower() == SYMBOL.lower()
#             else:
#                 return True  # fallback
#         except Exception as e:
#             logging.error(f"Error checking relevance: {e}")
#             return True  # fallback

# ###############################################################################
# # Additional utility classes from second script
# ###############################################################################
# class General_Wiki_Eval:
#     """ Retained if needed in your code. """
#     # ...
#     pass

# def remove_non_alphanumeric(s):
#     """Helper function for post-processing or annotation matching."""
#     import re
#     return re.sub(r'[^A-Za-z0-9]+', '', s.lower())

# ###############################################################################
# # End of factscore_utils.py
# ###############################################################################
