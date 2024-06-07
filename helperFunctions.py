#Helper functions
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import spacy
from nltk import Tree
import pandas as pd
import matplotlib.pyplot as plt
import re, requests
import rdflib

class GraphDB:
    def __init__(self):
        log = open('.graphdb/logs/main.log').read()
        pat = 'Started GraphDB in workbench mode at port (\d+)'
        self.port = int(re.findall(pat, log)[-1])
        try:
            requests.head(f'http://localhost:{self.port}')
            print(f'GraphDB running on port {self.port}.')
        except:
            print('GraphDB not running?')

    def create_repo(self, name):
        config = open('.graphdb/repo-config-template.ttl').read().format(name=name)
        return requests.post(
            f'http://localhost:{self.port}/rest/repositories', 
            files={"config": config})

    def load_data(self, repo_name, *filenames, graph_name = None):
        settings = {}
        if graph_name:
            settings = {
                "context": graph_name,
                "replaceGraphs": [graph_name],
            }
        return requests.post(
            f"http://localhost:{self.port}/rest/repositories/{repo_name}/import/server",
            json={"fileNames": filenames,
                  "importSettings": settings}
        )

def obj_to_triples(obj):
    assert isinstance(obj, dict)
    s = obj.pop('@id') if '@id' in obj else rdflib.BNode()
    for k,v in obj.items():
        vs = vs if isinstance(v, list) else [v]
        for v in vs:
            if isinstance(v, dict):
                yield s, k, v.setdefault('@id', rdflib.BNode())
                yield from obj_to_triples(v)
            elif v:
                yield s, k, v

nlp = spacy.load('en_core_web_sm')

def preprocess_dataset(df):
    # action string is cutoff at 51 so filter it out.
    df = df.copy()
    filtered_df = df[df['ACTION'].str.len() <= 50]
    return filtered_df

def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    # converts the words in word_tokens to lower case and then checks whether 
    #they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    #with no lower case conversion
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(f"{node.orth_} ({node.dep_}, {node.pos_})", 
                    [to_nltk_tree(child) for child in node.children])
    else:
        return f"{node.orth_} ({node.dep_}, {node.pos_})"

def print_nltk_tree(doc):
    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

def stem_sentence(sentence, stemmer):
    word_tokens = word_tokenize(sentence)
    return [stemmer.stem(token) for token in word_tokens]
    

def get_word_freq(df):
    root_word_freq = {}
    nsubj_word_freq = {}
    verb_word_freq = {}
    for doc_text in df:
        doc = nlp(doc_text.lower())
        for sent in doc.sents:
            root = sent.root
            root_word = root.text.lower()  # Convert to lowercase for case-insensitive counting
            root_word_freq[root_word] = root_word_freq.get(root_word, 0) + 1
        nsubjs = [tok.lemma_ for tok in doc if (tok.dep_ == "nsubj") ]
        for nsubj in nsubjs:
            nsubj_word_freq[nsubj] = nsubj_word_freq.get(nsubj, 0) + 1
            
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        for verb in verbs:
            verb_word_freq[verb] = verb_word_freq.get(verb, 0) + 1
    return root_word_freq, nsubj_word_freq, verb_word_freq

def displayGraph(word_list, title):
    # Convert the dictionary to a Pandas DataFrame for easy plotting
    df_freq = pd.DataFrame(list(word_list.items()), columns=[title, 'Frequency'])
    
    # Sort the DataFrame by frequency in descending order
    df_freq = df_freq.sort_values(by='Frequency', ascending=False)[0:50]
    
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df_freq[title], df_freq['Frequency'])
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.title(f'Most used {title}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def display_most_used(df_column):
    print("Most frequent words")
    root_word_freq, nsubj_word_freq, verb_word_freq = get_word_freq(df_column)
#     displayGraph(nsubj_word_freq, "Subject")
    displayGraph(verb_word_freq, "Verb")
    
    
def remove_stopwords(sentence):
    return ' '.join([word for word in sentence.split() if word not in stopwords.words('english')])

