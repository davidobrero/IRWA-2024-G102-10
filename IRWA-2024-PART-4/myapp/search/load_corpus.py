import pandas as pd
import json
from myapp.search.objects import Document
from pandas import json_normalize

_corpus = {}

def load_corpus(path) -> [Document]:
    #read all lines
    with open(path,encoding='utf-8') as fp:
        lines = fp.readlines()

    json_string = ''.join(lines)
    # Parse the string into a JSON object
    json_data = json.loads(json_string)
    df = json_normalize(json_data['quotes'])
    df['id'] = df.index.values
    df.apply(_row_to_doc_dict, axis=1)

    return _corpus


def _row_to_doc_dict(row: pd.Series):
    _corpus[row['id']] = Document(row['id'], row['quote'],row['author'])
