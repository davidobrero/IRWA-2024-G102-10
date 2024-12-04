import json

class Document:
    """
    Original corpus data as an object
    """

    def __init__(self, id, quote,author):
        self.id = id
        self.quote = quote
        self.author = author

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)

class ResultItem:
    def __init__(self, id, quote, author,url,ranking):
        self.id = id
        self.quote = quote
        self.author = author
        self.url = url
        self.ranking = ranking

