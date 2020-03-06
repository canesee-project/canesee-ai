import typing
from googletrans import Translator

_translator = Translator()


def translate_eg_ar(sentences: [str]):
    return list(map(lambda res: res.text, _translator.translate(sentences, src='en', dest='ar')))


