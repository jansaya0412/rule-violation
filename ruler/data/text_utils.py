import re

from bs4 import BeautifulSoup
from markdown import markdown


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ' '.join(soup.findAll(string=True))

    return text


def replace_urls(x, url_replacement_token='<URL>'):
    return re.sub("http(.+)?(\W|$)", url_replacement_token, x)
