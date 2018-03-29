
punctuation = set(string.punctuation)
stop = list(stopwords.words('english')) + list(punctuation) + list(['rt', 'via'])
now = str(datetime.now())

regex_str = r'(?:@[\w_]+)|(?:\#+[\w_]+[\w\'_\-]*[\w_]+)|\w+\'\w+|\w+-\w+|\w+_\w+|\w+\*+\w+|\w+|(?:(?:\d+,?)+(?:\.?\d+)?)'
v = 0
"""
    r'((www\.[^\s]+)|(https?://[^\s]+))',
    r'@[^\s]+',  # @-mentions
    r"#(\w+)",
    r'(?:[\w_]+)',  # other words
    # r'(?:\S)' # anything else
"""

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def without_tokenization():
    return CountVectorizer(lowercase=True, tokenizer=None, max_df=0.9, min_df=0.001, ngram_range=(1, 3))


def with_tokenization():
    return CountVectorizer(lowercase=True, tokenizer=preprocess, max_df=0.9, min_df=0.001, ngram_range=(1, 3))


def is_emoji(s):
    emojis = []
    for emoji in UNICODE_EMOJI:
        em = s.count(emoji)
        emj = []
        for i in range(em):
            emj.append(emoji)
        emojis = emojis + emj
    # print(emojis)
    return emojis


def tokenizer(s):
    tokens = s = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', s)
    tokens = re.findall(regex_str, s)
    emojis = is_emoji(s)
    return tokens + emojis


def get_tags(s):
    s = s[2:-1]
    s = s.split(', #')
    if (s != ''):
        # hashtag = r'/text=>(\\"+([\w])+\\"+)/'
        hashtag = r'text=>\"([\w]+)\"'
        tags = [''.join(map(str, re.findall(hashtag, t))) for t in s]
        return tags
    else:
        return []


def get_source(s):
    # print(s)
    return re.findall(r'<a[^>]*>\s*((?:.|\n)*?)</a>', s)[0] if (
        len(re.findall(r'<a[^>]*>\s*((?:.|\n)*?)</a>', s)) > 0) else ''
    # print(tags)


def get_number_tags(s):
    hashtags = dataset.iloc[:, 10]
    # spam_related = get_spam(tweets)
    str_h = [get_tags(s) if len(s) != 2 else [] for s in hashtags]
    number_hash = [len(t) for t in str_h]
    number_hash = pd.DataFrame(np.reshape(np.array(number_hash), (len(np.array(number_hash)), 1)),
                               columns=['number_hashtag'])
    return number_hash


def preprocess(s, lowercase=True):
    s = re.sub('[\s]+', ' ', s)
    tokens = tokenizer(s)
    if lowercase:
        tokens = [token if len(is_emoji(token)) > 0 else token.lower() for token in tokens]
    one = []
    for i in tokens:
        if len(i) > 1 or (len(is_emoji(i)) != 0):
            if i not in stop:
                # i.strip('\/,!?;:_')
                # re.sub(r':', '', i)
                # i = lemmatize.lemmatize(stemmer.stem(i.lower()), pos="v")
                # print("word ",i)
                one.append(i)
    return one


def get_popular(t, sorted_tags):
    sorted_dic = dict((y, t) for y, t in sorted_tags)
    d = [{s: sorted_dic.get(s)} for s in t]
    dic = dict(ChainMap(*d))
    sorted_dic = sorted(dic.items(), key=operator.itemgetter(1))[::-1]
    return sorted_dic[0][0]


def hashtag_features(dataset):
    hashtags = dataset.iloc[:, 10]
    str_h = [get_tags(s) if len(s) != 2 else [] for s in hashtags]
    a = np.array(sum(str_h, []))
    unique, counts = np.unique(a, return_counts=True)
    sorted_tags = sorted(dict(zip(unique, counts)).items(), key=operator.itemgetter(1))

    hashs = [get_popular(s, sorted_tags) if len(s) > 1 else s[0] if len(s) == 1 else '' for s in str_h]
    tweet_hashs = np.array(hashs)
    number_hash = [len(t) for t in str_h]
    number_hash = pd.DataFrame(np.reshape(np.array(number_hash), (len(np.array(number_hash)), 1)),
                               columns=['number_hashtag'])
    tweet_hashs = pd.DataFrame(np.reshape(tweet_hashs, (len(tweet_hashs), 1)), columns=['hash_tag'])
    return number_hash, tweet_hashs
