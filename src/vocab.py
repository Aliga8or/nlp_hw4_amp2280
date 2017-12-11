from collections import defaultdict


class Vocab:
    def __init__(self, data_path):
        '''
        sentences = open(data_path, 'r').read().strip().split('\n\n')

        # first read into counts
        word_count, tags = defaultdict(int), set()
        for sentence in sentences:
            lines = sentence.strip().split('\n')
            for line in lines:
                word, tag = line.strip().split('\t')
                word_count[word] += 1
                tags.add(tag)
        tags = list(tags)
        words = [word for word in word_count.keys() if word_count[word] > 1]

        # Including unknown as the first features. <s> and </s> are for start and end of sentence.
        self.words = ['<UNK>', '<s>', '</s>'] + words
        self.word_dict = {word: i for i, word in enumerate(self.words)}

        self.output_tags = tags
        self.output_tag_dict = {tag: i for i, tag in enumerate(self.output_tags)}

        # <s> is for start and end of sentence. This is because we are only using the previous tag. In some cases
        # the previous tag can be the start of sentence.
        self.feat_tags = ['<s>'] + tags
        self.feat_tags_dict = {tag: i for i, tag in enumerate(self.feat_tags)}
        '''
        
        word_file = data_path + "vocabs.word"
        tag_file = data_path + "vocabs.pos"
        label_file = data_path + "vocabs.labels"
        action_file = data_path + "vocabs.actions"
        
        self.words= []
        self.tags = []
        self.labels = []
        self.actions= []
        
        #self.word_dict = {}
        #self.tag_dict = {}
        #self.label_dict = {}
        #self.action_dict = {}
        
        with open(word_file, "r") as wptr:
            for line in wptr:
                symbols = line.split()
                #self.word_dict[symbols[0]] = symbols[1].strip("\n")
                self.words.append(symbols[0])
        self.words = ['<s>', '</s>'] + self.words
        self.word_dict = {word: i for i, word in enumerate(self.words)}

        with open(tag_file, "r") as tptr:
            for line in tptr:
                symbols = line.split()
                #self.output_tag_dict[symbols[0]] = symbols[1].strip("\n")
                self.tags.append(symbols[0])
        self.tags = ['<s>'] + self.tags
        self.tag_dict = {tag: i for i, tag in enumerate(self.tags)}
        
        with open(label_file, "r") as lptr:
            for line in lptr:
                symbols = line.split()
                #self.label_dict[symbols[0]] = symbols[1].strip("\n")
                self.labels.append(symbols[0])
        self.label_dict = {label: i for i, label in enumerate(self.labels)}
    
        with open(action_file, "r") as aptr:
            for line in aptr:
                symbols = line.split()
                #self.action_dict[symbols[0]] = symbols[1].strip("\n")
                self.actions.append(symbols[0])
        self.action_dict = {action: i for i, action in enumerate(self.actions)}
    
    '''
    def tagid2tag_str(self, id):
        return self.output_tags[id]
    '''

    def action2id(self, action):
        return self.action_dict[action]

    def feat_tag2id(self, tag):
        return self.tag_dict[tag] if tag in self.tag_dict else self.tag_dict['<null>']

    def feat_label2id(self, label):
        return self.label_dict[label]

    def word2id(self, word):
        return self.word_dict[word] if word in self.word_dict else self.word_dict['<unk>']

    def num_words(self):
        return len(self.words)

    def num_tag_feats(self):
        return len(self.tags)

    def num_label_feats(self):
        return len(self.labels)

    def num_actions(self):
        return len(self.actions)

    def actions(self):
        return self.actions
