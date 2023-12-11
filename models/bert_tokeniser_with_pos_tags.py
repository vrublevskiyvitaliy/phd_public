import nltk

from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class TokeniserWithPosTags:

  def __init__(self, tokeniser, padding, truncation, max_length, train_dataset=None, ):
    self._tokeniser = tokeniser
    self._padding = padding
    self._truncation = truncation
    self._max_length = max_length

    if train_dataset is not None:
      self.pos_tag_map = TokeniserWithPosTags.build_pos_tag_map(tokeniser, train_dataset)
    else:
      self.pos_tag_map = {}
      self.pos_tag_map['value_to_idx'] = TokeniserWithPosTags.load_pos_tag_value_to_idx()
      self.pos_tag_map['idx_to_value'] = {v: k for k, v in self.pos_tag_map['value_to_idx'].items()}


  def tokenise_one_sentance(self, s, verbose = False):
      subword_tokens = self._tokeniser.tokenize(s)
      pos_tag_tokens = []
      current_word = ""
      parts = 0
      # Add [" "] to do processing in one loop fully
      for subword_token in subword_tokens + ["dummy"]:
        if subword_token.startswith("##"):
          # continue with previous word
          current_word += subword_token[2:]
          parts += 1
        else:
          # it is a new word, we need to process accumulated one:
          pos_tag = nltk.pos_tag([current_word])[0][1]
          pos_tag_tokens.extend([pos_tag]*parts)
          current_word = subword_token
          parts = 1
      input_ids = self._tokeniser.convert_tokens_to_ids(subword_tokens)
      if verbose:
        for x, y in zip(subword_tokens, pos_tag_tokens):
          print(x, y)

      return {
          'input_ids': input_ids,
          'pos_tag_tokens': pos_tag_tokens,
      }

  def pos_tag_tokens_to_ids(self, pos_tokens):
     unknown_pos_tag_id = self.pos_tag_map['value_to_idx'].get('UNKNOWN')
     return [self.pos_tag_map['value_to_idx'].get(pos_tag, unknown_pos_tag_id) for pos_tag in pos_tokens]

  def internal_tokenise_two_sentances(self, s1, s2):
    s1_tokens = self.tokenise_one_sentance(s1)
    s2_tokens = self.tokenise_one_sentance(s2)

    return {
        's1_input_ids' : s1_tokens['input_ids'],
        's2_input_ids' : s2_tokens['input_ids'],
        's1_pos_tag_tokens' : s1_tokens['pos_tag_tokens'],
        's2_pos_tag_tokens' : s2_tokens['pos_tag_tokens'],
        's1_pos_tag_ids' : self.pos_tag_tokens_to_ids(s1_tokens['pos_tag_tokens']),
        's2_pos_tag_ids' : self.pos_tag_tokens_to_ids(s2_tokens['pos_tag_tokens']),
    }


  def tokenise_two_sentances(self, s1, s2):
    token_data = self.internal_tokenise_two_sentances(s1, s2)

    def _prepare_for_model(tokeniser, s1, s2):
      return tokeniser.prepare_for_model(
          s1,
          s2,
          padding=self._padding,
          truncation=self._truncation,
          max_length=self._max_length,
      )['input_ids']

    s1_s2_input_ids = _prepare_for_model(self._tokeniser, token_data['s1_input_ids'], token_data['s2_input_ids'])
    s1_s2_pos_tag_ids = _prepare_for_model(self._tokeniser, token_data['s1_pos_tag_ids'], token_data['s2_pos_tag_ids'])
    s1_s2_pos_tag_tokens = _prepare_for_model(self._tokeniser, token_data['s1_pos_tag_tokens'], token_data['s2_pos_tag_tokens'])

    cls_sep_pos_tag_id = self.pos_tag_map['value_to_idx'].get('NA')
    s1_s2_pos_tag_ids = [v if v not in [self._tokeniser.cls_token_id, self._tokeniser.sep_token_id] else cls_sep_pos_tag_id for v in s1_s2_pos_tag_ids]

    return {
        'input_ids': s1_s2_input_ids,
        'pos_tag_tokens': s1_s2_pos_tag_tokens,
        'pos_tag_ids': s1_s2_pos_tag_ids,
    }

  @staticmethod
  def load_pos_tag_value_to_idx():
  # Computed based on test part of MRPC dataset.
    return {
        '$': 1,
        "''": 2,
        '(': 3,
        ')': 4,
        ',': 5,
        '.': 6,
        ':': 7,
        'CC': 8,
        'CD': 9,
        'DT': 10,
        'EX': 11,
        'FW': 12,
        'IN': 13,
        'JJ': 14,
        'JJR': 15,
        'JJS': 16,
        'LS': 17,
        'MD': 18,
        'NA': 19,
        'NN': 20,
        'NNP': 21,
        'NNPS': 22,
        'NNS': 23,
        'PRP': 24,
        'PRP$': 25,
        'RB': 26,
        'RBR': 27,
        'SYM': 28,
        'TO': 29,
        'UH': 30,
        'UNKNOWN': 31,
        'VB': 32,
        'VBD': 33,
        'VBG': 34,
        'VBN': 35,
        'VBP': 36,
        'VBZ': 37,
        'WDT': 38,
        'WP': 39,
        'WP$': 40,
        'WRB': 41,
        '``': 42
      }

  @staticmethod
  def build_pos_tag_map(tokenizer, dataset):
    pos_tokeniser = TokeniserWithPosTags(tokeniser=tokenizer)
    pos_tag_set = set(['NA', 'UNKNOWN'])
    for t in dataset:
      s1 = pos_tokeniser.tokenise_one_sentance(t['sentence1'])
      s2 = pos_tokeniser.tokenise_one_sentance(t['sentence2'])
      pos_tag_set.update(s1['pos_tag_tokens'])
      pos_tag_set.update(s2['pos_tag_tokens'])
    pos_tag_list = sorted(list(pos_tag_set))
    pos_tag_value_to_idx = {pos_tag_list[i]:i + 1 for i in range(len(pos_tag_list))}
    pos_tag_idx_to_value = {v: k for k, v in pos_tag_value_to_idx.items()}
    return {
        'value_to_idx': pos_tag_value_to_idx,
        'idx_to_value': pos_tag_idx_to_value,
    }

def preprocess_dataset_with_pos_tags(examples, tokenizer, truncation, max_length, padding):
  basic_tokenizer_data = tokenizer(examples["sentence1"], examples["sentence2"], truncation=truncation, max_length=max_length, padding=padding)
  pos_tokeniser = TokeniserWithPosTags(tokeniser=tokenizer, truncation=truncation, max_length=max_length, padding=padding)
  pos_tag_tokeniser_data = pos_tokeniser.tokenise_two_sentances(examples["sentence1"], examples["sentence2"])
  assert(basic_tokenizer_data['input_ids'] == pos_tag_tokeniser_data['input_ids'])
  basic_tokenizer_data['pos_tag_ids'] = pos_tag_tokeniser_data['pos_tag_ids']
  return basic_tokenizer_data