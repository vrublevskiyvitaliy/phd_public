import nltk
import spacy
# Space module import
import en_core_web_md

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_spacy_module():
  return en_core_web_md.load()

nlp = get_spacy_module()

def get_sentence_tokens(s):
  doc = nlp(s)
  sentence_tokens = []

  for token in doc:
    data = {
      "token_id": token.i,
      "token_text": token.text,
      "token_connection_ids": token.head.i,
      "token_left_edge": token.idx,
      "token_right_edge": token.idx + len(token.text),
      "token_boundaries": (token.idx, token.idx + len(token.text)),
      "token_pos_tag": token.tag_,
    }

    sentence_tokens.append(data)
  return sentence_tokens


class BaseEnrichedTokeniser:
  def __init__(self, tokeniser):
    self._tokeniser = tokeniser
    self.feature_key = None

  def combine_transformer_and_sentence_features(self, transformer_tokens, sentence_features):
    for i in range(len(transformer_tokens)):
      transformer_tokens[i][self.feature_key] = None
      for j in range(len(sentence_features)):
        t_boundaries = transformer_tokens[i]['boundaries']
        s_boundaries = sentence_features[j]['boundaries']
        left = max(t_boundaries[0], s_boundaries[0])
        right = min(t_boundaries[1], s_boundaries[1])
        if left < right:
          transformer_tokens[i][self.feature_key] = sentence_features[j][self.feature_key]

      assert(transformer_tokens[i][self.feature_key] is not None)

    return transformer_tokens

  def get_feature(self, s):
    raise Exception("Not implemented")

  def enrich_tokens(self, s):
    t_tokens = self.get_transformer_sentence_tokens(s)
    feature = self.get_feature(s)
    combined_features = self.combine_transformer_and_sentence_features(t_tokens, feature)
    return combined_features

  """
  Return list of tokens that will be used in model
  [('boundaries':(int, int), 'input_id':int, 'index':int]
  """
  def get_transformer_sentence_tokens(self, s, verbose = False):
      encoded = self._tokeniser.batch_encode_plus([s], return_offsets_mapping=True, add_special_tokens=False)
      split_tokens = []
      for i in range(len(encoded['input_ids'][0])):
        split_tokens.append(
            {
                'index': i,
                'input_id': encoded['input_ids'][0][i],
                'boundaries': encoded['offset_mapping'][0][i],
            }
        )
      return split_tokens

class PosTagEnrichedTokeniser(BaseEnrichedTokeniser):

  def __init__(self, tokeniser):
    super().__init__(tokeniser)
    self.feature_key = 'pos_tag'

  def get_feature(self, s):
    """
    Return list of [('boundaries':(int, int), 'feature':string)]
    """
    all_tokens = get_sentence_tokens(s)
    return [{'boundaries':token['token_boundaries'], 'pos_tag': token['token_pos_tag']} for token in all_tokens]


class PosTagIdEnrichedTokeniser(BaseEnrichedTokeniser):
  def __init__(self, tokeniser):
    super().__init__(tokeniser)
    self.feature_key = 'pos_tag_ids'

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

  def get_feature(self, s):
    """
    Return list of [('boundaries':(int, int), 'feature':string)]
    """
    all_tokens = get_sentence_tokens(s)
    pos_tag_to_id_map = PosTagIdEnrichedTokeniser.load_pos_tag_value_to_idx()
    def get_id(pos_tag):
      if pos_tag in pos_tag_to_id_map:
        return pos_tag_to_id_map[pos_tag]
      return pos_tag_to_id_map['UNKNOWN']
    return [{'boundaries':token['token_boundaries'], 'pos_tag_id': get_id(token['token_pos_tag'])} for token in all_tokens]

  
class FinalTokeniser:
  def __init__(self, tokeniser):
    self._tokeniser = tokeniser

  def apply_tokenisers(self, s1, s2, tokeniser_list, padding, truncation, max_length):
    # Get base data first: input ids, token_ids, attention_mask
    encoded_base = self._tokeniser.batch_encode_plus([s1, s2], return_offsets_mapping=True, add_special_tokens=False)
    s1_input_ids = encoded_base['input_ids'][0]
    s2_input_ids = encoded_base['input_ids'][1]

    def _prepare_for_model(tokeniser, s1, s2):
      return tokeniser.prepare_for_model(
          s1,
          s2,
          padding=padding,
          truncation=truncation,
          max_length=max_length,
      )['input_ids']

    s1_s2_input_ids = _prepare_for_model(self._tokeniser, s1_input_ids, s2_input_ids)

    data = {
      'input_ids': s1_s2_input_ids,
    }

    for tokeniser in tokeniser_list:
      enriched_data_s1 = tokeniser.enrich_tokens(s1)
      enriched_data_s2 = tokeniser.enrich_tokens(s2)

      _s1_input_ids = [x['input_id'] for x in enriched_data_s1]
      _s2_input_ids = [x['input_id'] for x in enriched_data_s2]

      assert(s1_input_ids == _s1_input_ids)
      assert(s2_input_ids == _s2_input_ids)

      s1_feature = [x[tokeniser.feature_key] for x in enriched_data_s1]
      s2_feature = [x[tokeniser.feature_key] for x in enriched_data_s2]

      s1_s2_feature = _prepare_for_model(self._tokeniser, s1_feature, s2_feature)

      data[tokeniser.feature_key] = s1_s2_feature
    
    return data

  def tokenise_everything(self, s1, s2, padding, truncation, max_length):
    pos_tag_id_tokeniser = PosTagIdEnrichedTokeniser(self._tokeniser)
    
    return self.apply_tokenisers(s1, s2, [pos_tag_id_tokeniser], padding, truncation, max_length)


def preprocess_dataset_final(examples, tokenizer, truncation, max_length, padding):
  basic_tokenizer_data = tokenizer(examples["sentence1"], examples["sentence2"], truncation=truncation, max_length=max_length, padding=padding)
  final_tokeniser = FinalTokeniser(tokeniser=tokenizer)
  enriched_data = final_tokeniser.tokenise_everything(examples["sentence1"], examples["sentence2"], truncation=truncation, max_length=max_length, padding=padding)
  assert(basic_tokenizer_data['input_ids'] == enriched_data['input_ids'])
  for feature, value in enriched_data.items():
    basic_tokenizer_data[feature] = value
  return basic_tokenizer_data