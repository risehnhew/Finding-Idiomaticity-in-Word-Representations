import csv
import random
import numpy as np
from sklearn.metrics.pairwise  import paired_cosine_distances
import torch
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_csv( path ): 
  header = None
  data   = list()
  with open(path, encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile ) 
    for row in reader : 
      if header is None : 
        header = row
        continue
      data.append( row ) 
  return header, data

def insert_words(sentence, tag, compound):
  """
  This function inserts two words into a sentence at the positions indicated by True values in the tag.

  Args:
      sentence: The original sentence to modify.
      tag: A list of booleans indicating where to insert words (True for insertion, False otherwise).
      words: A list containing the two words to insert (corresponding to the True positions in the tag).

  Returns:
      The modified sentence with the words inserted.
  """
  # words = words.split()
  word_list = sentence.split()

  # Check if the tag and words list have the correct length
  if len(tag) != len(word_list):
    breakpoint()
    raise ValueError("Tag and words list must have the same length as the number of words in the sentence.")

  
  modified_sentence = ""
  for i, word in enumerate(word_list):
    if tag[i]:
      # breakpoint()
      if i == (len(tag)-1):
        modified_sentence += compound + " "  # Insert the compound
        continue
      elif tag[i+1]:
        continue
      else:
        modified_sentence += compound + " "  # Insert the compound
        
    else:
      modified_sentence += word + " "
  return modified_sentence.strip()

def extract_nc(sent, tag):
  NC = []

  if isinstance(tag, str):
    tag = eval(tag)
  start_index = tag.index(True)
  for word, sign in zip(sent.split(), tag):
    if sign:
      NC.append(word)
  return (' '.join(NC), start_index)
def filter_out_overlap(sentences1,tags1,sentences2,tags2,compounds):
  sentences12,tags12, sentences22, tags22, compounds2= [],[],[],[],[]

  for sen1, t1, sen2, t2, com in zip(sentences1,tags1,sentences2,tags2,compounds):
    sen12 = set(extract_nc(sen1, t1)[0].split())
    sen22 = set(extract_nc(sen2, t2)[0].split())
    # breakpoint()
    if sen12.isdisjoint(sen22):

      sentences12.append(sen1),tags12.append(t1), sentences22.append(sen2), tags22.append(t2), compounds2.append(com)

  return sentences12,tags12, sentences22, tags22, compounds2
def get_similarities2(language, ori_sentences, head_sens, model_name,model, tokenizer, probe=0, i=None, address= None, tags1=None, tags2=None, embedding=None) : 
  # breakpoint()
  # ori_sentences = ori_sentences
  # head_sens = head_sens
  all_embeddings_head = []
  all_embeddings_ori = []
  origin_sent_file = 'embeddings/filter_out/'+model_name+'/sent/'+language+ '_'+address[-5:]+'_original_sent_embedding.pth'
  if i!=None:
    probe_sent_file = 'embeddings/filter_out/'+model_name+'/sent/'+str(i)+language+'_'+address[-5:]+'_probe'+str(probe)+'_sent_embedding.pth'
  else:
    probe_sent_file = 'embeddings/filter_out/'+model_name+'/sent/'+language+'_'+address[-5:]+'_probe'+str(probe)+'_sent_embedding.pth'
  # breakpoint()
  all_embeddings_ori = get_and_load_em(origin_sent_file, ori_sentences, tags1 = tags1, if_mwe=True, tokenizer=tokenizer, model= model, embedding = embedding, model_name=model_name)
  all_embeddings_head = get_and_load_em(probe_sent_file, head_sens, tags1 = tags2,if_mwe=True, tokenizer=tokenizer, model= model, embedding = embedding, model_name=model_name)

  if probe == 6:
    return  all_embeddings_ori
  else:
    # breakpoint()
    cosine_scores_head = calculating_similarity(all_embeddings_head, all_embeddings_ori) # calculate similairty
    return cosine_scores_head.tolist()
def select_most_sim(language,ori_sentences, head_only_sents, head_only_tags, modifier_only_sents, modifier_only_tags, model, tokenizer, location,model_name='LLama2', nc_tags=None, embedding=None):

  new_sents = []
  new_tags = []

  if model_name =='OpenAI':
    modi_scores = get_OpenAI_score(ori_sentences, modifier_only_sents, 'aa', 'aa')
    head_scores = get_OpenAI_score(ori_sentences, head_only_sents, 'aa', 'aa')
  else:

    modi_scores = get_similarities2(language,ori_sentences, modifier_only_sents, model_name,model, tokenizer,probe=2,address = location, tags1 =nc_tags, tags2=modifier_only_tags, embedding=embedding)
    head_scores = get_similarities2(language,ori_sentences, head_only_sents, model_name,model, tokenizer,probe=2,address = location, tags1 =nc_tags,tags2=head_only_tags, embedding=embedding)

  for modi_score, modifier_sent, header_sent, head_score, modifier_only_tag, head_only_tag in zip(modi_scores, modifier_only_sents, head_only_sents, head_scores, modifier_only_tags, head_only_tags):
    if modi_score > head_score:
      new_sents.append(modifier_sent)
      new_tags.append(modifier_only_tag)
    else:
      new_sents.append(header_sent)
      new_tags.append(head_only_tag)
  return new_sents, new_tags
def combined_probe(location, probe, model=None, tokenizer=None, model_name=None, language=None):
    # Initialize lists based on probe
    ori_sentences, nc_tags = [], []
    
    if probe == 1:
        sym_sents, nc_syn_tags = [], []
    
    elif probe == 2:
        head_only_sents, head_only_tags = [], []
        modifier_only_sents, modifier_only_tags = [], []
    
    elif probe == 3:
        NC_syn_both_sents, nc_both_syn_tags = [], []
    
    elif probe == 4:
        freq_sent, freq_sent_tag, freq_sents, freq_ran_tags = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []]
    
    elif probe == 5:
        random_sents, random_tags = [[], [], [], [], []], [[], [], [], [], []]
    
    elif probe == 6:
        compounds, sym_sents = [], []

    # Load data
    header, data = load_csv(location)

    # Common extraction logic
    for elem in data:
        if 'neutral sentence' in header:
            sentence = elem[header.index('neutral sentence')]
        else:
            sentence = elem[header.index('original sentence')]

        nc_tag = elem[header.index('original sentence_tag')]
        ori_sentences.append(sentence)
        nc_tags.append(nc_tag)

        # Probe-specific logic
        if probe == 1:
            sym_sent = elem[header.index('synonym for compound')]
            NC_syn_tag = elem[header.index('synonym for compound_tag')]
            sym_sents.append(sym_sent)
            nc_syn_tags.append(NC_syn_tag)

        elif probe == 2:
            head_only_sent = elem[header.index('original head only')]
            head_only_tag = elem[header.index('original head only_tag')]
            head_only_sents.append(head_only_sent)
            head_only_tags.append(head_only_tag)

            modifier_only_sent = elem[header.index('original modifier only')]
            modifier_only_tag = elem[header.index('original modifier only_tag')]
            modifier_only_sents.append(modifier_only_sent)
            modifier_only_tags.append(modifier_only_tag)

        elif probe == 3:
            NC_syn_both_sent = elem[header.index('synonym both')]
            NC_both_tag = elem[header.index('synonym both_tag')]
            NC_syn_both_sents.append(NC_syn_both_sent)
            nc_both_syn_tags.append(NC_both_tag)

        elif probe == 4:
            for i in range(5):
                freq_sent[i] = elem[header.index('nc rand freq sentence' + str(i + 1))]
                freq_sent_tag[i] = elem[header.index('nc rand freq sentence' + str(i + 1) + '_tag')]
                freq_sents[i].append(freq_sent[i])
                freq_ran_tags[i].append(freq_sent_tag[i])

        elif probe == 5:
            compounds = [d1[0] for d1 in data]
            replace_list = np.load('replaces.npy')

            for elem, replaces in zip(data, replace_list):
                target_nc = elem[0]
                filtered_compounds = [item for item in compounds if item != target_nc]
                replaces = random.sample(filtered_compounds, 5)

                for i, replace in enumerate(replaces):
                    new_sentence = insert_words(sentence, nc_tag, replace)
                    random_sents[i].append(new_sentence)
                    random_tags[i].append(nc_tag)

        elif probe == 6:
            compound = elem[0]
            sym_sent = elem[header.index('synonym for compound')]
            sym_sents.append(sym_sent)
            compounds.append(compound)

    # Return values based on probe
    if probe == 2:
        new_sent_com, new_tags = select_most_sim(language, ori_sentences, head_only_sents, head_only_tags, modifier_only_sents, modifier_only_tags, model, tokenizer, location, model_name=model_name, nc_tags=nc_tags, embedding=model)
        return ori_sentences, nc_tags, new_sent_com, new_tags

    if probe == 1:
        return ori_sentences, nc_tags, sym_sents, nc_syn_tags
    elif probe == 3:
        return ori_sentences, nc_tags, NC_syn_both_sents, nc_both_syn_tags
    elif probe == 4:
        return ori_sentences, nc_tags, freq_sents, freq_ran_tags
    elif probe == 5:
        return ori_sentences, nc_tags, random_sents, random_tags
    elif probe == 6:
        return ori_sentences, nc_tags, sym_sents, compounds

def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  for delay_secs in (2**x for x in range(0, 6)):
    try:
      embeddings = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
      break
    
    except openai.OpenAIError as e:
      randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
      sleep_dur = delay_secs + randomness_collision_avoidance
      print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
      time.sleep(sleep_dur)
      continue
  return embeddings
def get_OpenAI_score(sym_sents, ori_sentences,embed_name1, embed_name2):
  sim_scores_all = []
  for time in tqdm(range(1)): 
    sent_mwes_embeddings = []
    new_dao_embeddings = []
    if not os.path.exists(embed_name1):
      for mwe in sym_sents:
        mwe_embeddings = get_embedding(mwe, model="text-embedding-ada-002")
        sent_mwes_embeddings.append(torch.tensor(mwe_embeddings))
      # torch.save(sent_mwes_embeddings, embed_name1)
    else:
      sent_mwes_embeddings = torch.load(embed_name1)
    if not os.path.exists(embed_name2):
      for new_dao in ori_sentences:
        new_dao_embedding = get_embedding(new_dao, model="text-embedding-ada-002")
        new_dao_embeddings.append(torch.tensor(new_dao_embedding))
      # torch.save(new_dao_embeddings, embed_name2)
    else:
      new_dao_embeddings = torch.load(embed_name2)
    sim_scores_all.append(calculating_similarity(sent_mwes_embeddings, new_dao_embeddings))
  sim_scores_all_new= torch.stack(sim_scores_all, dim=0)
  sim_scores_mean = torch.mean(sim_scores_all_new, dim=0).tolist()
  return sim_scores_mean


def calculating_similarity(embed1, embed2, layers=-4):
    # list_sims = []
    emb1s, emb2s = [],[]
    for emb1, emb2 in zip(embed1, embed2):
        # breakpoint()
        embed1_4 = [emd.mean(dim=0) for emd in emb1[layers:]]
        embed2_4 = [emd.mean(dim=0) for emd in emb2[layers:]]
        # breakpoint()
        emb1s.append(torch.mean(torch.stack(embed1_4).to('cpu'),0))
        emb2s.append(torch.mean(torch.stack(embed2_4).to('cpu'),0))

    list_sims = 1 - (paired_cosine_distances(emb1s, emb2s))
    # breakpoint()
    return list_sims
def single_token_fun(sents, tags):
  new_sents = []
  for sent, tag in zip(sents, tags):
    NC, _ = extract_nc(sent, tag)
    
    new_mwe = 'ID' + re.sub( r'[\s|-]', '', NC ).lower() + 'ID'
    # breakpoint()
    replaced1 = re.sub( NC, new_mwe, sent, flags=re.I)
    new_sents.append(replaced1)
  return new_sents


# def remove_rogue_dimensions(embeddings, rogue_indices):
#     """
#     Remove rogue dimensions from embeddings.

#     Args:
#     embeddings (torch.Tensor): The embeddings with shape (sent_num, layers, num_token, dim).
#     rogue_indices (list of int): List of indices corresponding to rogue dimensions to be removed.

#     Returns:
#     torch.Tensor: Embeddings with rogue dimensions removed.
#     """
#     # Create an index tensor for all dimensions except rogue ones
#     all_indices = torch.arange(embeddings.size(-1))
#     breakpoint()
#     valid_indices = all_indices[~torch.tensor(rogue_indices).isin(all_indices)]
    
#     # Select valid dimensions
#     # cleaned_embeddings = embeddings.index_select(-1, valid_indices)
#     cleaned_embeddings = torch.stack([layer.index_select(-1, valid_indices) for layer in embeddings], dim=1)
    
#     return cleaned_embeddings
def remove_rogue_dimensions(embeddings, rogue_indices):
    """
    Remove rogue dimensions from embeddings for each layer.

    Args:
    embeddings (torch.Tensor): The embeddings with shape (layers, num_token, dim).
    rogue_indices (list of list of int): A list of lists where each sublist corresponds to rogue dimensions for a specific layer.

    Returns:
    torch.Tensor: Embeddings with rogue dimensions removed for each layer.
    """
    cleaned_layers = []

    # Iterate over each layer
    for i, (layer, rogue_dims) in enumerate(zip(embeddings, rogue_indices)):
        # Create an index tensor for all dimensions except rogue ones for this layer
        all_indices = torch.arange(layer.size(-1))
        # if rogue_dims
        mask = ~torch.isin(all_indices, torch.tensor(rogue_dims))
        valid_indices = all_indices[mask]
        # valid_indices = all_indices[~torch.tensor(rogue_dims).isin(all_indices)]
        
        # Select valid dimensions for this layer
        cleaned_layer = layer.index_select(-1, valid_indices)

        # Append the cleaned layer to the list
        cleaned_layers.append(cleaned_layer)
    # breakpoint()
    # # Stack the cleaned layers back into a single tensor
    # cleaned_embeddings = torch.stack(cleaned_layers, dim=0)
    # breakpoint()


    return cleaned_layers
def get_embeddings_HF(texts, model, tokenizer):
    # Load pre-trained model tokenizer


    # Tokenize input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        # Get hidden states from the model
        outputs = model(**inputs)
    # The last hidden state is the output

    all_hidden_states = outputs.hidden_states
    # breakpoint()
    all_h_ss = [emb.squeeze() for emb in all_hidden_states]
    # all_hidden_states = torch.tensor(all_hidden_states)

    return all_h_ss


def In_context_eval_raw(sentences, embedding, tags, probe=0, model_name=[], if_mwe=False):
    


  mwe_incontext_embeddings = []
  # init embedding
  
  mwe_outcontext_embeddings = []
  mwe_outcontext_embeddings_raw = []
  # create a sentence
  # model_add = result_dict[model_name]

  # tokenizer_hf = AutoTokenizer.from_pretrained(model_add)
  tokenizer_hf = embedding.tokenizer
  # if 'llama' in model_add:
  #   tokenizer_hf.pad_token = tokenizer_hf.eos_token
  # Load pre-trained model
  
  # model_hf = AutoModel.from_pretrained(model_add, output_hidden_states=True).to('cuda')
  model_hf = embedding.model

  start_indexs = []
  mwes = []
  start_indexs = []
  for tag, sent in zip(tags, sentences):
    mwe, start_index = extract_nc(sent, tag)
    mwes.append(mwe)
    start_indexs.append(start_index)
  for i,(sent, mwe, start_index) in tqdm(enumerate(zip(sentences, mwes, start_indexs))):
    
    # words = mwe.split() if len(mwe.split())>1 else mwe
    # # breakpoint()

    # sentence = Sentence(sent) # Creat a sentence

    # # embed words in sentence
    # embedding.embed(sentence)
    
    mwe_incontext_embedding_raw = get_embeddings_HF(sent, tokenizer=tokenizer_hf, model=model_hf) #layers*num_token*dim   

        
    # mwe_incontext_embedding = sentence.embedding.cpu()
    # mwe_incontext_embedding_raw = sentence.embedding.cpu()

    # mwe_incontext_embeddings.append(mwe_incontext_embedding)
    mwe_outcontext_embeddings_raw.append(mwe_incontext_embedding_raw)
  return mwe_outcontext_embeddings_raw, mwe_outcontext_embeddings_raw


def get_phrase_embedding(sentences: str, tags, model, tokenizer, if_mwe):

    phrase_embeddings = []
    for tag, sent in tqdm(zip(tags, sentences)):
      phrase, _ = extract_nc(sent, tag)
      sentence_embedding = get_embeddings_HF(sent, tokenizer=tokenizer, model=model)
      # breakpoint()
      if if_mwe:
        # Encode the sentence
        # breakpoint()

        inputs = tokenizer(sent, return_tensors='pt').to(device)

        # # Get the hidden states from BERT
        # with torch.no_grad():
        #     outputs = model(**inputs)
        # # Extract the token embeddings
        # # token_embeddings = outputs.last_hidden_state.squeeze(0)
        # token_embeddings = outputs.hidden_states
        # get_embeddings_HF(sent, model, tokenizer)
        # mean_p_token_embeddings = [emb.mean(dim=1).squeeze(0) for emb in token_embeddings] # mean pool for the sentence embedding and take all the layers

        # Tokenize the phrase and find its start and end indices in the input sentence
        phrase_tokens = tokenizer.tokenize(phrase)
        phrase_ids = tokenizer.convert_tokens_to_ids(phrase_tokens)

        # Find the starting position of the phrase in the input_ids
        input_ids = inputs['input_ids'].squeeze().tolist()
        start_index = None
        
        for i in range(len(input_ids) - len(phrase_ids) + 1):
            if input_ids[i:i+len(phrase_ids)] == phrase_ids:
                start_index = i
                break
        # breakpoint()
        if start_index is None:
            raise ValueError(f"Phrase '{phrase}' not found in the sentence.")

        # Get the embeddings for the phrase
        end_index = start_index + len(phrase_ids)
        phr_embed=[]
        for token_layer in sentence_embedding:
            phr_embed.append(token_layer[start_index:end_index])
        phr_embed = [token_layer[start_index:end_index] for token_layer in sentence_embedding]
        phrase_embeddings.append(phr_embed)
      else:
        phrase_embeddings.append(sentence_embedding)

    return phrase_embeddings
