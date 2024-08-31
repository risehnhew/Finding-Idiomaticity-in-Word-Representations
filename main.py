import csv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise         import paired_cosine_distances
import os
import re
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
import flair
import torch
from tqdm import tqdm
# import openai
import warnings


# from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import AutoTokenizer, AutoModel
from for_probes import select_most_sim, sim_scores, In_context_eval, load_csv
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from flair.embeddings import WordEmbeddings
from flair.embeddings import ELMoEmbeddings




device = "cuda" if torch.cuda.is_available() else "cpu"

def tokenise_idiom( phrase ) :
  return re.sub( r'[\s|-]', '_', phrase ).lower() 
def extract_nc(sent, tag):
  NC = []
  # breakpoint()
  tags = tag.strip('[]').split(', ')
  # breakpoint()
  start_index = tags.index('True')
  for word, sign in zip(sent.split(), tags):
    if sign == 'True':
      NC.append(word)
  return (' '.join(NC), start_index)



def remove_tags(string):
  return string.replace("<b>", "").replace("</b>", "").replace("<strong>", "").replace("</strong>", "").replace("[", "").replace("]", "")



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
def get_similarities2( ori_sentences, head_sens,model, tokenizer, probe=0) : 

  all_embeddings_head = []
  all_embeddings_ori = []
  with torch.no_grad():
    for sen1, sen2 in tqdm(zip(ori_sentences, head_sens)):
      input_ori=tokenizer(sen1, padding=True, return_tensors='pt').to(device)
      input_head = tokenizer(sen2, padding=True, return_tensors='pt').to(device)
    
      model_output_ori = model(**input_ori)
      model_output_head = model(**input_head)
      # Compute cosine-similarits
      sentence_embeddings_ori = mean_pooling(model_output_ori, input_ori['attention_mask'])
      sentence_embeddings_head = mean_pooling(model_output_head, input_head['attention_mask'])
      # breakpoint()
      all_embeddings_head.append(sentence_embeddings_ori.squeeze(0).to('cpu'))
      all_embeddings_ori.append(sentence_embeddings_head.squeeze(0).to('cpu')) 
  if probe == 6:
    return  all_embeddings_ori
  else:
  # breakpoint()
    cosine_scores_head = 1 - (paired_cosine_distances(all_embeddings_head, all_embeddings_ori))

    return cosine_scores_head.tolist()

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



def get_openai_score(sym_sents, ori_sentences):
  sim_scores_all = []
  for time in tqdm(range(1)): 
    sent_mwes_embeddings = []
    new_dao_embeddings = []
    for mwe in sym_sents:
      mwe_embeddings = get_embedding(mwe, model="text-embedding-ada-002")
      sent_mwes_embeddings.append(torch.tensor(mwe_embeddings))
    for new_dao in ori_sentences:
      new_dao_embedding = get_embedding(new_dao, model="text-embedding-ada-002")
      new_dao_embeddings.append(torch.tensor(new_dao_embedding))

    sim_scores_all.append(torch.tensor(1 - (paired_cosine_distances(sent_mwes_embeddings, new_dao_embeddings))))
  sim_scores_all_new= torch.stack(sim_scores_all, dim=0)
  sim_scores_mean = torch.mean(sim_scores_all_new, dim=0).tolist()
  return sim_scores_mean

def get_sims_file(address, sent_type):
  header, data = load_csv(address)
  index = header.index(sent_type)  
  similairities = []
  for elem in data:
    similairities = elem[index]
  return similairities

def Sim_R(sim1s4, sim2s4, probe, s_p):
  all_scores = []
    
  for sim1ss, sim2ss in zip(sim1s4, sim2s4):

    downs = [1- simr for simr in sim2ss]
    results = []
    for i , (sim1, sim_r1, down) in enumerate(zip(sim1ss, sim2ss, downs)):
      if down == 0:
        print(i)
      results.append((sim1 - sim_r1)/down) 


    all_scores.append(results)
  average_natural = [(a + b + c) / 3 for a, b, c in zip(all_scores[0], all_scores[1], all_scores[2])]
  neutral_score = all_scores[3]
  return average_natural, neutral_score


def write_scores(file_name, all_score, probe):
  
  with open(file_name, 'a', newline='', encoding='UTF-16') as f1:
    writer = csv.writer(f1)
    writer.writerow(['file','Cosine Similarity', 'Type', 'Model_name', 'mode','experiment', 'compound'])

    ex = 'P' + str(probe)
    # breakpoint()
    for natural1,natural2,natural3,neutral_score,compound,language,level,model_name,addresses in zip(all_score['natural1'], all_score['natural2'],all_score['natural3'],all_score['neutral_score'],all_score['compounds'],all_score['language'],all_score['level'],all_score['model_name'],all_score['addresses']):
      # breakpoint()
      for i, score1 in enumerate(natural1):

        writer.writerow([addresses[0],score1, language+'-Nat', model_name, level, ex, compound[0][i]])
      for i, score2 in enumerate(natural2):
        writer.writerow([addresses[1],score2, language+'-Nat', model_name, level, ex, compound[1][i]])

      for i, score3 in enumerate(natural3):
        writer.writerow([addresses[2],score3, language+'-Nat', model_name, level, ex, compound[2][i]])
      for i, score4 in enumerate(neutral_score):
        writer.writerow([addresses[3],score4, language+'-Neut', model_name, level, ex, compound[3][i]])

def write_scores2(file_name, model_name2, average_natural, neutral_score, language, affinity=None, level=None, metric=None):
  with open(file_name, 'a', newline='') as f1:
    for model_name1, average_natural1, neutral_score1, language1, affinity1, level1, metric1 in zip(model_name2, average_natural, neutral_score, language, affinity, level, metric):
      if language1 == 'pt':
        sign = 'PT'
      elif language1 == 'en':
        sign = 'EN'      
        writer = csv.writer(f1)
        if affinity1 != None:
          writer.writerow(['Affinity Score', 'Type', 'Model_name', 'mode', 'metric'])
        else:
          writer.writerow(['Cosine Similarity', 'Type', 'Model_name', 'mode', 'metric'])       
        for score1 in average_natural1:
          writer.writerow([score1, sign+'-Nat', model_name1, level1, metric1])
        for score2 in neutral_score1:
          writer.writerow([score2, sign+'-Neut', model_name1, level1, metric1])


if __name__ == '__main__':


  device = "cuda" if torch.cuda.is_available() else "cpu"

  s_ps = [3]
  probes = [2] #1,2,3,4,5
  languages = ['PT','EN'] #,'EN'


#   |    Model   |               English              |             Portuguese             |
# |:----------:|:----------------------------------:|:----------------------------------:|
# | GloVe      | models/glove.840B.300d             | models/glove_pt_s300               |
# | ELMo       | small                              | pt                                 |
# | mSBERT       | distiluse-base-multilingual-cased       | distiluse-base-multilingual-cased       |
# | BERT | google-bert/bert-large-uncased | bert-large-portuguese-cased |neuralmind/bert-large-portuguese-cased
# | mBERT       | google-bert/bert-base-multilingual-cased       | google-bert/bert-base-multilingual-cased       |

# | mDistilBERT | distilbert-base-multilingual-cased | distilbert-base-multilingual-cased |

   # 'distilbert/distilbert-base-multilingual-cased'
  # model_adds = [ 'Word2Vec','GloVe'] #'Word2Vec',
  model_names = ['SBERT ML', 'BERT', 'BERT ML', 'DistilBERT ML', 'LLama2'] #'SBERT ML', 'BERT', 'BERT ML', 'DistilBERT ML', 'LLama2', 'Word2Vec', 'GloVe'['ELMo']
  # model_adds = ['/mnt/fastdata/ac1whe/evaluation_scripts/llama_models/LLama2'] #'Word2Vec','sentence-transformers/distiluse-base-multilingual-cased','google-bert/bert-large-uncased','google-bert/bert-base-multilingual-cased',
  model_adds = ['sentence-transformers/distiluse-base-multilingual-cased','google-bert/bert-large-uncased','google-bert/bert-base-multilingual-cased','distilbert/distilbert-base-multilingual-cased','/mnt/fastdata/ac1whe/evaluation_scripts/llama_models/LLama2'] #,'Word2Vec', 'GloVe'  

  sim_rr = False
  levels = ['NC','Sent'] # ,'NC',

  aff_indicator = False
  affinity_pairs = [[1,3]] # A1, A2, A3
  # breakpoint()
  for language in languages:
    if language == 'PT':
      addresses = ['PT/naturalistics_examplesent1.csv','PT/naturalistics_examplesent2.csv','PT/naturalistics_examplesent3.csv', 'PT/neutral.csv']
      model_adds = ['sentence-transformers/distiluse-base-multilingual-cased','neuralmind/bert-large-portuguese-cased','google-bert/bert-base-multilingual-cased','distilbert/distilbert-base-multilingual-cased', '/mnt/parscratch/users/ac1whe/llama/llama-2-7b/hf_7B'] #'Word2Vec','sentence-transformers/distiluse-base-multilingual-cased','google-bert/bert-large-uncased','google-bert/bert-base-multilingual-cased', 

    elif language == 'EN':
      addresses = [ 'EN/naturalistics_examplesent1.csv', 'EN/naturalistics_examplesent2.csv','EN/naturalistics_examplesent3.csv','EN/neutral.csv']   
      # model_adds = ['ELMo']
      # model_adds = ['Word2Vec', 'GloVe']
    for model_name, model_add in zip(model_names, model_adds):
      print(model_add)
      for level in levels:
        # with torch.cuda.device("cuda:0"):
        if model_name =='Word2Vec': 
          # filename = './other_models/freebase-vectors-skipgram1000.bin' 
          filename = './other_models/GoogleNews-vectors-negative300.bin' 
          model = KeyedVectors.load_word2vec_format(filename, binary=True) if language =='EN' else KeyedVectors.load_word2vec_format('./other_models/PT/skip_s1000.txt', binary=False)
          embedding = model
          # breakpoint()
        elif model_name == 'GloVe':  
          model = KeyedVectors.load_word2vec_format('./other_models/glove.840B.300d.txt', binary=False, no_header=True) if language =='EN' else KeyedVectors.load_word2vec_format('./other_models/PT/glove_s1000.txt', binary=False)
          embedding = model
        elif model_name == 'ELMo':

          embedding = ELMoEmbeddings('small') if language =='EN' else ELMoEmbeddings('pt')
          model = embedding
        NC = level=='NC'
        if model_name not in ('Word2Vec','GloVe', 'ELMo'):
          if level == 'NC':
            # NC = True
            embedding = TransformerWordEmbeddings(model_add,torch_dtype=torch.float16, use_safetensors=True).to(device)
            if model_name=='OpenAI':
              continue
          elif model_name == 'OpenAI':
            # if probe==6:
            #   continue

            NC = False
            tokenizer = None
            embedding = None
          else:

            embedding = TransformerDocumentEmbeddings(model_add, torch_dtype=torch.float16, use_safetensors=True).to(device)
            NC = False 
            # breakpoint()
          

          if 'llama' in model_add:
            embedding.tokenizer.pad_token = embedding.tokenizer.eos_token
          # model = embedding.model
          tokenizer = embedding.tokenizer
        else:
          tokenizer = []

        
          


        for probe in probes:
          all_score={'natural1':[], 'natural2':[],'natural3':[],'compounds':[],'neutral_score':[], 'language':[], 'level':[], 'model_name':[],'addresses':[] }
          

          new_fname = 'Portuguese_results/results823/'+model_name+str(probe)+'_4layer.csv'

        
          for s_p in s_ps:
            # model_names2, average_natural_affs, neutral_score_affs, aff_indicators, levels, metrics=[],[],[],[],[],[]
  
            if sim_rr == True:
              sim1s = openai_sim_scores(model_add, addresses, model, tokenizer, model_name, s_p, embedding=embedding, NC_embedding=NC, sim_rr = sim_rr, language = language)
              sim2s = openai_sim_scores(model_add, addresses, model, tokenizer, model_name, probe, embedding=embedding, NC_embedding=NC, sim_rr = sim_rr, language = language)
              average_natural, neutral_score =Sim_R(sim1s, sim2s, probe, s_p) 
            elif aff_indicator: # calculate affinity score             
              for i, (a1, a2) in enumerate(affinity_pairs):
                aff1s = openai_sim_scores(model_add, addresses, model, tokenizer, model_name, a1, embedding=embedding, NC_embedding=NC, sim_rr = True, language = language)
                affinity_scores = []
                for a11, a22 in zip(aff1s, aff2s):
                  affinity_scores.append([x - y for x, y in zip(a11, a22)])
                average_natural_aff = [(a + b + c) / 3 for a, b, c in zip(affinity_scores[0], affinity_scores[1], affinity_scores[2])]
                neutral_score_aff = affinity_scores[3]
                met = 'A'+ str(i+1)
                # breakpoint()
                model_names.append(model_name)
                average_natural_affs.append(average_natural_aff)
                neutral_score_affs.append(neutral_score_aff)
                languages.append(language)
                aff_indicators.append(aff_indicator)
                levels.append(level)
                metrics.append(met)
            else:
              natural1, natural2, natural3, neutral_score, compounds = sim_scores(addresses, tokenizer, model_name, probe, embedding=embedding, NC_embedding=NC, sim_rr = sim_rr, language = language, single_token=False)
              # # all_score['average_natural'].append(average_natural)

              all_score['natural1'].append(natural1)
              all_score['natural2'].append(natural2)
              all_score['natural3'].append(natural3)
              all_score['compounds'].append(compounds)
              all_score['neutral_score'].append(neutral_score)
              all_score['language'].append(language)
              all_score['level'].append(level)
              all_score['model_name'].append(model_name)
              all_score['addresses'].append(addresses)
        
          write_scores(new_fname, all_score, probe)   
    # write_scores2('Affinity.csv', model_names, average_natural_affs, neutral_score_affs, languages, aff_indicators, levels, metrics)
