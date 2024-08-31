import csv
# from sentence_transformers    import SentenceTransformer,  losses, models
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import ELMoEmbeddings
import pandas as pd

# import openai

# from openai.embeddings_utils import get_embedding, cosine_similarity

from flair.data import Sentence
import flair
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os,re
import random, time
import numpy as np

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
# from allennlp.modules.elmo import Elmo, batch_to_ids
from utils import load_csv,extract_nc, calculating_similarity, filter_out_overlap, single_token_fun, combined_probe
from utils import get_phrase_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_path = './raw_embedding/'


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

def In_context_eval(sentences, embedding, tags, probe=0, model_name=[], if_mwe=False):

  mwe_incontext_embeddings = []
  # init embedding
  # mwe_outcontext_embeddings = []
  mwe_outcontext_embeddings_raw = []
  # create a sentence

  start_indexs = []
  mwes = []
  start_indexs = []
  # breakpoint()
  for tag, sent in zip(tags, sentences):
    mwe, start_index = extract_nc(sent, tag)
    mwes.append(mwe)
    start_indexs.append(start_index)
  
  for i,(sent, mwe, start_index) in tqdm(enumerate(zip(sentences, mwes, start_indexs))):
    
    words = mwe.split() if len(mwe.split())>1 else mwe
    if model_name in ('Word2Vec', 'GloVe'): 
      
      # mwe_embedding = embedding[mwe]
      words = words if if_mwe else sent.split()
      if len(words)>1:
        word_list1 = []
        for word in words:
          if word not in embedding:
            word ='word'
          word_list1.append(embedding[word])
        vectors = word_list1
      else:
        vectors = [embedding[words]]
      if vectors:
        # mwe_incontext_embedding = torch.tensor(np.mean(vectors, axis=0))
        mwe_incontext_embedding_raw = torch.tensor(vectors) # Wei

      else:
        print('not match')
        # mwe_incontext_embedding = 1
        breakpoint()
    else:
      # if model_name == 'ELMo':
      #   character_ids = batch_to_ids(mwe)

      #   mwe_embedding = embedding(character_ids1)
      #   embedding = ELMoEmbeddings()
      sentence = Sentence(sent) # Creat a sentence

      # embed words in sentence
      # breakpoint()
      embedding.embed(sentence)
      # breakpoint()
      if if_mwe:
        list_sent = sent.split()   
        start_indexs.append(start_index)
 
        mwe_len= len(words)

        tokens = sentence[start_index: start_index+mwe_len]
        mwe_embedding = []

        for token in tokens:
          cpu_embedding = token.embedding.cpu()
          mwe_embedding.append(cpu_embedding)
        mwe_incontext_embedding_raw = torch.stack(mwe_embedding)
      else:
        if model_name == 'ELMo':
          tokens = sentence[:]
          mwes_embed = [token.embedding.cpu() for token in tokens]
          mwe_incontext_embedding_raw = sentence.embedding.cpu()
        else:
          mwe_incontext_embedding_raw = sentence.embedding.cpu()
    mwe_outcontext_embeddings_raw.append(mwe_incontext_embedding_raw)
  return mwe_outcontext_embeddings_raw

def get_and_load_em(ori_file, sentences1, if_mwe, embedding=None, tags1=None, tokenizer=None, model= None,  model_name=[]):
  
  # breakpoint()
  path_folder = os.path.dirname(ori_file)
  # raw_file = 'raw_embedding/'+ori_file
  # breakpoint()
  if not os.path.exists(ori_file):
    # embed1_raw = In_context_eval(sentences1, embedding, tags1, model_name=model_name, if_mwe=if_mwe)# original
    # breakpoint()
    embed1_raw = get_phrase_embedding(sentences1, tags1, embedding.model, tokenizer, if_mwe)
    # embed1, embed1_raw = In_context_eval_raw(sentences1, embedding, tags1, model_name=model_name, if_mwe=if_mwe)
    # breakpoint()

    if not os.path.exists(path_folder):
      os.makedirs(path_folder)
    torch.save(embed1_raw, ori_file)
  else:
    embed1_raw = torch.load(ori_file)
  return embed1_raw

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
def sim_scores(addresses, tokenizer, model_name, probe, embedding, NC_embedding, sim_rr, language, single_token):
  all_scores, average_natural, neutral_score = [], [], [] 
  compounds_all = [] 
  for address in addresses:
    result_data = combined_probe(address, probe)
    if probe == 2:
      if model_name == 'OpenAI':
        result_data = combined_probe(address,probe, model = None, tokenizer=None, model_name=model_name, language=language)
      else:
        result_data = combined_probe(address, probe, model = embedding, tokenizer=tokenizer, model_name=model_name, language=language)

    sentences1= result_data[0]
    tags1= result_data[1]
    sentences2= result_data[2]
    tags2= result_data[3]
    compounds = combined_probe(address, 6)[3]
    # breakpoint()
    # try:
    #   sentences1,tags1,sentences2,tags2,compounds = filter_out_overlap(sentences1_o,tags1_o,sentences2_o,tags2_o,compounds_o)
    # except:
    #   breakpoint()
    compounds_all.append(compounds)
    # df_data[address] = compounds
    # address = address.replace('/','_')
    # # breakpoint()
    # df_data.to_csv('filter_'+address,encoding='utf-16')


    # breakpoint()
    if NC_embedding:
        mode_name = 'NC'
    else:
        mode_name = 'Sent'
    if single_token:
      sentences1 = single_token_fun(sentences1, tags1)
    # if model_name == 'OpenAI':    
    #   if probe==4 or probe==5:
    #       sim_5 = []
    #       for i, (sent2, sent1) in enumerate(zip(sentences2, [sentences1]*5)):
    #         file_probe22 = 'embeddings/OpenAI/'+str(i)+language+'_'+'_probe'+ str(probe) +'_'+mode_name+'_embedding.pth'
    #         file_probe11 = 'embeddings/OpenAI/'+language+'_'+'_probe'+ 'ori' +'_NC_embedding.pth'
    #         sim_5.append(get_OpenAI_score(sent2, sent1, file_probe22, file_probe11))
          
    #       similarities = [(a + b + c+d+e) / 5 for a, b, c, d, e in zip(sim_5[0], sim_5[1], sim_5[2], sim_5[3], sim_5[4])]
    #       all_scores.append(similarities)
      # else:
      #   file_probe222 = 'embeddings/OpenAI/'+language+'_'+'_probe'+ str(probe) +'_'+mode_name+'_embedding.pth'
      #   file_probe111 = 'embeddings/OpenAI/'+language+'_'+'_probe'+ 'ori' +'_'+mode_name+'_embedding.pth'
      #   all_scores.append(get_OpenAI_score(sentences2, sentences1, file_probe222, file_probe111))

    else:

      ori_file = embedding_path+model_name+'/'+language+'_'+address[-5:]+'_original_'+mode_name+'_embedding.pth'
      # ori_file = embedding_path+model_name
      # breakpoint()
      embed1 = get_and_load_em(ori_file, sentences1, embedding=embedding, tags1=tags1, if_mwe=NC_embedding, model_name=model_name,tokenizer = tokenizer)
      if probe==4 or probe==5:
        sim_5 = []
        for i, (sent2, tag2) in enumerate(zip(sentences2, tags2)):
          file_probe = 'embeddings/filter_out/'+model_name+'/'+str(i)+language+'_'+address[-5:]+'_probe'+ str(probe) +'_'+mode_name+'_embedding.pth'
          embed2 = get_and_load_em(file_probe, sent2, embedding=embedding, tags1=tag2, if_mwe=NC_embedding, model_name=model_name, tokenizer = tokenizer)   
          similarities = calculating_similarity(embed1,embed2)
          # breakpoint()
          sim_5.append(similarities)
        # breakpoint()
        similarities = [(a + b + c+d+e) / 5 for a, b, c, d, e in zip(sim_5[0], sim_5[1], sim_5[2], sim_5[3], sim_5[4])]
      else:
        file_probe2 = embedding_path+model_name+'/'+language+'_'+address[-5:]+'_probe'+ str(probe) +'_'+mode_name+'_embedding.pth'
        # breakpoint()
        embed2 = get_and_load_em(file_probe2, sentences2, embedding=embedding, tags1=tags2, if_mwe=NC_embedding, model_name=model_name,tokenizer = tokenizer)
        try:
          similarities = calculating_similarity(embed1,embed2)
        except:
          breakpoint()
        # breakpoint()
      all_scores.append(similarities)
  average_natural = [(a + b + c) / 3 for a, b, c in zip(all_scores[0], all_scores[1], all_scores[2])]
  neutral_score = all_scores[3]
  return all_scores[0], all_scores[1], all_scores[2], neutral_score, compounds_all
