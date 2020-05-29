#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Script to load trained model and perform translations


import torch
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor

EN_EMBEDDING_DIM = 256
ZH_EMBEDDING_DIM = 256
HIDDEN_DIM = 256

CUDA_DEVICE = -1

# Loading the reader, vocab, embeddings and model structure
reader = Seq2SeqDatasetReader(
    source_tokenizer=WordTokenizer(),
    target_tokenizer=CharacterTokenizer(),
    source_token_indexers={'tokens': SingleIdTokenIndexer()},
    target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')},
    lazy=True)

vocab = Vocabulary.from_files('/home/earendil/NLP/neural_machine_translation/checkpoint_vocab_epoch_13')

en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                         embedding_dim=EN_EMBEDDING_DIM)

encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128,
                                      feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

attention = DotProductAttention()

max_decoding_steps = 300
model_pred = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                           target_embedding_dim=ZH_EMBEDDING_DIM,
                           target_namespace='target_tokens',
                           attention=attention,
                           beam_size=8,
                           use_bleu=True)

# Reload the trained model.
with open('/home/earendil/NLP/neural_machine_translation/checkpoint_model_epoch_13', 'rb') as f:
    model_pred.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    model_pred.eval()

# Predict on new text using loaded model
predictor = SimpleSeq2SeqPredictor(model_pred, dataset_reader=reader)

import speech_recognition as sr
import os
# Import the required module for text  
# to speech conversion 
from gtts import gTTS

# In[ ]:

# initialize the recognizer
r = sr.Recognizer()

with sr.Microphone() as source:
    # read the audio data from the default microphone
    print("Start speaking...")
    audio_data = r.record(source, duration=5)
    print("Recognizing...")
    # convert speech to text
    text = r.recognize_google(audio_data, language="el-GR", show_all=False)
    print(text)

test = text
# Parsing the output to remove quotes and irrelevant characters

import re

regex = r"', '"
regex2 = r", \"'\", '"
subst = ""
subst2 = ""

p = predictor.predict(test)['predicted_tokens']
result = re.sub(regex, subst, str(p), 0, re.MULTILINE | re.IGNORECASE)
result = re.sub(regex2, subst2, str(result), 0, re.MULTILINE | re.IGNORECASE)

print(result)

# The text that you want to convert to audio
mytext = result

# Language in which you want to convert 
language = 'en'

# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in a mp3 file named 
# welcome  
myobj.save("/home/earendil/Desktop/translated.mp3")

# Playing the converted file 
os.system("mpg321 /home/earendil/Desktop/translated.mp3")

# In[ ]:
