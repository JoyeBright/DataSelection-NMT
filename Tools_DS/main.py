import argparse
import torch
torch.cuda.is_available()
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
# logging.basicConfig(level = logging.INFO)
import pickle
import numpy as np
import pandas as pd
import math

if __name__ == '__main__':
    #----------------------------------
    # Takes the arguments from the input
    #----------------------------------
    argParser = argparse.ArgumentParser()
    # The first two arguments are the path to OOD source and target
    argParser.add_argument("-ood_src", "--generic_src", help="the path to source-side generic corpus", required=True)
    argParser.add_argument("-ood_tgt", "--generic_tgt", help="the path to target-side generic corpus", required=True)
    # The third argument is the path to ID
    argParser.add_argument("-id", "--specific", help="the path to domain-specific corpus", required=True)
    # The fourth argument is the desired number of generated data to be selected
    argParser.add_argument("-n", "--number", type=int, help="your desired number of generated data to be selected.", required=False)
    args = argParser.parse_args()
    print("Below are the arguments entered ...")
    print("source-side OOD= %s" % args.generic_src)
    print("target-side OOD= %s" % args.generic_tgt)
    print("ID= %s" % args.specific)
    print("N= %s" % args.number)
    #----------------------------------
    # Embedding OOD
    #----------------------------------
    OOD_src = args.generic_src
    OOD_tgt = args.generic_tgt
    ID = args.specific
    Number = args.number
    #----------------------------------
    with open(OOD_src, 'rb') as e:
        content = e.readlines()
        content = [x.strip() for x in tqdm(content)]
    source = content
    source_new = []
    for sentences in source:
        source_new.append(sentences.decode('utf-8'))
    source = source_new
    print("Source length:",len(source))
    #------------------------------------
    with open(OOD_tgt, 'rb') as f:
        content2 = f.readlines()
        content2 = [x.strip() for x in tqdm(content2)]
    target = content2
    target_new = []
    for sentences in target:
        target_new.append(sentences.decode('utf-8'))   
    target = target_new
    print("Target length:",len(target))
    #-------------------------------------
    OOD_sentences = source
    # Invoke the model
    model = SentenceTransformer('joyebright/stsb-xlm-r-multilingual-32dim', device='cuda')
    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()
    # Compute the embeddings using the multi-process pool
    emb = model.encode_multi_process(OOD_sentences, pool)
    print("Embeddings computed shape:", emb.shape)
    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)
    #-------------------------------------
    print("The OOD is being saved into a file named: 32dim.pkl")
    # Save shuffle for ODD parallel corpora
    with open('32dim.pkl', 'wb') as pickle_file:
        pickle.dump({'source_sentences': source, 'source_embeddings': emb, 'target_sentences': target}, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    #-------------------------------------
    print("A sample of OOD: ", OOD_sentences[0])
    print("A sample of OOD embedding vector: ", emb[0][:])
    #-------------------------------------
    # Loading the pkl file
    with open('32dim.pkl', 'rb') as pickle_load:
        OOD_data = pickle.load(pickle_load)
        OOD_sentences_source = OOD_data['source_sentences']
        OOD_embeddings = OOD_data['source_embeddings']
        OOD_sentences_target = OOD_data['target_sentences']
    #-------------------------------------
    # Stats
    M = len(OOD_sentences_source)
    print("OOD Source length:", M)
    print("OOD Source embedding shape:", OOD_embeddings.shape)
    N = len(OOD_sentences_target)
    print("OOD target length:", N)
    #-------------------------------------
    # Move the OOD embs to the GPU
    OOD_embeddings = torch.tensor(OOD_embeddings, device="cuda")
    #-------------------------------------
    # Read the ID
    with open(ID) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    ID = content
    #---------------Chunks----------------
    def split(list_a, chunk_size):
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i:i + chunk_size]
    #-------------------------------------
    if Number == None:
        Number = len(ID)
        splits_raw = 1
        splits = M
    elif Number > len(ID):
        splits_raw = math.ceil(Number/len(ID))
        splits = math.ceil(M/splits_raw)
        print("The desired number exceeds the number of available ID samples. The tool splits your OOD into ", splits, " to generate more ID sentences.")
    else:
        splits_raw = 1
        splits = M
    #-------------------------------------    
    queries = ID[:Number]
    #-------------------------------------
    OOD_sentences_source = list(split(OOD_sentences_source, splits))
    OOD_sentences_target = list(split(OOD_sentences_target, splits))
    OOD_embeddings = list(split(OOD_embeddings,splits))
    #-------------------------------------
    print("ID length: ", len(queries))
    #-------------------------------------
    for i in range(0, splits_raw):
        print("Split ", i)
        embedder = SentenceTransformer('joyebright/stsb-xlm-r-multilingual-32dim')
        top_k = min(5, len(OOD_sentences_source[i]))
        dat = pd.DataFrame([])
        cols = ['Query','top1', 'top1_trg', 'top1_score',
                        'top2', 'top2_trg', 'top2_score', 
                        'top3', 'top3_trg', 'top3_score', 
                        'top4', 'top4_trg', 'top4_score', 
                        'top5', 'top5_trg', 'top5_score']

        dat = pd.DataFrame(columns = cols)
        index = 0
        for query in queries:  
            print(index)  
            index+=1
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.pytorch_cos_sim(query_embedding, OOD_embeddings[i])[0]
            top_results = torch.topk(cos_scores, k=top_k)
            print(query)
            print(OOD_sentences_source[i][top_results[1][0]])
            print(OOD_sentences_target[i][top_results[1][0]])

            dat = dat.append({'Query': query,
                            
                            'top1':OOD_sentences_source[i][top_results[1][0]],
                            'top1_trg':OOD_sentences_target[i][top_results[1][0]],
                            'top1_score': "(Score: {:.4f})".format(top_results[0][0]),
                            
                            'top2':OOD_sentences_source[i][top_results[1][1]],
                            'top2_trg':OOD_sentences_target[i][top_results[1][1]],
                            'top2_score': "(Score: {:.4f})".format(top_results[0][1]),
                        
                            'top3':OOD_sentences_source[i][top_results[1][2]],
                            'top3_trg':OOD_sentences_target[i][top_results[1][2]],
                            'top3_score': "(Score: {:.4f})".format(top_results[0][2]),
                        
                            'top4':OOD_sentences_source[i][top_results[1][3]],
                            'top4_trg':OOD_sentences_target[i][top_results[1][3]],
                            'top4_score': "(Score: {:.4f})".format(top_results[0][3]),
                        
                            'top5':OOD_sentences_source[i][top_results[1][4]],
                            'top5_trg':OOD_sentences_target[i][top_results[1][4]],
                            'top5_score': "(Score: {:.4f})".format(top_results[0][4])}, ignore_index=True) 
        print("Done...")
        dat.to_csv("final_"+str(i+1)+".csv",index=True)
