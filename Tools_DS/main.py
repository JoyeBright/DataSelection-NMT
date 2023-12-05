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
import os

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
    argParser.add_argument("-k", "--k", type=int, default = 5, help="your desired number of samples selected per entry.", required=False)
    argParser.add_argument("-n", "--number", type=int, help="your desired number of generated data to be selected.", required=False)
    argParser.add_argument("-dis", "--dissimilar", action="store_true", help="To find similar or dissimilar instances.", required=False)
    argParser.add_argument("-fn", "--filename", type=str, help="the output file name", required=False)
    args = argParser.parse_args()
    print("Below are the arguments entered ...")
    print("source-side OOD= %s" % args.generic_src)
    print("target-side OOD= %s" % args.generic_tgt)
    print("ID= %s" % args.specific)
    print("K= %s" % args.k)
    print("N= %s" % args.number)
    print("Dissimilar= %s" % args.dissimilar)
    print("FileName= %s" % args.filename)
    #----------------------------------
    # Embedding OOD
    #----------------------------------
    OOD_src = args.generic_src
    OOD_tgt = args.generic_tgt
    ID = args.specific
    K = args.k
    Number = args.number
    Dissimilar = args.dissimilar
    FileName = args.filename
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
    print("Load the model ...")
    model = SentenceTransformer('joyebright/stsb-xlm-r-multilingual-32dim', device='cuda')
    #-------------------------------------
    # Save shuffle for ODD parallel corpora
    if not os.path.exists('OOD.pkl'):
        # Start the multi-process pool on all available CUDA devices
        pool = model.start_multi_process_pool()
        # Compute the embeddings using the multi-process pool
        emb = model.encode_multi_process(OOD_sentences, pool)
        print("Embeddings computed shape:", emb.shape)
        # Optional: Stop the proccesses in the pool
        model.stop_multi_process_pool(pool)
        print("The OOD is being saved into a file named, OOD.pkl")
        with open('OOD.pkl', 'wb') as pickle_file:
            pickle.dump({'source_sentences': source, 'source_embeddings': emb, 'target_sentences': target}, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    #-------------------------------------
    # Loading the pkl file
    with open('OOD.pkl', 'rb') as pickle_load:
        OOD_data = pickle.load(pickle_load)
        OOD_sentences_source = OOD_data['source_sentences']
        OOD_embeddings = OOD_data['source_embeddings']
        OOD_sentences_target = OOD_data['target_sentences']
    #-------------------------------------
    print("A sample of OOD: ", OOD_sentences[0])
    print("A sample of OOD embedding vector: ", OOD_embeddings[0][:])
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
        top_k = min(K, len(OOD_sentences_source[i]))
        dat = pd.DataFrame([])
        
        cols = ['Query']
        for j in range(0, K):
            cols.append('top'+str(j+1))
            cols.append('top'+str(j+1)+'_trg'+str(j+1))
            cols.append('top'+str(j+1)+'_score'+str(j+1))

        dat = pd.DataFrame(columns = cols)
        index = 0
        for query in queries:  
            print(index)  
            index+=1
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.pytorch_cos_sim(query_embedding, OOD_embeddings[i])[0]
            if Dissimilar == False:
                top_results = torch.topk(cos_scores, k=top_k)
            if Dissimilar == True:
                top_results = torch.topk(cos_scores, k=top_k, largest=False)
            print(query)
            print(OOD_sentences_source[i][top_results[1][0]])
            print(OOD_sentences_target[i][top_results[1][0]])

            S =[]
            S.append(query)
            for n in range(0, K):
                S.append(OOD_sentences_source[i][top_results[1][n]])
                S.append(OOD_sentences_target[i][top_results[1][n]])
                S.append("(Score: {:.4f})".format(top_results[0][n]))

            new_S = pd.Series(S, index=dat.columns)
            dat = dat._append(new_S, ignore_index=True)

        print("We are done ...")

        if FileName == None:
            dat.to_csv("final_similar_" + str(i+1)+".csv",index=True)
        else:
            dat.to_csv(FileName + "_" + str(i+1) + ".csv",index=True)
