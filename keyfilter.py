from sentence_transformers import SentenceTransformer, util
import torch
sim_evaluator = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
import re



def compute_sentences_similarity(sents_A, sents_B):
    embeddings1 = sim_evaluator.encode(sents_A, convert_to_tensor=True)
    embeddings2 = sim_evaluator.encode(sents_B, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)

    for i in range(len(sents_A)):
        digits1 = re.findall(r'\d+', sents_A[i])
        for j in range(i, len(sents_B)):
            digits2 = re.findall(r'\d+', sents_B[j])
            if i == j:
                cosine_score[i, j] = 0
            elif digits1 != digits2:
                cosine_score[i, j] = 0
                cosine_score[j, i] = 0

    return cosine_score

def read_data(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip()
            datas.append(data_line)
    return datas

if __name__ == "__main__":
    path = "data.txt"
    datas = read_data(path)
    num_keys = len(datas)
    print("Number of data lines before filtering: {}".format(num_keys))
    similarity_matrix = compute_sentences_similarity(datas, datas)

    output = "output.txt"
    file = open(output, 'w')

    seen = set()
    outputs = list()
    for i in range(len(datas)):
        if i not in seen:
            outputs.append(datas[i])
        seen.add(i)
        for j in range(i, len(datas)):
            if similarity_matrix[i, j] > 0.9:
                seen.add(j)

    
    print("Number of data lines after filtering: {}".format(outputs))

    for line in outputs:
        file.write("{}\n".format(line))
    print("DONE!")
