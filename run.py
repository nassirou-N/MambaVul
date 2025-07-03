import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow logging

from config.fragment_vectorizer import FragmentVectorizer
from config.models.mamba import MAMBA_Model
import pandas as pd
from config.arg_parser import parameter_parser
import warnings
import numpy as np

# Set print options to display the entire array
np.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")

args = parameter_parser()

for arg in vars(args):
    print(arg, getattr(args, arg))

def parse_file(filename):
    print('parsing file... (', filename, ')')
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 40 in line and fragment:
                yield fragment, fragment_val
                fragment = []
            elif stripped.split()[0].isdigit():
                if fragment:
                    if stripped.isdigit():
                        fragment_val = int(stripped)
                    else:
                        fragment.append(stripped)
            else:
                fragment.append(stripped)

def get_vectors_df(filename, vec_len):
    fragments = []
    count = 0
    vectorizer = FragmentVectorizer(vec_len)

    for fragment, val in parse_file(filename):
        count += 1
        print("Collecting fragments...", count, end="\r")
        vectorizer.add_fragment(fragment)

        row = {"fragment": fragment, "val": val}
        fragments.append(row)
    print('Found {} forward slices and {} backward slices'.format(vectorizer.forward_slices, vectorizer.backward_slices))

    print("Training Word2Vec model...", end="\r")
    vectorizer.train_model()
    print()
    vectors = []
    count = 0

    for fragment in fragments:
        count += 1
        print("Processing fragments...", count, end="\r")
        vector = vectorizer.vectorize(fragment["fragment"])
        row = {"vector": vector, "val": fragment["val"]}
        vectors.append(row)

    # convert vectors to dataframe
    df = pd.DataFrame(vectors)
    return df

def main():
    filename = args.filename        # smart contract source file
    vtype = args.vt                 # vulnerability type (re, ts, io)

    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_fragment_vectors.pkl"
    dataset = "config/train_data/" + vector_filename
    print(f"Dataset path: {dataset}\n")
    
    vector_length = args.vec_length
    
    if os.path.exists(dataset):
        print("Loading existing vector dataframe...")
        df = pd.read_pickle(dataset)
    else:
        print('Generating vector dataframe...')
        df = get_vectors_df(filename, vector_length)
        # Save dataframe to pickle
        os.makedirs("config/train_data", exist_ok=True)
        df.to_pickle(dataset)
        print(f"Saved vectors to {dataset}")

    print(f"Vector shape: {df.iloc[0, 0].shape}")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples: {sum(df.iloc[:, 1])}")
    print(f"Negative samples: {len(df) - sum(df.iloc[:, 1])}")
    
    # Create and train Mamba model
    model = MAMBA_Model(df, args)
    
    print("\n" + "="*50)
    print("MAMBA MODEL TRAINING")
    print("="*50)
    
    model.train()
    
    print("\n" + "="*50)
    print("MAMBA MODEL TESTING")
    print("="*50)
    
    results = model.test()
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()