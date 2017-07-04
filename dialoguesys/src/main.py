import tensorflow as tf
import pickle
import pdb
from utils import generate_batches
from utils import Preprocessing
from configuration import get_configuration
from configuration import print_configuration
from models.basic_model import BasicModel
from models.seq2seq_model import Seq2SeqModel
from models.seq2seq_model import Seq2SeqModelAttention

# Load configuration and create model
config = get_configuration()
model = Seq2SeqModelAttention(config)

# Prepare vocabulary and triples data
preprocess = Preprocessing(config = config)
preprocess.create_vocabulary("Training_Shuffled_Dataset.txt")
preprocess.prepare_data()
# Preprocess Cornell data
preprocess = Preprocessing(train_path_file ="cornell_dataset.txt",
             test_path_file = "Validation_Shuffled_Dataset.txt",
             train_path_file_target = "input_train_cornell",
             test_path_file_target = "input_test_triples",
             triples=False,
             config = config)
preprocess.prepare_data()
# Preprocess Twitter data
preprocess = Preprocessing(train_path_file ="twitter_dataset.txt",
             test_path_file = "Validation_Shuffled_Dataset.txt",
             train_path_file_target ="input_train_twitter",
             test_path_file_target ="input_test_triples",
             triples=False,
             config = config)
preprocess.prepare_data()

# Load input data and train model on the input data
# Create complete training set
triples_data = pickle.load(open((config.data_dir) + '/input_train_triples.pkl', 'rb'))
cornell_data = pickle.load(open((config.data_dir) + '/input_train_cornell.pkl', 'rb'))
twitter_data = pickle.load(open((config.data_dir) + '/input_train_twitter.pkl', 'rb'))
train_data = [triples_data[0] + cornell_data[0] + twitter_data[0],
    triples_data[1] + cornell_data[1] + twitter_data[1]]
eval_data = pickle.load(open((config.data_dir) + '/input_test_triples.pkl', 'rb'))
train_batches = generate_batches(train_data, batch_size=config.batch_size,
                                    num_epochs=config.n_epochs)
eval_batches = generate_batches(eval_data, batch_size=config.batch_size,
                                    num_epochs=1)
# model.train(train_batches, eval_batches, verbose=True)

# # Evaluate perplexity of trained model on validation data
# model_dir = 'runs/1496648182'
# model_file = '/model-12000'
# # print(model.evaluate(eval_batches, model_dir=model_dir, model_file=model_file))
#
# Infer outputs on a subset of test data
model_dir = 'runs/149664818'
model_file = '/model-12000'
checkpoint_file = model_dir + model_file
test_data = pickle.load(open((config.data_dir) + '/input_test.pkl', 'rb'))[0][:10]
predicted_outputs = model.infer(checkpoint_file, test_data)
preprocess.initialize_vocabulary()
# Reverse back the test sentences
test_data = [list(reversed(sentence)) for sentence in test_data]
messages = preprocess.token_ids_to_sentence(test_data)
responses = preprocess.token_ids_to_sentence(predicted_outputs)
# Print to file
with open(model_dir+"/model-12000-conversations", 'w') as f:
    for idx, message in enumerate(messages):
        f.write(message+" ====> "+responses[idx]+"\n")
