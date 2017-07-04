from models.seq2seq_model import Seq2SeqModel
from models.seq2seq_model import Seq2SeqModelAttention
from configuration import get_configuration
from utils import Preprocessing

# Initialize the model
config = get_configuration()
preprocess = Preprocessing(config = config)
model = Seq2SeqModelAttention(config)
checkpoint_file = 'runs/baseline-cornell-twitter-attn-dropout/model-18000'

# Launch chat interface
print("*** Hi there. Ask me a question. I will try my best to reply to you with something intelligible.\
 If you think that is not happening, enter \"q\" and quit ***")
query = input(">")
while query != "q":
    # Tokenize the query
    preprocess.initialize_vocabulary()
    token_ids = preprocess.sentence_to_token_ids(query)
    # Reverse the token ids and feed into the RNN
    reverse_token_ids = [list(reversed(token_ids))]
    output_tokens = model.infer(checkpoint_file, reverse_token_ids, verbose=False)
    # Convert token ids back to words and print to output
    output = preprocess.token_ids_to_sentence(output_tokens)
    print(output[0])
    query = input(">")
