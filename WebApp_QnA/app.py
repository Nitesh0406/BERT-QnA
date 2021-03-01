from flask import Flask , request , render_template ,redirect , send_file
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import transformers

import numpy as np

app = Flask(__name__)

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response

@app.route('/',methods = ['GET','POST'])
def get_ans():
    if request.method == 'POST':
        passage = request.form.get('paragraph')
        question = request.form.get('question')
    
        input_ids = tokenizer.encode(question, passage)

        # tokenizer's behavior, let's also get the token strings and display them.
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(tokenizer.sep_token_id)

        # The number of segment A tokens includes the [SEP] token istelf.
        num_seg_a = sep_index + 1

        # The remainder are segment B.
        num_seg_b = len(input_ids) - num_seg_a

        # Construct the list of 0s and 1s.
        segment_ids = [0]*num_seg_a + [1]*num_seg_b

        outputs = model(
                torch.tensor([input_ids]), # The tokens representing our input text.
                token_type_ids=torch.tensor([segment_ids]),# The segment IDs to differentiate question from answer_text
                return_dict=True
                    )
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        

        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        # Combine the tokens in the answer and print it out.
        answer = ' '.join(tokens[answer_start:answer_end+1])
        answer = answer.replace(' ##','')
        

        return render_template('index1.html', 
                                ans = answer)
                                
    
    else:
        return render_template('index.html',
                                ans = None)

if __name__ == '__main__':
    app.run(debug=True)

